import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import build_canonical_args, load_dataset, resolve_save_dir
from eval_causal_qwen3 import (
    apply_setwise_selector,
    build_config,
    build_doc_text_to_chunk_id,
    compute_slice_metrics,
    get_gold_answers,
    get_gold_docs,
)
from requirement_beam_utils import (
    DEFAULT_NEED_UNIT_BRIDGE_ALPHA_BY_TYPE,
    DEFAULT_NEED_UNIT_CONTRADICTION_BETA,
    DEFAULT_REQUIREMENT_CF_TAU,
    DEFAULT_REQUIREMENT_SMOOTH_TAU,
    NEED_UNIT_ATOMIC_FEATURE_NAMES,
    NEED_UNIT_ATOMIC_LABELS,
    NEED_UNIT_ATOMIC_SCORER_VERSION,
    NEED_UNIT_CACHE_VERSION,
    NEED_UNIT_MATCHER_FEATURE_NAMES,
    derive_need_unit_atomic_pseudo_label,
    derive_need_unit_atomic_weak_label,
    extract_need_unit_atomic_features,
    get_counterfactual_sets,
    get_positive_units,
    is_need_unit_cache_version,
    load_requirement_cache,
    requirement_feature_rows_to_matrix,
    resolve_requirement_cache_entry,
    score_need_unit_support,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import QuerySolution, string_to_bool
from train_requirement_setwise import (
    build_requirement_training_rows,
    compute_binary_metrics,
    compute_bucket_qa_summary,
    compute_full_support_rate,
    instantiate_model,
    split_indices,
    subset_list,
    subset_query_solutions,
)


def _split_doc_text(doc_text: str) -> tuple[str, str]:
    return (str(doc_text).split("\n", 1) + [""])[:2]


def _load_label_overrides(path: str) -> dict[str, str]:
    resolved_path = str(path or "").strip()
    if not resolved_path:
        return {}
    source = Path(resolved_path)
    if not source.exists():
        raise ValueError(f"Atomic label override file not found: {resolved_path}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text())
        if isinstance(payload, dict):
            return {
                str(sample_id): str(label)
                for sample_id, label in payload.items()
                if str(label) in NEED_UNIT_ATOMIC_LABELS
            }
        raise ValueError("Atomic label override JSON must be a mapping from sample_id to label")

    overrides: dict[str, str] = {}
    for line in source.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        sample_id = str(payload.get("sample_id", "")).strip()
        label = str(payload.get("label", payload.get("llm_label", payload.get("final_label", "")))).strip()
        if sample_id and label in NEED_UNIT_ATOMIC_LABELS:
            overrides[sample_id] = label
    return overrides


def _build_doc_text_to_entities_map(corpus: list[dict],
                                    hipporag: HippoRAG,
                                    doc_text_to_chunk_id: dict[str, str]) -> dict[str, list[str]]:
    doc_text_to_entities: dict[str, list[str]] = {}
    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}
    passage_node_key_to_doc_idx = getattr(hipporag, "passage_node_key_to_doc_idx", {}) or {}
    for doc in corpus:
        doc_text = f"{doc['title']}\n{doc['text']}"
        chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
        mapped_doc_id = passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
        if mapped_doc_id is None:
            doc_text_to_entities[doc_text] = []
            continue
        doc_text_to_entities[doc_text] = list(
            hipporag.doc_idx_to_structure_entities.get(int(mapped_doc_id), set())
        )
    return doc_text_to_entities


def _build_sample_id(question_key: str,
                     pool_position: int,
                     unit_id: str,
                     cf_id: str = "") -> str:
    prefix = f"{question_key}:p{int(pool_position)}:{unit_id}"
    return f"{prefix}:{cf_id}" if cf_id else prefix


def build_need_unit_atomic_training_rows(query_solutions: list[QuerySolution],
                                         gold_docs: list[list[str]],
                                         requirement_cache: dict,
                                         doc_text_to_entities: dict[str, list[str]] | None = None,
                                         label_overrides: dict[str, str] | None = None,
                                         include_counterfactual_samples: bool = True) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    label_overrides = dict(label_overrides or {})
    doc_text_to_entities = dict(doc_text_to_entities or {})
    label_source_counter: Counter[str] = Counter()
    final_label_counter: Counter[str] = Counter()
    bridge_hard_negative_candidates = 0

    for q_idx, qs in enumerate(query_solutions):
        cache_entry = resolve_requirement_cache_entry(requirement_cache, qs.question)
        effective_pool_limit = min(len(qs.docs), int(cache_entry.get("annotation_pool_k", 0) or 0))
        if effective_pool_limit <= 0:
            continue

        annotations_by_position = {
            int(annotation["pool_position"]): annotation
            for annotation in cache_entry.get("doc_annotations", [])
        }
        gold_set = set(gold_docs[q_idx])
        question_key = str(cache_entry.get("question_key", f"q{q_idx}"))

        positive_units = [
            dict(unit)
            for unit in get_positive_units(cache_entry)
            if bool(unit.get("selector_enabled", True))
        ]
        counterfactual_sets = list(get_counterfactual_sets(cache_entry))

        for pool_position in range(effective_pool_limit):
            doc_text = str(qs.docs[pool_position])
            doc_title, doc_body = _split_doc_text(doc_text)
            annotation = annotations_by_position.get(pool_position, {})
            doc_entities = list(annotation.get("doc_entities", []) or doc_text_to_entities.get(doc_text, []))
            is_gold_doc = bool(doc_text in gold_set)

            def append_row(unit: dict, cf_id: str = "", is_counterfactual: bool = False) -> None:
                nonlocal bridge_hard_negative_candidates
                unit_id = str(unit.get("unit_id", unit.get("requirement_id", "")))
                feature_row = extract_need_unit_atomic_features(
                    unit=unit,
                    doc_title=doc_title,
                    doc_body=doc_body,
                    doc_entities=doc_entities,
                )
                heuristic_scores = score_need_unit_support(
                    unit=unit,
                    doc_title=doc_title,
                    doc_body=doc_body,
                    doc_entities=doc_entities,
                    score_mode="heuristic",
                )
                weak_label, weak_weight = derive_need_unit_atomic_weak_label(
                    unit=unit,
                    feature_row=feature_row,
                    heuristic_scores=heuristic_scores,
                    is_gold_doc=is_gold_doc,
                    is_counterfactual=is_counterfactual,
                )
                sample_id = _build_sample_id(
                    question_key=question_key,
                    pool_position=pool_position,
                    unit_id=unit_id,
                    cf_id=cf_id,
                )
                llm_label = str(label_overrides.get(sample_id, "")).strip()
                if llm_label in NEED_UNIT_ATOMIC_LABELS:
                    final_label = llm_label
                    label_source = "llm"
                    label_weight = 1.0
                elif weak_label in NEED_UNIT_ATOMIC_LABELS:
                    final_label = str(weak_label)
                    label_source = "weak"
                    label_weight = float(weak_weight)
                else:
                    final_label = derive_need_unit_atomic_pseudo_label(heuristic_scores)
                    label_source = "heuristic"
                    label_weight = 0.4
                if (
                    not is_gold_doc
                    and float(feature_row.get("bridge_feature_count", 0.0)) >= 1.0
                    and float(heuristic_scores.get("alignment_score", 0.0) or 0.0) >= 0.35
                    and float(heuristic_scores.get("contradiction_prob", 0.0) or 0.0) < 0.2
                    and not is_counterfactual
                ):
                    bridge_hard_negative_candidates += 1
                rows.append({
                    "sample_id": sample_id,
                    "dataset": "",
                    "question_id": question_key,
                    "query_index": int(q_idx),
                    "unit_id": unit_id,
                    "cf_id": str(cf_id),
                    "is_counterfactual": bool(is_counterfactual),
                    "question": qs.question,
                    "need_unit": dict(unit),
                    "doc_title": doc_title,
                    "doc_body": doc_body,
                    "doc_entities": list(doc_entities),
                    "features": dict(feature_row),
                    "weak_label": weak_label,
                    "llm_label": llm_label or None,
                    "final_label": final_label,
                    "label_source": label_source,
                    "label_weight": float(label_weight),
                    "heuristic_scores": {
                        key: float(value) if isinstance(value, (int, float)) else value
                        for key, value in heuristic_scores.items()
                    },
                })
                label_source_counter[label_source] += 1
                final_label_counter[final_label] += 1

            for unit in positive_units:
                append_row(unit=unit, cf_id="", is_counterfactual=False)

            if not include_counterfactual_samples:
                continue
            for cf_set in counterfactual_sets:
                cf_id = str(cf_set.get("cf_id", ""))
                for unit in cf_set.get("requirements", []):
                    if not bool(unit.get("selector_enabled", True)):
                        continue
                    append_row(unit=dict(unit), cf_id=cf_id, is_counterfactual=True)

    summary = {
        "row_count": int(len(rows)),
        "include_counterfactual_samples": bool(include_counterfactual_samples),
        "label_source_counts": dict(label_source_counter),
        "final_label_counts": dict(final_label_counter),
        "bridge_hard_negative_candidates": int(bridge_hard_negative_candidates),
    }
    return rows, summary


def _build_atomic_sample_weights(rows: list[dict]) -> np.ndarray:
    label_counts = Counter(str(row["final_label"]) for row in rows)
    class_count = max(1, len(label_counts))
    total = max(1, len(rows))
    weights = []
    for row in rows:
        label = str(row["final_label"])
        class_weight = float(total / float(class_count * max(1, label_counts[label])))
        weights.append(float(row.get("label_weight", 1.0)) * class_weight)
    return np.asarray(weights, dtype=float)


def compute_multiclass_metrics(labels: np.ndarray,
                               probability_matrix: np.ndarray,
                               class_names: list[str]) -> dict:
    metrics = {
        "row_count": int(labels.size),
        "class_names": list(class_names),
    }
    if labels.size == 0:
        return metrics
    predicted_indices = np.argmax(probability_matrix, axis=1)
    predicted_labels = np.asarray([class_names[idx] for idx in predicted_indices], dtype=object)
    metrics["accuracy"] = round(float(np.mean(predicted_labels == labels)), 4)
    gold_counts = Counter(str(label) for label in labels.tolist())
    pred_counts = Counter(str(label) for label in predicted_labels.tolist())
    metrics["gold_label_counts"] = dict(gold_counts)
    metrics["predicted_label_counts"] = dict(pred_counts)
    return metrics


def _save_rows_jsonl(rows_path: Path, rows: list[dict]) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_legacy_binary_training(args, logger, resolved_save_dir, corpus, samples, requirement_cache, docs, queries,
                               gold_docs, gold_answers, doc_text_to_chunk_id, train_indices, eval_indices,
                               query_solutions, output_dir, model_path, output_json) -> None:
    train_query_solutions = subset_query_solutions(query_solutions, train_indices)
    eval_query_solutions = subset_query_solutions(query_solutions, eval_indices)
    train_gold_docs = subset_list(gold_docs, train_indices)
    eval_gold_docs = subset_list(gold_docs, eval_indices)
    eval_gold_answers = subset_list(gold_answers, eval_indices)

    train_rows, train_row_summary = build_requirement_training_rows(
        query_solutions=train_query_solutions,
        gold_docs=train_gold_docs,
        requirement_cache=requirement_cache,
        pool_k=int(args.setwise_pool_k),
        qa_top_k=int(args.qa_top_k),
        anchor_count=int(args.setwise_anchor_count),
        reserve_top_m=int(args.setwise_reserve_top_m),
        max_bridge_slots=int(args.setwise_max_bridge_slots),
        smooth_tau=float(args.smooth_tau),
        counterfactual_tau=float(args.counterfactual_tau),
        training_focus=args.training_focus,
    )
    if not train_rows:
        raise ValueError("No need-unit training rows were generated; check the cache and split sizes")

    train_matrix = requirement_feature_rows_to_matrix(
        train_rows,
        feature_names=NEED_UNIT_MATCHER_FEATURE_NAMES,
    )
    train_labels = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    model = instantiate_model(args.model_type, args.split_seed)
    model.fit(train_matrix, train_labels)
    train_probabilities = model.predict_proba(train_matrix)[:, 1]
    train_metrics = compute_binary_metrics(train_labels, train_probabilities)

    model_bundle = {
        "model": model,
        "feature_names": list(NEED_UNIT_MATCHER_FEATURE_NAMES),
        "model_type": args.model_type,
        "dataset": args.dataset,
        "save_dir": resolved_save_dir,
        "pool_k": int(args.setwise_pool_k),
        "qa_top_k": int(args.qa_top_k),
        "anchor_count": int(args.setwise_anchor_count),
        "reserve_top_m": int(args.setwise_reserve_top_m),
        "max_bridge_slots": int(args.setwise_max_bridge_slots),
        "training_focus": args.training_focus,
        "split_seed": int(args.split_seed),
        "smooth_tau": float(args.smooth_tau),
        "counterfactual_tau": float(args.counterfactual_tau),
        "requirement_cache_path": str(args.requirement_cache_path),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_row_summary": train_row_summary,
        "train_metrics": train_metrics,
    }
    joblib.dump(model_bundle, model_path)
    logger.info("Saved need-unit matcher bundle to %s", model_path)

    baseline_retrieval = compute_slice_metrics(
        config=build_config(build_canonical_args(args, resolved_save_dir), corpus_len=len(corpus)),
        query_solutions=eval_query_solutions,
        gold_docs=eval_gold_docs,
        gold_answers=None,
    )
    requirement_selector_bundle = {
        "cache": requirement_cache,
        "cache_path": str(args.requirement_cache_path),
        "mode": "learned",
        "model_bundle": model_bundle,
        "model_path": str(model_path),
        "smooth_tau": float(args.smooth_tau),
        "counterfactual_tau": float(args.counterfactual_tau),
    }
    config = build_config(build_canonical_args(args, resolved_save_dir), corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    selected_eval_solutions, selector_summary = apply_setwise_selector(
        hipporag=hipporag,
        query_solutions=eval_query_solutions,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
        pool_k=int(args.setwise_pool_k),
        qa_top_k=int(args.qa_top_k),
        selector_name="requirement_beam",
        score_mode="bridge",
        anchor_count=int(args.setwise_anchor_count),
        reserve_top_m=int(args.setwise_reserve_top_m),
        max_bridge_slots=int(args.setwise_max_bridge_slots),
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        learned_model_bundle=None,
        beam_width=4,
        beam_expand_per_state=4,
        beam_projected_shortlist_factor=1,
        non_anchor_title_dedup=True,
        query_entity_source="seed",
        gate_mode="none",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
        requirement_selector_bundle=requirement_selector_bundle,
    )
    selector_retrieval = compute_slice_metrics(
        config=config,
        query_solutions=selected_eval_solutions,
        gold_docs=eval_gold_docs,
        gold_answers=None,
    )
    baseline_fs5 = compute_full_support_rate(eval_query_solutions, eval_gold_docs, k=int(args.qa_top_k))
    selector_fs5 = compute_full_support_rate(selected_eval_solutions, eval_gold_docs, k=int(args.qa_top_k))

    baseline_qa_results = None
    selector_qa_results = None
    bucket_qa_summary = {}
    if string_to_bool(args.eval_with_qa):
        baseline_eval_solutions, _, _, _, baseline_qa_results = hipporag.rag_qa(
            queries=[QuerySolution(
                question=qs.question,
                docs=list(qs.docs),
                doc_scores=np.asarray(qs.doc_scores, dtype=float) if qs.doc_scores is not None else None,
                gold_docs=qs.gold_docs,
                gold_answers=qs.gold_answers,
                retrieval_trace=qs.retrieval_trace,
                qa_trace=qs.qa_trace,
            ) for qs in eval_query_solutions],
            gold_docs=eval_gold_docs,
            gold_answers=eval_gold_answers,
        )
        selected_eval_solutions, _, _, _, selector_qa_results = hipporag.rag_qa(
            queries=selected_eval_solutions,
            gold_docs=eval_gold_docs,
            gold_answers=eval_gold_answers,
        )
        bucket_qa_summary = compute_bucket_qa_summary(
            config=config,
            baseline_solutions=baseline_eval_solutions,
            selector_solutions=selected_eval_solutions,
            gold_docs=eval_gold_docs,
            gold_answers=eval_gold_answers,
        )

    report = {
        "dataset": args.dataset,
        "limit": len(samples),
        "resolved_save_dir": resolved_save_dir,
        "need_unit_cache_path": str(args.requirement_cache_path),
        "split": {
            "train_query_count": len(train_indices),
            "eval_query_count": len(eval_indices),
            "split_seed": int(args.split_seed),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        },
        "selector_config": {
            "selector": "requirement_beam",
            "mode": "learned",
            "schema_version": NEED_UNIT_CACHE_VERSION,
            "model_type": args.model_type,
            "model_path": str(model_path),
            "pool_k": int(args.setwise_pool_k),
            "qa_top_k": int(args.qa_top_k),
            "anchor_count": int(args.setwise_anchor_count),
            "reserve_top_m": int(args.setwise_reserve_top_m),
            "max_bridge_slots": int(args.setwise_max_bridge_slots),
            "training_focus": args.training_focus,
            "smooth_tau": float(args.smooth_tau),
            "counterfactual_tau": float(args.counterfactual_tau),
        },
        "training": {
            **train_row_summary,
            "train_metrics": train_metrics,
        },
        "eval_retrieval": {
            "baseline": baseline_retrieval["overall"],
            "need_unit_beam_learned": selector_retrieval["overall"],
            "baseline_FS@5": round(float(baseline_fs5), 4),
            "need_unit_beam_learned_FS@5": round(float(selector_fs5), 4),
            "FS@5_delta": round(float(selector_fs5 - baseline_fs5), 4),
            "selector_summary": selector_summary,
        },
        "eval_qa": None,
    }
    if baseline_qa_results is not None and selector_qa_results is not None:
        baseline_em = float(baseline_qa_results.get("ExactMatch", 0.0))
        baseline_f1 = float(baseline_qa_results.get("F1", 0.0))
        selector_em = float(selector_qa_results.get("ExactMatch", 0.0))
        selector_f1 = float(selector_qa_results.get("F1", 0.0))
        report["eval_qa"] = {
            "baseline": baseline_qa_results,
            "need_unit_beam_learned": selector_qa_results,
            "EM_delta": round(selector_em - baseline_em, 4),
            "F1_delta": round(selector_f1 - baseline_f1, 4),
            "per_bucket": bucket_qa_summary,
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))


def run_atomic_multiclass_training(args, logger, resolved_save_dir, corpus, samples, requirement_cache, docs, queries,
                                   gold_docs, doc_text_to_chunk_id, train_indices, eval_indices,
                                   query_solutions, output_dir, model_path, output_json) -> None:
    config = build_config(build_canonical_args(args, resolved_save_dir), corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    doc_text_to_entities = _build_doc_text_to_entities_map(
        corpus=corpus,
        hipporag=hipporag,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
    )
    label_overrides = _load_label_overrides(args.label_overrides_path)
    include_counterfactual_samples = string_to_bool(args.include_counterfactual_samples)

    train_query_solutions = subset_query_solutions(query_solutions, train_indices)
    eval_query_solutions = subset_query_solutions(query_solutions, eval_indices)
    train_gold_docs = subset_list(gold_docs, train_indices)
    eval_gold_docs = subset_list(gold_docs, eval_indices)

    train_rows, train_row_summary = build_need_unit_atomic_training_rows(
        query_solutions=train_query_solutions,
        gold_docs=train_gold_docs,
        requirement_cache=requirement_cache,
        doc_text_to_entities=doc_text_to_entities,
        label_overrides=label_overrides,
        include_counterfactual_samples=include_counterfactual_samples,
    )
    eval_rows, eval_row_summary = build_need_unit_atomic_training_rows(
        query_solutions=eval_query_solutions,
        gold_docs=eval_gold_docs,
        requirement_cache=requirement_cache,
        doc_text_to_entities=doc_text_to_entities,
        label_overrides=label_overrides,
        include_counterfactual_samples=include_counterfactual_samples,
    )
    if not train_rows:
        raise ValueError("No atomic need-unit training rows were generated; check the cache and retrieval pool")

    rows_path = Path(args.rows_path) if args.rows_path else model_path.with_suffix(".rows.jsonl")
    _save_rows_jsonl(rows_path, train_rows + eval_rows)

    train_matrix = requirement_feature_rows_to_matrix(
        [row["features"] for row in train_rows],
        feature_names=NEED_UNIT_ATOMIC_FEATURE_NAMES,
    )
    train_labels = np.asarray([str(row["final_label"]) for row in train_rows], dtype=object)
    train_weights = _build_atomic_sample_weights(train_rows)
    model = instantiate_model(args.model_type, args.split_seed)
    model.fit(train_matrix, train_labels, sample_weight=train_weights)
    train_probabilities = model.predict_proba(train_matrix)
    train_metrics = compute_multiclass_metrics(
        labels=train_labels,
        probability_matrix=train_probabilities,
        class_names=[str(label) for label in model.classes_],
    )

    eval_metrics = None
    if eval_rows:
        eval_matrix = requirement_feature_rows_to_matrix(
            [row["features"] for row in eval_rows],
            feature_names=NEED_UNIT_ATOMIC_FEATURE_NAMES,
        )
        eval_labels = np.asarray([str(row["final_label"]) for row in eval_rows], dtype=object)
        eval_probabilities = model.predict_proba(eval_matrix)
        eval_metrics = compute_multiclass_metrics(
            labels=eval_labels,
            probability_matrix=eval_probabilities,
            class_names=[str(label) for label in model.classes_],
        )

    model_bundle = {
        "task": "atomic_multiclass",
        "model": model,
        "feature_names": list(NEED_UNIT_ATOMIC_FEATURE_NAMES),
        "labels": list(NEED_UNIT_ATOMIC_LABELS),
        "scorer_version": NEED_UNIT_ATOMIC_SCORER_VERSION,
        "bridge_alpha_by_unit_type": dict(DEFAULT_NEED_UNIT_BRIDGE_ALPHA_BY_TYPE),
        "beta_contradiction": float(DEFAULT_NEED_UNIT_CONTRADICTION_BETA),
        "model_type": args.model_type,
        "dataset": args.dataset,
        "save_dir": resolved_save_dir,
        "requirement_cache_path": str(args.requirement_cache_path),
        "rows_path": str(rows_path),
        "label_overrides_path": str(args.label_overrides_path or ""),
        "include_counterfactual_samples": include_counterfactual_samples,
        "split_seed": int(args.split_seed),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_row_summary": train_row_summary,
        "eval_row_summary": eval_row_summary,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    joblib.dump(model_bundle, model_path)
    logger.info("Saved atomic need-unit scorer bundle to %s", model_path)

    report = {
        "dataset": args.dataset,
        "limit": len(samples),
        "resolved_save_dir": resolved_save_dir,
        "need_unit_cache_path": str(args.requirement_cache_path),
        "training_task": "atomic_multiclass",
        "rows_path": str(rows_path),
        "split": {
            "train_query_count": len(train_indices),
            "eval_query_count": len(eval_indices),
            "split_seed": int(args.split_seed),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        },
        "model": {
            "model_type": args.model_type,
            "model_path": str(model_path),
            "feature_names": list(NEED_UNIT_ATOMIC_FEATURE_NAMES),
            "labels": list(NEED_UNIT_ATOMIC_LABELS),
            "scorer_version": NEED_UNIT_ATOMIC_SCORER_VERSION,
            "bridge_alpha_by_unit_type": dict(DEFAULT_NEED_UNIT_BRIDGE_ALPHA_BY_TYPE),
            "beta_contradiction": float(DEFAULT_NEED_UNIT_CONTRADICTION_BETA),
        },
        "training": {
            **train_row_summary,
            "metrics": train_metrics,
        },
        "eval": {
            **eval_row_summary,
            "metrics": eval_metrics,
        },
        "note": "This script trains the atomic doc-by-need-unit scorer only. Rebuild doc annotations with annotate_need_unit_support.py before rerunning selector smokes.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train need-unit scorers for requirement_beam.")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--train_queries", type=int, default=160)
    parser.add_argument("--eval_queries", type=int, default=40)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
    parser.add_argument("--requirement_cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="research_memory/emnlp_expand_then_compose/models")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_type", choices=["logistic_regression", "hist_gbdt"], default="hist_gbdt")
    parser.add_argument("--training_task", choices=["atomic_multiclass", "legacy_binary"], default="atomic_multiclass")
    parser.add_argument("--training_focus", choices=["all", "missing_topk", "promote_missing_only"], default="missing_topk")
    parser.add_argument("--label_overrides_path", type=str, default="")
    parser.add_argument("--rows_path", type=str, default="")
    parser.add_argument("--include_counterfactual_samples", type=str, default="true")
    parser.add_argument("--setwise_pool_k", type=int, default=100)
    parser.add_argument("--setwise_anchor_count", type=int, default=1)
    parser.add_argument("--setwise_reserve_top_m", type=int, default=1)
    parser.add_argument("--setwise_max_bridge_slots", type=int, default=0)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--smooth_tau", type=float, default=DEFAULT_REQUIREMENT_SMOOTH_TAU)
    parser.add_argument("--counterfactual_tau", type=float, default=DEFAULT_REQUIREMENT_CF_TAU)
    parser.add_argument("--eval_with_qa", type=str, default="true")
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    resolved_save_dir = resolve_save_dir(args.save_dir, args.dataset)
    corpus, samples = load_dataset(args.dataset, args.limit)
    if len(samples) < 2:
        raise ValueError("Need at least two samples to build a train/eval split")

    requirement_cache = load_requirement_cache(args.requirement_cache_path)
    if not is_need_unit_cache_version(requirement_cache):
        raise ValueError(
            f"train_need_unit_scorer expects a V2 need-unit cache, got {requirement_cache.get('version')}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_stem = (
        f"{args.dataset}_need_unit_{args.training_task}_{args.model_type}_pool{args.setwise_pool_k}_limit{len(samples)}"
    )
    model_path = Path(args.model_path) if args.model_path else (output_dir / f"{model_stem}.joblib")
    output_json = Path(args.output_json) if args.output_json else model_path.with_suffix(".json")

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples, args.dataset, corpus=corpus)
    gold_answers = get_gold_answers(samples)
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)

    train_indices, eval_indices = split_indices(len(samples), args.train_queries, args.eval_queries, args.split_seed)
    if not train_indices or not eval_indices:
        raise ValueError("Train/eval split is empty; increase --limit or adjust --train_queries/--eval_queries")

    config = build_config(build_canonical_args(args, resolved_save_dir), corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    logger.info(
        "Retrieving %d queries for need-unit scorer training/eval on %s (train=%d, eval=%d)",
        len(queries),
        args.dataset,
        len(train_indices),
        len(eval_indices),
    )
    query_solutions, _ = hipporag.retrieve(
        queries=queries,
        gold_docs=gold_docs,
    )

    if args.training_task == "legacy_binary":
        run_legacy_binary_training(
            args=args,
            logger=logger,
            resolved_save_dir=resolved_save_dir,
            corpus=corpus,
            samples=samples,
            requirement_cache=requirement_cache,
            docs=docs,
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
            train_indices=train_indices,
            eval_indices=eval_indices,
            query_solutions=query_solutions,
            output_dir=output_dir,
            model_path=model_path,
            output_json=output_json,
        )
        return

    run_atomic_multiclass_training(
        args=args,
        logger=logger,
        resolved_save_dir=resolved_save_dir,
        corpus=corpus,
        samples=samples,
        requirement_cache=requirement_cache,
        docs=docs,
        queries=queries,
        gold_docs=gold_docs,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
        train_indices=train_indices,
        eval_indices=eval_indices,
        query_solutions=query_solutions,
        output_dir=output_dir,
        model_path=model_path,
        output_json=output_json,
    )


if __name__ == "__main__":
    main()
