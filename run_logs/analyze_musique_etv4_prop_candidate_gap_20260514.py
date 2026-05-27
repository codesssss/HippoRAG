#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Iterable

ROOT = Path("/mnt/nvme/code/HippoRAG")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_query_grounded_sto_fresh_e2e import embedding_endpoint_model_id, normalize_embedding_name_for_runtime
from evidence_transition_graphragv4_fact_witnessed_sto.source_authorized_vocab_strict_retrieval.candidate_generator import (
    DenseEmbeddingCandidateGenerator,
)
from evidence_transition_graphragv4_fact_witnessed_sto.source_authorized_vocab_strict_retrieval.evaluate_report import (
    load_nodes,
)


METHOD = "evidence_transition_graphragv4_fact_witnessed_sto"
ETV4 = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
PROP_POOL = ROOT / "run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/pools/proprag/musique_proprag_qwen32b_nothink_pool200.json"
PROP_READER_INPUT = ROOT / "run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/reader_inputs/proprag/musique_proprag_qwen32b_nothink_pool200_reader_input.json"
OPENIE = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/openie_results_ner_qwen3-32b-judge.json"
EMBEDDINGS = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/qwen3-32b-judge_VLLM_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet"
OUT_DIR = ROOT / "run_logs/etv4_clean_root_cause_20260513"
OUT_JSON = OUT_DIR / "musique_etv4_prop_candidate_gap_20260514.json"
OUT_MD = OUT_DIR / "musique_etv4_prop_candidate_gap_20260514.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ints(values: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values or []:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def norm(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def title_map() -> dict[int, str]:
    docs = load(OPENIE).get("docs", []) or []
    mapping: dict[int, str] = {}
    for i, doc in enumerate(docs):
        passage = str(doc.get("passage", "") or "")
        title = passage.splitlines()[0].strip() if passage else ""
        mapping[int(i)] = title
    return mapping


def title_set(doc_ids: Iterable[int], titles: dict[int, str]) -> set[str]:
    return {norm(titles.get(int(doc_id), "")) for doc_id in doc_ids if norm(titles.get(int(doc_id), ""))}


def title_all(doc_ids: Iterable[int], gold_ids: Iterable[int], titles: dict[int, str]) -> bool:
    gold_titles = title_set(gold_ids, titles)
    return bool(gold_titles) and gold_titles <= title_set(doc_ids, titles)


def exact_all(doc_ids: Iterable[int], gold_ids: Iterable[int]) -> bool:
    gold = set(ints(gold_ids))
    return bool(gold) and gold <= set(ints(doc_ids))


def title_recall(doc_ids: Iterable[int], gold_ids: Iterable[int], titles: dict[int, str]) -> float:
    gold_titles = title_set(gold_ids, titles)
    if not gold_titles:
        return 0.0
    return len(gold_titles & title_set(doc_ids, titles)) / len(gold_titles)


def exact_recall(doc_ids: Iterable[int], gold_ids: Iterable[int]) -> float:
    gold = set(ints(gold_ids))
    if not gold:
        return 0.0
    return len(gold & set(ints(doc_ids))) / len(gold)


def prop_records_by_query() -> dict[int, dict[str, Any]]:
    payload = load(PROP_POOL)
    return {int(row["query_idx"]): row for row in payload["records"]}


def prop_top5_by_query() -> dict[int, list[int]]:
    payload = load(PROP_READER_INPUT)
    return {int(row["query_index"]): ints(row.get("retrieved_doc_ids", []))[:5] for row in payload["examples"]}


def etv4_pool(row: dict[str, Any]) -> list[int]:
    return ints(((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get("candidate_doc_indices", []))[:200]


def etv4_trace_docs(row: dict[str, Any], key: str) -> list[int]:
    return ints(((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get(key, []))


def rank_map(doc_ids: Iterable[int]) -> dict[int, int]:
    return {int(doc_id): rank for rank, doc_id in enumerate(ints(doc_ids), start=1)}


def title_rank(doc_ids: Iterable[int], wanted_title: str, titles: dict[int, str]) -> int | None:
    target = norm(wanted_title)
    for rank, doc_id in enumerate(ints(doc_ids), start=1):
        if norm(titles.get(int(doc_id), "")) == target:
            return rank
    return None


def classify_missing_title(
    *,
    missing_title: str,
    exact_gold_docs_for_title: list[int],
    et_pool: list[int],
    et_prefix: list[int],
    et_seeds: list[int],
    et_admitted: list[int],
    et_graph_tail: list[int],
    dense200: list[int],
    prop_pool: list[int],
    titles: dict[int, str],
) -> str:
    in_dense = norm(missing_title) in title_set(dense200, titles)
    in_prop = norm(missing_title) in title_set(prop_pool, titles)
    in_seed = norm(missing_title) in title_set(et_seeds, titles)
    in_admitted = norm(missing_title) in title_set(et_admitted, titles)
    in_graph_tail = norm(missing_title) in title_set(et_graph_tail, titles)
    in_prefix = norm(missing_title) in title_set(et_prefix, titles)
    in_pool = norm(missing_title) in title_set(et_pool, titles)
    exact_in_dense = bool(set(exact_gold_docs_for_title) & set(dense200))
    exact_in_pool = bool(set(exact_gold_docs_for_title) & set(et_pool))

    if in_pool or exact_in_pool:
        return "not_missing_by_title_check"
    if in_prefix:
        return "prefix_accounting_mismatch"
    if in_dense and not in_pool:
        return "dense200_recalled_but_displaced_by_graph_budget"
    if exact_in_dense and not exact_in_pool:
        return "exact_gold_dense200_recalled_but_displaced"
    if in_seed and not in_admitted:
        return "seeded_but_not_graph_admitted"
    if in_admitted or in_graph_tail:
        return "graph_admitted_but_cut_or_title_mismatch"
    if in_prop and not in_dense:
        return "prop_specific_candidate_generation"
    if not in_prop and not in_dense:
        return "absent_from_prop_and_dense_title_universe"
    return "other"


def build_dense_generator() -> DenseEmbeddingCandidateGenerator:
    nodes = load_nodes(OPENIE)
    embedding_model = embedding_endpoint_model_id(
        normalize_embedding_name_for_runtime(
            embedding_name="VLLM/nvidia/NV-Embed-v2",
            embedding_base_url="http://localhost:8019/v1/embeddings",
        )
    )
    return DenseEmbeddingCandidateGenerator.from_parquet(
        nodes=tuple(nodes),
        embedding_store_path=EMBEDDINGS,
        embedding_base_url="http://localhost:8019/v1/embeddings",
        embedding_model_name=embedding_model,
    )


def dense_topk_by_query(
    *,
    dense: DenseEmbeddingCandidateGenerator,
    et_rows: list[dict[str, Any]],
    query_indices: list[int],
    top_k: int = 200,
) -> dict[int, list[int]]:
    import numpy as np

    queries = [str(et_rows[qi]["question"]) for qi in query_indices]
    dense.prepare_queries(queries)
    matrix = np.asarray(dense.embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"unexpected dense matrix shape: {matrix.shape}")

    result: dict[int, list[int]] = {}
    for qi in query_indices:
        query_embedding = np.asarray(dense.query_encoder(str(et_rows[qi]["question"])), dtype=float)
        if query_embedding.ndim != 1:
            query_embedding = np.squeeze(query_embedding)
        if query_embedding.ndim != 1 or matrix.shape[1] != query_embedding.shape[0]:
            raise ValueError(
                "dense embedding generator dimension mismatch: "
                f"matrix={matrix.shape}, query={query_embedding.shape}"
            )
        scores = np.dot(matrix, query_embedding.T)
        ranked_positions = np.argsort(scores)[::-1][: max(int(top_k), 1)]
        result[qi] = [int(dense.doc_indices[int(position)]) for position in ranked_positions]
    return result


def main() -> int:
    titles = title_map()
    et_rows = load(ETV4)["rows"]
    prop_pool_by_q = prop_records_by_query()
    prop_top5 = prop_top5_by_query()

    prop_pool_only_queries: list[int] = []
    prop_top5_only_queries: list[int] = []
    etv4_top5_only_queries: list[int] = []
    for row in et_rows:
        qi = int(row["query_index"])
        gold = ints(row["gold_doc_indices"])
        et_top5 = ints(row["retrieved_doc_indices_top5"])
        et_pool = etv4_pool(row)
        prop_pool = ints(prop_pool_by_q[qi].get("pool_doc_ids", []))[:200]
        prop5 = prop_top5.get(qi, [])
        if title_all(prop_pool, gold, titles) and not title_all(et_pool, gold, titles):
            prop_pool_only_queries.append(qi)
        if title_all(prop5, gold, titles) and not title_all(et_top5, gold, titles):
            prop_top5_only_queries.append(qi)
        if title_all(et_top5, gold, titles) and not title_all(prop5, gold, titles):
            etv4_top5_only_queries.append(qi)

    query_indices_for_dense = sorted(set(prop_pool_only_queries) | set(prop_top5_only_queries))
    dense = build_dense_generator()
    dense200_by_q = dense_topk_by_query(
        dense=dense,
        et_rows=et_rows,
        query_indices=query_indices_for_dense,
        top_k=200,
    )

    missing_title_events: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    dense_displaced_exact_events = 0
    dense_displaced_title_queries: set[int] = set()
    prop_specific_queries: set[int] = set()
    seed_not_admitted_queries: set[int] = set()

    for qi in prop_pool_only_queries:
        row = et_rows[qi]
        gold = ints(row["gold_doc_indices"])
        gold_titles = title_set(gold, titles)
        et_pool = etv4_pool(row)
        et_pool_titles = title_set(et_pool, titles)
        missing_titles = sorted(gold_titles - et_pool_titles)
        prop_pool = ints(prop_pool_by_q[qi].get("pool_doc_ids", []))[:200]
        dense200 = dense200_by_q[qi]
        et_prefix = etv4_trace_docs(row, "source_prior_prefix_doc_indices")
        et_seeds = etv4_trace_docs(row, "agsto_seed_doc_indices")
        et_admitted = etv4_trace_docs(row, "admissible_doc_indices")
        et_graph_tail = etv4_trace_docs(row, "agsto_graph_tail_doc_indices")
        gold_docs_by_title: dict[str, list[int]] = defaultdict(list)
        for doc_id in gold:
            gold_docs_by_title[norm(titles.get(doc_id, ""))].append(doc_id)

        for missing_title in missing_titles:
            exact_gold_docs = gold_docs_by_title.get(norm(missing_title), [])
            label = classify_missing_title(
                missing_title=missing_title,
                exact_gold_docs_for_title=exact_gold_docs,
                et_pool=et_pool,
                et_prefix=et_prefix,
                et_seeds=et_seeds,
                et_admitted=et_admitted,
                et_graph_tail=et_graph_tail,
                dense200=dense200,
                prop_pool=prop_pool,
                titles=titles,
            )
            classification_counts[label] += 1
            if "dense200" in label or label == "exact_gold_dense200_recalled_but_displaced":
                dense_displaced_title_queries.add(qi)
            if label == "exact_gold_dense200_recalled_but_displaced":
                dense_displaced_exact_events += 1
            if label == "prop_specific_candidate_generation":
                prop_specific_queries.add(qi)
            if label == "seeded_but_not_graph_admitted":
                seed_not_admitted_queries.add(qi)
            missing_title_events.append(
                {
                    "query_index": qi,
                    "question": row["question"],
                    "missing_title": missing_title,
                    "gold_doc_indices_for_title": exact_gold_docs,
                    "classification": label,
                    "etv4_top5": ints(row["retrieved_doc_indices_top5"]),
                    "prop_top5": prop_top5.get(qi, []),
                    "etv4_pool_title_rank": title_rank(et_pool, missing_title, titles),
                    "dense200_title_rank": title_rank(dense200, missing_title, titles),
                    "prop_pool_title_rank": title_rank(prop_pool, missing_title, titles),
                    "etv4_seed_title_rank": title_rank(et_seeds, missing_title, titles),
                    "etv4_admitted_title_rank": title_rank(et_admitted, missing_title, titles),
                    "etv4_graph_tail_title_rank": title_rank(et_graph_tail, missing_title, titles),
                }
            )

    prop_top5_only_breakdown: Counter[str] = Counter()
    prop_top5_only_details: list[dict[str, Any]] = []
    for qi in prop_top5_only_queries:
        row = et_rows[qi]
        gold = ints(row["gold_doc_indices"])
        et_pool = etv4_pool(row)
        dense200 = dense200_by_q[qi]
        prop_pool = ints(prop_pool_by_q[qi].get("pool_doc_ids", []))[:200]
        if not title_all(et_pool, gold, titles):
            bucket = "etv4_pool_missing_title_gold"
        elif title_all(dense200, gold, titles):
            bucket = "etv4_pool_has_gold_readout_miss_dense_also_has"
        else:
            bucket = "etv4_pool_has_gold_readout_miss_graph_needed"
        prop_top5_only_breakdown[bucket] += 1
        prop_top5_only_details.append(
            {
                "query_index": qi,
                "question": row["question"],
                "bucket": bucket,
                "etv4_title_recall_top5": title_recall(ints(row["retrieved_doc_indices_top5"]), gold, titles),
                "prop_title_recall_top5": title_recall(prop_top5.get(qi, []), gold, titles),
                "etv4_title_all_pool200": title_all(et_pool, gold, titles),
                "dense_title_all200": title_all(dense200, gold, titles),
                "prop_title_all_pool200": title_all(prop_pool, gold, titles),
            }
        )

    summary = {
        "dataset": "musique",
        "count": len(et_rows),
        "prop_pool_title_all200_only_query_count": len(prop_pool_only_queries),
        "prop_top5_title_all_only_query_count": len(prop_top5_only_queries),
        "etv4_top5_title_all_only_query_count": len(etv4_top5_only_queries),
        "prop_pool_only_missing_title_event_count": len(missing_title_events),
        "prop_pool_only_missing_title_classification": dict(classification_counts),
        "prop_pool_only_dense_displaced_title_query_count": len(dense_displaced_title_queries),
        "prop_pool_only_prop_specific_query_count": len(prop_specific_queries),
        "prop_pool_only_seed_not_admitted_query_count": len(seed_not_admitted_queries),
        "prop_top5_only_breakdown": dict(prop_top5_only_breakdown),
        "mean_prop_pool_rank_for_missing_title": mean(
            event["prop_pool_title_rank"] for event in missing_title_events if event["prop_pool_title_rank"] is not None
        )
        if any(event["prop_pool_title_rank"] is not None for event in missing_title_events)
        else None,
        "mean_dense_rank_for_missing_title_when_present": mean(
            event["dense200_title_rank"] for event in missing_title_events if event["dense200_title_rank"] is not None
        )
        if any(event["dense200_title_rank"] is not None for event in missing_title_events)
        else None,
    }
    payload = {
        "summary": summary,
        "prop_pool_only_queries": prop_pool_only_queries,
        "prop_top5_only_queries": prop_top5_only_queries,
        "etv4_top5_only_queries": etv4_top5_only_queries,
        "missing_title_events": missing_title_events,
        "prop_top5_only_details": prop_top5_only_details,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# MuSiQue ETv4 Clean vs PropRAG Candidate Gap",
        "",
        "Diagnostic-only. Gold labels are used only for offline failure classification.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            text = f"{value:.2f}"
        else:
            text = str(value)
        lines.append(f"| {key} | {text} |")
    lines.extend(
        [
            "",
            "## Missing Title Event Classification",
            "",
            "| Class | Count |",
            "|---|---:|",
        ]
    )
    for key, value in classification_counts.most_common():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Prop Top5-only Breakdown",
            "",
            "| Bucket | Count |",
            "|---|---:|",
        ]
    )
    for key, value in prop_top5_only_breakdown.most_common():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Representative Missing-title Events",
            "",
            "| qid | class | missing title | dense rank | prop rank | seed rank | admitted rank | question |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for event in missing_title_events[:20]:
        question = str(event["question"]).replace("|", "\\|")
        lines.append(
            f"| {event['query_index']} | {event['classification']} | {event['missing_title']} | "
            f"{event['dense200_title_rank'] or ''} | {event['prop_pool_title_rank'] or ''} | "
            f"{event['etv4_seed_title_rank'] or ''} | {event['etv4_admitted_title_rank'] or ''} | {question} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(str(OUT_JSON))
    print(str(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
