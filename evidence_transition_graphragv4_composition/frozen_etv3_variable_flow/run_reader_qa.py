#!/usr/bin/env python3
"""Reader-only QA for ETv3 retrieval reports.

This runner is intentionally independent from ``run_transition_top5_qa.py``.
The legacy transition QA runner is coupled to an SFB source-report variant; ETv3
only needs fixed top-5 documents from its own retrieval report plus query/gold
metadata from the minimal per-query report created during fresh indexing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from compare_graph_retrievers import (
    get_worktree_baseline_runtime,
    install_qwen_disable_thinking,
    make_config,
)
from evidence_transition_graphragv4_composition.frozen_etv3_variable_flow.contract import METHOD_NAME, QA_DOC_KEY
from hipporag.llm import _get_llm_class
from hipporag.prompts.prompt_template_manager import PromptTemplateManager
from src.hipporag.utils.eval_utils import normalize_answer


METADATA_VARIANT = "hipporag_v2"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_report_path(raw_path: Any, *, retrieval_report_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    repo_path = (_REPO_ROOT / path).resolve()
    if repo_path.exists():
        return repo_path
    return (retrieval_report_path.parent / path).resolve()


def unique_ints(values: Iterable[Any], *, limit: int = 5) -> List[int]:
    seen: set[int] = set()
    result: List[int] = []
    for value in values:
        doc_idx = int(value)
        if doc_idx < 0 or doc_idx in seen:
            continue
        seen.add(doc_idx)
        result.append(doc_idx)
        if len(result) >= limit:
            break
    return result


def safe_gold_indices(row: Mapping[str, Any]) -> List[int]:
    result: List[int] = []
    for value in row.get("gold_doc_indices", []) or []:
        if value is None:
            continue
        result.append(int(value))
    return result


def title_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    if int(doc_idx) < 0 or int(doc_idx) >= len(openie_docs):
        return ""
    passage = str(openie_docs[int(doc_idx)].get("passage") or "")
    return passage.split("\n", 1)[0].strip()


def passage_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    if int(doc_idx) < 0 or int(doc_idx) >= len(openie_docs):
        return ""
    return str(openie_docs[int(doc_idx)].get("passage") or "")


def answer_string_hit(gold_answers: Sequence[str], passages: Sequence[str]) -> bool:
    normalized_context = normalize_answer("\n".join(str(passage) for passage in passages))
    if not normalized_context:
        return False
    for answer in gold_answers:
        normalized_answer = normalize_answer(str(answer))
        if normalized_answer and normalized_answer in normalized_context:
            return True
    return False


def recall_at_5(gold_doc_indices: Sequence[int], top5_doc_indices: Sequence[int]) -> float:
    gold = {int(value) for value in gold_doc_indices}
    if not gold:
        return 0.0
    return float(len(gold & set(int(value) for value in top5_doc_indices[:5]))) / float(len(gold))


def evaluate_top5_row(
    *,
    row: Mapping[str, Any],
    doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    top5 = unique_ints(doc_indices, limit=5)
    gold = {int(value) for value in row.get("gold_doc_indices", []) or []}
    hit_gold = sorted(gold & set(top5))
    passages = [passage_for_doc(openie_docs, doc_idx) for doc_idx in top5]
    return {
        "method": METHOD_NAME,
        "doc_indices_top5": top5,
        "titles_top5": [title_for_doc(openie_docs, doc_idx) for doc_idx in top5],
        "gold_count_at5": len(hit_gold),
        "recall_at5": recall_at_5(list(gold), top5),
        "any_gold_at5": bool(hit_gold),
        "all_gold_at5": bool(gold) and gold.issubset(set(top5)),
        "answer_string_hit_at5": answer_string_hit(row.get("gold_answers", []) or [], passages),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    denom = float(max(len(rows), 1))
    return {
        "count": len(rows),
        "r5": round(sum(float(row["recall_at5"]) for row in rows) / denom, 6),
        "all_gold_at5": round(sum(1.0 for row in rows if row["all_gold_at5"]) / denom, 6),
        "any_gold_at5": round(sum(1.0 for row in rows if row["any_gold_at5"]) / denom, 6),
        "answer_string_hit_at5": round(
            sum(1.0 for row in rows if row["answer_string_hit_at5"]) / denom,
            6,
        ),
        "mean_gold_count_at5": round(
            sum(float(row["gold_count_at5"]) for row in rows) / denom,
            6,
        ),
    }


def top5_failure_bucket(row: Mapping[str, Any], exact_match: float | None) -> str:
    if exact_match is None:
        return "qa_not_run"
    if float(exact_match) >= 1.0:
        if bool(row.get("all_gold_at5")):
            return "qa_exact_all_gold_top5"
        if bool(row.get("any_gold_at5")):
            return "qa_exact_partial_gold_top5"
        return "qa_exact_no_gold_top5"
    if not bool(row.get("any_gold_at5")):
        return "no_gold_top5_qa_wrong"
    if bool(row.get("all_gold_at5")):
        return "all_gold_top5_qa_wrong"
    return "partial_gold_top5_qa_wrong"


def load_minimal_metadata(report_path: Path) -> Dict[int, Mapping[str, Any]]:
    payload = load_json(report_path)
    variants = payload.get("variants", {}) if isinstance(payload, Mapping) else {}
    rows = variants.get(METADATA_VARIANT) if isinstance(variants, Mapping) else None
    if rows is None:
        raise KeyError(
            f"ETv3 reader QA requires minimal metadata variant {METADATA_VARIANT!r}; "
            f"found={sorted(variants) if isinstance(variants, Mapping) else []} in {report_path}"
        )
    return {int(row["query_index"]): row for row in rows}


class ReaderOnlySystem:
    """Minimal system surface needed for HippoRAG's QA prompt path."""

    def __init__(self, global_config, *, qwen_disable_thinking: bool = False):
        self.global_config = global_config
        self.llm_model = _get_llm_class(global_config)
        install_qwen_disable_thinking(
            self.llm_model,
            SimpleNamespace(qwen_disable_thinking=bool(qwen_disable_thinking)),
        )
        self.prompt_template_manager = PromptTemplateManager()

    def qa(self, query_solutions):
        all_messages = []
        qa_top_k = int(getattr(self.global_config, "qa_top_k", 5))
        for query_idx, query_solution in enumerate(query_solutions):
            if query_idx % 10 == 0:
                print(f"[reader] build prompt {query_idx + 1}/{len(query_solutions)}", flush=True)
            prompt_user = ""
            for passage in query_solution.docs[:qa_top_k]:
                prompt_user += f"Wikipedia Title: {passage}\n\n"
            prompt_user += "Question: " + query_solution.question + "\nThought: "
            if self.prompt_template_manager.is_template_name_valid(name=f"rag_qa_{self.global_config.dataset}"):
                prompt_dataset_name = self.global_config.dataset
            else:
                prompt_dataset_name = "musique"
            all_messages.append(
                self.prompt_template_manager.render(
                    name=f"rag_qa_{prompt_dataset_name}",
                    prompt_user=prompt_user,
                )
            )

        responses = []
        for query_idx, messages in enumerate(all_messages):
            if query_idx % 10 == 0:
                print(f"[reader] infer {query_idx + 1}/{len(all_messages)}", flush=True)
            responses.append(self.llm_model.infer(messages))
        response_messages, metadata, _cache_hit = zip(*responses) if responses else ([], [], [])
        answered = []
        for query_solution, response_content in zip(query_solutions, response_messages):
            try:
                pred_ans = str(response_content).split("Answer:", 1)[1].strip()
            except Exception:
                pred_ans = str(response_content)
            query_solution.answer = pred_ans
            answered.append(query_solution)
        return answered, list(response_messages), list(metadata)


def make_reader_config(
    *,
    dataset_name: str,
    retrieval_config: Mapping[str, Any],
    corpus_len: int,
    args: argparse.Namespace,
    config_cls,
):
    llm_name = args.llm_name or retrieval_config.get("reader_llm_name") or retrieval_config.get("llm_name")
    llm_base_url = (
        args.llm_base_url
        or retrieval_config.get("reader_llm_base_url")
        or retrieval_config.get("llm_base_url")
    )
    embedding_name = (
        args.embedding_name
        or retrieval_config.get("reader_embedding_name")
        or retrieval_config.get("embedding_endpoint_model_id")
        or retrieval_config.get("embedding_name")
        or "/mnt/nvme/Qwen3-Embedding-8B"
    )
    embedding_base_url = (
        args.embedding_base_url
        or retrieval_config.get("reader_embedding_base_url")
        or retrieval_config.get("embedding_base_url")
    )
    minimal_args = SimpleNamespace(
        save_dir=args.save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        embedding_name=embedding_name,
        embedding_base_url=embedding_base_url,
        force_index_from_scratch="False",
        force_openie_from_scratch="False",
        openie_mode=args.openie_mode,
        retrieval_top_k=args.qa_top_k,
        qa_top_k=args.qa_top_k,
        max_new_tokens=args.max_new_tokens,
        qwen_disable_thinking=bool(args.qwen_disable_thinking),
        embedding_batch_size=args.embedding_batch_size,
    )
    return make_config(
        minimal_args,
        dataset_name,
        args.save_dir,
        corpus_len,
        config_cls=config_cls,
    )


def build_query_solutions(
    *,
    retrieval_report: Mapping[str, Any],
    retrieval_report_path: Path,
    qa_top_k: int,
    max_queries: int,
    context_mode: str,
    context_max_chars: int,
    query_solution_cls,
) -> Tuple[List[Any], List[List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if context_mode != "full":
        raise ValueError("ETv3 reader QA currently supports context_mode='full' only.")

    openie_path = resolve_report_path(
        retrieval_report["openie_path"],
        retrieval_report_path=retrieval_report_path,
    )
    openie_docs = list(load_json(openie_path).get("docs", []) or [])

    metadata_path = resolve_report_path(
        retrieval_report["input_report"],
        retrieval_report_path=retrieval_report_path,
    )
    metadata_rows = load_minimal_metadata(metadata_path)

    rows = list(retrieval_report.get("rows", []) or [])
    if int(max_queries) > 0:
        rows = rows[: int(max_queries)]

    query_solutions: List[Any] = []
    gold_answers: List[List[str]] = []
    evidence_rows: List[Dict[str, Any]] = []
    per_query: List[Dict[str, Any]] = []

    for row in rows:
        query_index = int(row["query_index"])
        metadata_row = metadata_rows[query_index]
        query = str(row.get("question") or metadata_row.get("question") or "")
        answers = [str(answer) for answer in metadata_row.get("gold_answers", []) or []]
        gold_doc_indices = safe_gold_indices(row) or safe_gold_indices(metadata_row)
        doc_indices = unique_ints(
            row.get(QA_DOC_KEY) or row.get("retrieved_doc_indices_top5") or [],
            limit=int(qa_top_k),
        )
        selected_docs = []
        for doc_idx in doc_indices:
            passage = passage_for_doc(openie_docs, doc_idx)
            if context_max_chars > 0:
                passage = passage[: int(context_max_chars)]
            if passage:
                selected_docs.append(passage)
        doc_scores = np.asarray([1.0 / float(rank + 1) for rank in range(len(selected_docs))], dtype=np.float32)
        merged_row = {
            **dict(row),
            "question": query,
            "gold_answers": answers,
            "gold_doc_indices": gold_doc_indices,
        }
        top5_eval = evaluate_top5_row(
            row=merged_row,
            doc_indices=doc_indices,
            openie_docs=openie_docs,
        )
        query_solutions.append(
            query_solution_cls(
                question=query,
                docs=selected_docs,
                doc_scores=doc_scores,
                gold_answers=answers,
                gold_docs=[passage_for_doc(openie_docs, idx) for idx in gold_doc_indices],
            )
        )
        gold_answers.append(answers)
        evidence_rows.append(top5_eval)
        per_query.append(
            {
                "query_index": query_index,
                "question": query,
                "gold_answers": answers,
                "gold_doc_indices": gold_doc_indices,
                "doc_indices_top5": top5_eval["doc_indices_top5"],
                "titles_top5": top5_eval["titles_top5"],
                "recall_at5": top5_eval["recall_at5"],
                "gold_count_at5": top5_eval["gold_count_at5"],
                "any_gold_at5": top5_eval["any_gold_at5"],
                "all_gold_at5": top5_eval["all_gold_at5"],
                "answer_string_hit_at5": top5_eval["answer_string_hit_at5"],
            }
        )
    return query_solutions, gold_answers, evidence_rows, per_query


def evaluate_qa(
    *,
    system: ReaderOnlySystem | None,
    query_solutions: Sequence[Any],
    gold_answers: Sequence[Sequence[str]],
    qa_em_cls,
    qa_f1_cls,
    skip_qa: bool,
) -> Dict[str, Any]:
    if skip_qa:
        return {
            "predicted_answers": [None for _ in query_solutions],
            "responses": [],
            "metadata": [],
            "qa_metrics": {},
            "example_em": [None for _ in query_solutions],
            "example_f1": [None for _ in query_solutions],
        }
    if system is None:
        raise ValueError("system is required unless skip_qa=True")
    answered_solutions, responses, metadata = system.qa(list(query_solutions))
    predicted_answers = [str(solution.answer or "") for solution in answered_solutions]
    qa_em = qa_em_cls(global_config=system.global_config)
    qa_f1 = qa_f1_cls(global_config=system.global_config)
    em_results, em_examples = qa_em.calculate_metric_scores(
        gold_answers=[list(items) for items in gold_answers],
        predicted_answers=predicted_answers,
    )
    f1_results, f1_examples = qa_f1.calculate_metric_scores(
        gold_answers=[list(items) for items in gold_answers],
        predicted_answers=predicted_answers,
    )
    return {
        "predicted_answers": predicted_answers,
        "responses": responses,
        "metadata": metadata,
        "qa_metrics": {
            "ExactMatch": round(float(em_results.get("ExactMatch", 0.0)), 4),
            "F1": round(float(f1_results.get("F1", 0.0)), 4),
        },
        "example_em": [float(item["ExactMatch"]) for item in em_examples],
        "example_f1": [float(item["F1"]) for item in f1_examples],
    }


def summarize_result(
    *,
    evidence_rows: Sequence[Mapping[str, Any]],
    per_query_base: Sequence[Mapping[str, Any]],
    qa_result: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = aggregate_rows(evidence_rows)
    metrics.update(qa_result.get("qa_metrics", {}))
    predicted_answers = list(qa_result.get("predicted_answers", []) or [])
    example_em = list(qa_result.get("example_em", []) or [])
    example_f1 = list(qa_result.get("example_f1", []) or [])
    per_query: List[Dict[str, Any]] = []
    buckets: Counter[str] = Counter()
    for idx, base_row in enumerate(per_query_base):
        em_value = example_em[idx] if idx < len(example_em) else None
        f1_value = example_f1[idx] if idx < len(example_f1) else None
        row = dict(base_row)
        row["predicted_answer"] = predicted_answers[idx] if idx < len(predicted_answers) else None
        row["ExactMatch"] = None if em_value is None else round(float(em_value), 4)
        row["F1"] = None if f1_value is None else round(float(f1_value), 4)
        row["qa_bucket"] = top5_failure_bucket(row, em_value)
        buckets[row["qa_bucket"]] += 1
        per_query.append(row)
    return {
        "method": METHOD_NAME,
        "metrics": metrics,
        "qa_buckets": dict(buckets),
        "per_query": per_query,
    }


def evaluate_dataset_report(
    *,
    retrieval_report_path: Path,
    args: argparse.Namespace,
    runtime,
) -> Dict[str, Any]:
    retrieval_report = load_json(retrieval_report_path)
    dataset_name = str(retrieval_report["dataset"])
    openie_path = resolve_report_path(
        retrieval_report["openie_path"],
        retrieval_report_path=retrieval_report_path,
    )
    openie_docs = list(load_json(openie_path).get("docs", []) or [])
    config = make_reader_config(
        dataset_name=dataset_name,
        retrieval_config=retrieval_report.get("config", {}) or {},
        corpus_len=len(openie_docs),
        args=args,
        config_cls=runtime.BaseConfigCls,
    )
    system = None if bool(args.skip_qa) else ReaderOnlySystem(
        global_config=config,
        qwen_disable_thinking=bool(args.qwen_disable_thinking),
    )
    query_solutions, answers, evidence_rows, per_query_base = build_query_solutions(
        retrieval_report=retrieval_report,
        retrieval_report_path=retrieval_report_path,
        qa_top_k=int(args.qa_top_k),
        max_queries=int(args.max_queries),
        context_mode=str(args.context_mode),
        context_max_chars=int(args.context_max_chars),
        query_solution_cls=runtime.QuerySolutionCls,
    )
    qa_result = evaluate_qa(
        system=system,
        query_solutions=query_solutions,
        gold_answers=answers,
        qa_em_cls=runtime.QAExactMatchCls,
        qa_f1_cls=runtime.QAF1ScoreCls,
        skip_qa=bool(args.skip_qa),
    )
    return {
        "dataset": dataset_name,
        "num_queries": len(per_query_base),
        "source": {
            "retrieval_report_path": str(retrieval_report_path.resolve()),
            "metadata_report_path": str(Path(str(retrieval_report["input_report"])).resolve()),
            "metadata_variant": METADATA_VARIANT,
            "openie_path": str(openie_path.resolve()),
            "doc_key": "retrieved_doc_indices_top5",
            "uses_sfb_variant": False,
            "uses_legacy_transition_qa_runner": False,
        },
        "reader": {
            "llm_name": getattr(config, "llm_name", None),
            "llm_base_url": getattr(config, "llm_base_url", None),
            "qa_top_k": getattr(config, "qa_top_k", None),
        },
        "methods": {
            METHOD_NAME: summarize_result(
                evidence_rows=evidence_rows,
                per_query_base=per_query_base,
                qa_result=qa_result,
            )
        },
    }


def write_markdown(payload: Mapping[str, Any], output_md: Path) -> None:
    lines = [
        f"# {METHOD_NAME} Reader QA",
        "",
        "| Dataset | Count | R@5 | All-gold@5 | EM | F1 | SFB variant | Legacy QA runner |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset in payload.get("datasets", []) or []:
        method = (dataset.get("methods", {}) or {}).get(METHOD_NAME, {}) or {}
        metrics = method.get("metrics", {}) or {}
        source = dataset.get("source", {}) or {}
        lines.append(
            f"| {dataset.get('dataset', '')} | "
            f"{int(metrics.get('count', 0) or 0)} | "
            f"{float(metrics.get('r5', 0.0)):.4f} | "
            f"{float(metrics.get('all_gold_at5', 0.0)):.4f} | "
            f"{float(metrics.get('ExactMatch', 0.0)):.4f} | "
            f"{float(metrics.get('F1', 0.0)):.4f} | "
            f"{bool(source.get('uses_sfb_variant', False))} | "
            f"{bool(source.get('uses_legacy_transition_qa_runner', False))} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_reports(value: str) -> List[Path]:
    return [Path(item.strip()).expanduser() for item in str(value).split(",") if item.strip()]


def run_reader_qa(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = get_worktree_baseline_runtime()
    datasets = [
        evaluate_dataset_report(
            retrieval_report_path=path,
            args=args,
            runtime=runtime,
        )
        for path in parse_reports(args.retrieval_reports)
    ]
    payload = {
        "method": METHOD_NAME,
        "format": "etv3_reader_only_qa",
        "qa_doc_key": "retrieved_doc_indices_top5",
        "metadata_variant": METADATA_VARIANT,
        "uses_sfb_variant": False,
        "uses_legacy_transition_qa_runner": False,
        "datasets": datasets,
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(payload, Path(args.output_md).expanduser())
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-reports", required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--llm-name", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--embedding-name", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--openie-mode", default="online")
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--context-mode", default="full", choices=["full"])
    parser.add_argument("--context-max-chars", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_reader_qa(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
