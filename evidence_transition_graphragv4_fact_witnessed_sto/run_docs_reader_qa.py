#!/usr/bin/env python3
"""Reader-only QA for baseline outputs that already contain reader documents.

This script does not change retrieval. It replays the documents saved in
``examples[*].docs`` or the top documents from each example's exported candidate
pool through the same reader path used by the V4 reader runner. This lets
baselines be compared under the same answer-generation budget and context size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compare_graph_retrievers import get_worktree_baseline_runtime, make_config
from evidence_transition_graphragv4_fact_witnessed_sto.run_reader_qa import (
    ReaderOnlySystem,
    evaluate_qa,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_method_name(path: Path, payload: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    name = path.name.lower()
    if "proprag" in name:
        return "proprag"
    if "hipporag" in name:
        return "hipporag_v2"
    config = payload.get("config", {}) if isinstance(payload, Mapping) else {}
    source = str(config.get("external_pool_source_name") or payload.get("source") or "").lower()
    if "proprag" in source:
        return "proprag"
    if "hipporag" in source:
        return "hipporag_v2"
    return path.stem


def coerce_doc_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, Mapping):
        for key in ("passage", "text", "content", "doc", "document"):
            value = doc.get(key)
            if value:
                return str(value)
    return str(doc or "")


def load_pool_records(pool_path: Path) -> Dict[int, Mapping[str, Any]]:
    payload = load_json(pool_path)
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    result: Dict[int, Mapping[str, Any]] = {}
    for offset, record in enumerate(records or []):
        if not isinstance(record, Mapping):
            continue
        query_idx = int(record.get("query_idx", offset))
        result[query_idx] = record
    return result


def pool_path_from_example(example: Mapping[str, Any]) -> Path | None:
    trace = example.get("retrieval_trace") or {}
    if not isinstance(trace, Mapping):
        return None
    value = trace.get("external_pool_path")
    if not value:
        return None
    return Path(str(value)).expanduser()


def pool_query_index_from_example(example: Mapping[str, Any], fallback: int) -> int:
    trace = example.get("retrieval_trace") or {}
    if isinstance(trace, Mapping) and trace.get("external_query_idx") is not None:
        return int(trace["external_query_idx"])
    if example.get("query_index") is not None:
        return int(example["query_index"])
    return int(fallback)


def docs_from_exported_pool(
    *,
    example: Mapping[str, Any],
    fallback_index: int,
    pool_cache: Dict[Path, Dict[int, Mapping[str, Any]]],
    qa_top_k: int,
) -> List[str]:
    pool_path = pool_path_from_example(example)
    if pool_path is None:
        return []
    resolved = pool_path.resolve()
    if resolved not in pool_cache:
        pool_cache[resolved] = load_pool_records(resolved)
    query_idx = pool_query_index_from_example(example, fallback=fallback_index)
    record = pool_cache[resolved].get(query_idx)
    if not isinstance(record, Mapping):
        return []
    return [coerce_doc_text(doc) for doc in (record.get("pool_docs", []) or [])[:qa_top_k]]


def get_retrieval_r5(payload: Mapping[str, Any]) -> float | None:
    for key in ("overall_recomputed", "overall_from_pipeline", "metrics", "overall"):
        metrics = payload.get(key)
        if not isinstance(metrics, Mapping):
            continue
        for metric_key in ("Recall@5", "r5", "R@5"):
            if metric_key in metrics:
                return float(metrics[metric_key])
    return None


def make_reader_config(
    *,
    dataset_name: str,
    input_payload: Mapping[str, Any],
    args: argparse.Namespace,
    config_cls,
):
    llm_name = args.llm_name or input_payload.get("llm_request_name") or input_payload.get("llm_name")
    llm_base_url = args.llm_base_url or input_payload.get("llm_base_url")
    embedding_name = (
        args.embedding_name
        or input_payload.get("embedding_name")
        or "/mnt/nvme/Qwen3-Embedding-8B"
    )
    embedding_base_url = args.embedding_base_url or input_payload.get("embedding_base_url")
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
        corpus_len=0,
        config_cls=config_cls,
    )


def build_query_solutions(
    *,
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    query_solution_cls,
) -> tuple[List[Any], List[List[str]], List[Dict[str, Any]]]:
    examples = list(payload.get("examples", []) or [])
    if args.max_queries > 0:
        examples = examples[: int(args.max_queries)]

    query_solutions: List[Any] = []
    gold_answers: List[List[str]] = []
    per_query: List[Dict[str, Any]] = []
    pool_cache: Dict[Path, Dict[int, Mapping[str, Any]]] = {}
    for query_index, example in enumerate(examples):
        question = str(example.get("question") or "")
        answers = [str(answer) for answer in example.get("gold_answers", []) or []]
        if not answers and example.get("answer") is not None:
            answers = [str(example.get("answer"))]
        if args.doc_source == "external_pool_topk_docs":
            docs = docs_from_exported_pool(
                example=example,
                fallback_index=query_index,
                pool_cache=pool_cache,
                qa_top_k=int(args.qa_top_k),
            )
            if not docs:
                raise KeyError(
                    "doc_source=external_pool_topk_docs requires retrieval_trace.external_pool_path "
                    f"and a matching pool record for query {query_index}"
                )
        else:
            docs = [coerce_doc_text(doc) for doc in (example.get("docs", []) or [])[: int(args.qa_top_k)]]
        docs = [doc[: int(args.context_max_chars)] if args.context_max_chars > 0 else doc for doc in docs]
        docs = [doc for doc in docs if doc]
        doc_scores = np.asarray([1.0 / float(rank + 1) for rank in range(len(docs))], dtype=np.float32)
        query_solutions.append(
            query_solution_cls(
                question=question,
                docs=docs,
                doc_scores=doc_scores,
                gold_answers=answers,
                gold_docs=[],
            )
        )
        gold_answers.append(answers)
        per_query.append(
            {
                "query_index": int(example.get("query_index", query_index)),
                "question": question,
                "gold_answers": answers,
                "num_reader_docs": len(docs),
                "reader_doc_source": str(args.doc_source),
                "retrieved_doc_ids_top5": list(example.get("retrieved_doc_ids", []) or [])[:5],
            }
        )
    return query_solutions, gold_answers, per_query


def summarize_qa(
    *,
    method_name: str,
    retrieval_r5: float | None,
    qa_result: Mapping[str, Any],
    per_query_base: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    predicted_answers = list(qa_result.get("predicted_answers", []) or [])
    example_em = list(qa_result.get("example_em", []) or [])
    example_f1 = list(qa_result.get("example_f1", []) or [])
    per_query: List[Dict[str, Any]] = []
    for idx, base_row in enumerate(per_query_base):
        em_value = example_em[idx] if idx < len(example_em) else None
        f1_value = example_f1[idx] if idx < len(example_f1) else None
        row = dict(base_row)
        row["predicted_answer"] = predicted_answers[idx] if idx < len(predicted_answers) else None
        row["ExactMatch"] = None if em_value is None else round(float(em_value), 4)
        row["F1"] = None if f1_value is None else round(float(f1_value), 4)
        per_query.append(row)

    qa_metrics = dict(qa_result.get("qa_metrics", {}) or {})
    metrics = {
        "count": len(per_query_base),
        "r5": None if retrieval_r5 is None else round(float(retrieval_r5), 6),
        "ExactMatch": qa_metrics.get("ExactMatch"),
        "F1": qa_metrics.get("F1"),
        "mean_reader_docs": round(
            sum(float(row["num_reader_docs"]) for row in per_query_base) / float(max(len(per_query_base), 1)),
            6,
        ),
    }
    return {
        "method": method_name,
        "metrics": metrics,
        "per_query": per_query,
    }


def evaluate_input_file(
    *,
    input_path: Path,
    args: argparse.Namespace,
    runtime,
) -> Dict[str, Any]:
    payload = load_json(input_path)
    dataset_name = str(payload.get("dataset") or input_path.stem.split("_", 1)[0])
    method_name = infer_method_name(input_path, payload, args.method_name)
    config = make_reader_config(
        dataset_name=dataset_name,
        input_payload=payload,
        args=args,
        config_cls=runtime.BaseConfigCls,
    )
    system = None if bool(args.skip_qa) else ReaderOnlySystem(
        global_config=config,
        qwen_disable_thinking=bool(args.qwen_disable_thinking),
    )
    query_solutions, answers, per_query_base = build_query_solutions(
        payload=payload,
        args=args,
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
            "input_path": str(input_path.resolve()),
            "doc_source": str(args.doc_source),
            "uses_saved_reader_docs": args.doc_source == "saved_docs",
            "uses_exported_pool_topk_docs": args.doc_source == "external_pool_topk_docs",
            "does_not_modify_retrieval": True,
            "retrieval_r5_source": "input_file.overall_recomputed.Recall@5",
        },
        "reader": {
            "llm_name": getattr(config, "llm_name", None),
            "llm_base_url": getattr(config, "llm_base_url", None),
            "qa_top_k": getattr(config, "qa_top_k", None),
            "max_new_tokens": getattr(config, "max_new_tokens", None),
        },
        "methods": {
            method_name: summarize_qa(
                method_name=method_name,
                retrieval_r5=get_retrieval_r5(payload),
                qa_result=qa_result,
                per_query_base=per_query_base,
            )
        },
    }


def write_markdown(payload: Mapping[str, Any], output_md: Path) -> None:
    lines = [
        "# Saved-Docs Reader QA",
        "",
        "| Dataset | Method | Count | R@5 | EM | F1 | Reader docs | Max tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in payload.get("datasets", []) or []:
        reader = dataset.get("reader", {}) or {}
        for method_name, method in (dataset.get("methods", {}) or {}).items():
            metrics = method.get("metrics", {}) or {}
            r5 = metrics.get("r5")
            em = metrics.get("ExactMatch")
            f1 = metrics.get("F1")
            lines.append(
                f"| {dataset.get('dataset', '')} | "
                f"{method_name} | "
                f"{int(metrics.get('count', 0) or 0)} | "
                f"{'' if r5 is None else f'{float(r5):.4f}'} | "
                f"{'' if em is None else f'{float(em):.4f}'} | "
                f"{'' if f1 is None else f'{float(f1):.4f}'} | "
                f"{float(metrics.get('mean_reader_docs', 0.0)):.2f} | "
                f"{int(reader.get('max_new_tokens', 0) or 0)} |"
            )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_input_files(value: str) -> List[Path]:
    return [Path(item.strip()).expanduser() for item in str(value).split(",") if item.strip()]


def run_docs_reader_qa(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = get_worktree_baseline_runtime()
    datasets = [
        evaluate_input_file(input_path=input_path, args=args, runtime=runtime)
        for input_path in parse_input_files(args.input_files)
    ]
    payload = {
        "format": "saved_docs_reader_only_qa",
        "max_new_tokens": int(args.max_new_tokens),
        "qa_top_k": int(args.qa_top_k),
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
    parser.add_argument("--input-files", required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument(
        "--doc-source",
        choices=["saved_docs", "external_pool_topk_docs"],
        default="saved_docs",
        help=(
            "saved_docs replays examples[*].docs; external_pool_topk_docs uses "
            "pool_docs[:qa_top_k] from retrieval_trace.external_pool_path."
        ),
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--llm-name", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--embedding-name", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--openie-mode", default="online")
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--context-max-chars", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_docs_reader_qa(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
