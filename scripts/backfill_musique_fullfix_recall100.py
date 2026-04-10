import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LOCAL_SRC_DIR = ROOT_DIR / "src"
existing_src_module = sys.modules.get("src")
existing_src_paths = list(getattr(existing_src_module, "__path__", [])) if existing_src_module is not None else []
if LOCAL_SRC_DIR.exists() and str(LOCAL_SRC_DIR) not in existing_src_paths:
    local_src_module = types.ModuleType("src")
    local_src_module.__path__ = [str(LOCAL_SRC_DIR)]
    sys.modules["src"] = local_src_module

from eval_causal_qwen3 import (
    build_chunk_id_to_doc_text,
    compute_retrieval_recall_metrics,
    get_gold_docs,
)
from src.hipporag.utils.misc_utils import QuerySolution


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    corpus_path: Path
    sample_path: Path
    baseline_cache_path: Path


@dataclass(frozen=True)
class MethodReportSpec:
    run_type: str
    report_path: Path


MUSIQUE_SPEC = DatasetSpec(
    dataset_name="musique",
    corpus_path=ROOT_DIR / "reproduce/dataset/musique_corpus.json",
    sample_path=ROOT_DIR / "reproduce/dataset/musique.json",
    baseline_cache_path=(
        ROOT_DIR
        / "outputs_step0_general_musique/eval_reports/"
        / "causal_eval_musique_full_qwen3-8b_qatopk5_baseline_20260409fullfix.retrieval_cache.json"
    ),
)

MUSIQUE_REPORTS = (
    MethodReportSpec(
        run_type="baseline_top10_plus_ce",
        report_path=(
            ROOT_DIR
            / "outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409fullfix.json"
        ),
    ),
    MethodReportSpec(
        run_type="random3_deep_plus_ce",
        report_path=(
            ROOT_DIR
            / "outputs_step0_general_musique/eval_reports/random3_deep_plus_ce_qatopk5_20260409fullfix.json"
        ),
    ),
    MethodReportSpec(
        run_type="bridge_append_plus_ce",
        report_path=(
            ROOT_DIR
            / "outputs_step0_general_musique/eval_reports/bridge_append_plus_ce_qatopk5_20260409fullfix.json"
        ),
    ),
)

VALIDATION_CASES = (
    (
        DatasetSpec(
            dataset_name="hotpotqa",
            corpus_path=ROOT_DIR / "reproduce/dataset/hotpotqa_corpus.json",
            sample_path=ROOT_DIR / "reproduce/dataset/hotpotqa.json",
            baseline_cache_path=(
                ROOT_DIR
                / "outputs_step0_general_hotpotqa/eval_reports/"
                / "causal_eval_hotpotqa_full_qwen3-8b_qatopk5_baseline_20260409fullfix.retrieval_cache.json"
            ),
        ),
        MethodReportSpec(
            run_type="bridge_append_plus_ce",
            report_path=(
                ROOT_DIR
                / "outputs_step0_general_hotpotqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409fullfix.json"
            ),
        ),
    ),
    (
        DatasetSpec(
            dataset_name="2wikimultihopqa",
            corpus_path=ROOT_DIR / "reproduce/dataset/2wikimultihopqa_corpus.json",
            sample_path=ROOT_DIR / "reproduce/dataset/2wikimultihopqa.json",
            baseline_cache_path=(
                ROOT_DIR
                / "outputs_step0_general_2wikimultihopqa/eval_reports/"
                / "causal_eval_2wikimultihopqa_full_qwen3-8b_qatopk5_baseline_20260409fullfix.retrieval_cache.json"
            ),
        ),
        MethodReportSpec(
            run_type="bridge_append_plus_ce",
            report_path=(
                ROOT_DIR
                / "outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409fullfix.json"
            ),
        ),
    ),
)

MUSIQUE_SUMMARY_JSON = ROOT_DIR / "run_logs/fullscale_width_matched_control_musique_top5_remaining_20260409.summary.json"
MUSIQUE_SUMMARY_MD = ROOT_DIR / "run_logs/fullscale_width_matched_control_musique_top5_remaining_20260409.summary.md"
FULLFIX_SUMMARY_JSON = ROOT_DIR / "run_logs/fullscale_width_matched_control_20260409fullfix.summary.json"
FULLFIX_SUMMARY_MD = ROOT_DIR / "run_logs/fullscale_width_matched_control_20260409fullfix.summary.md"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_doc_text(corpus_row: Mapping[str, object]) -> str:
    return f"{corpus_row['title']}\n{corpus_row['text']}"


def reconstruct_method_docs(
    front_doc_ids: Sequence[int | str],
    baseline_retrieved_doc_ids: Sequence[str | None],
    corpus: Sequence[Mapping[str, object]],
    chunk_id_to_doc_text: Mapping[str, str],
) -> List[str]:
    docs: List[str] = []
    seen_docs = set()

    for doc_id in front_doc_ids:
        doc_index = int(doc_id)
        if doc_index < 0 or doc_index >= len(corpus):
            raise IndexError(f"Front doc id {doc_id} is out of corpus bounds [0, {len(corpus)})")
        doc_text = build_doc_text(corpus[doc_index])
        if doc_text in seen_docs:
            continue
        docs.append(doc_text)
        seen_docs.add(doc_text)

    for chunk_id in baseline_retrieved_doc_ids:
        if not chunk_id:
            continue
        doc_text = chunk_id_to_doc_text.get(str(chunk_id))
        if doc_text is None or doc_text in seen_docs:
            continue
        docs.append(doc_text)
        seen_docs.add(doc_text)

    return docs


def compute_method_metrics(dataset_spec: DatasetSpec, report_path: Path) -> Dict[str, float]:
    report_payload = load_json(report_path)
    cache_payload = load_json(dataset_spec.baseline_cache_path)
    corpus = list(load_json(dataset_spec.corpus_path))
    samples = list(load_json(dataset_spec.sample_path))

    traces = list(report_payload.get("expand_assemble_query_traces") or [])
    examples = list(cache_payload.get("examples") or [])
    if len(traces) != len(examples):
        raise ValueError(
            f"Trace/cache query count mismatch for {report_path}: {len(traces)} != {len(examples)}"
        )

    gold_docs = get_gold_docs(samples, dataset_name=dataset_spec.dataset_name, corpus=corpus)
    chunk_id_to_doc_text = build_chunk_id_to_doc_text(corpus)

    query_solutions: List[QuerySolution] = []
    for trace, example in zip(traces, examples):
        expand_trace = dict(trace.get("expand_assemble_trace") or {})
        front_doc_ids = list(expand_trace.get("final_front_doc_ids") or [])
        if not front_doc_ids:
            raise ValueError(f"Missing final_front_doc_ids in report trace: {report_path}")
        baseline_doc_ids = list((example or {}).get("retrieved_doc_ids") or [])
        docs = reconstruct_method_docs(
            front_doc_ids=front_doc_ids,
            baseline_retrieved_doc_ids=baseline_doc_ids,
            corpus=corpus,
            chunk_id_to_doc_text=chunk_id_to_doc_text,
        )
        query_solutions.append(
            QuerySolution(
                question=str(trace.get("question", "")),
                docs=docs,
            )
        )

    return compute_retrieval_recall_metrics(query_solutions=query_solutions, gold_docs=gold_docs)


def verify_existing_metrics(dataset_spec: DatasetSpec, report_spec: MethodReportSpec) -> None:
    report_payload = load_json(report_spec.report_path)
    stored_metrics = dict(
        ((report_payload.get("expand_assemble_qa") or {}).get("method_retrieval_metrics") or {})
    )
    recomputed_metrics = compute_method_metrics(dataset_spec, report_spec.report_path)
    for key, stored_value in stored_metrics.items():
        if key not in recomputed_metrics:
            continue
        if stored_value is None:
            continue
        if float(stored_value) != float(recomputed_metrics[key]):
            raise AssertionError(
                f"{dataset_spec.dataset_name}/{report_spec.run_type} metric mismatch for {key}: "
                f"stored={stored_value} recomputed={recomputed_metrics[key]}"
            )


def backfill_musique_reports() -> Dict[str, Dict[str, float]]:
    updated_metrics: Dict[str, Dict[str, float]] = {}
    for report_spec in MUSIQUE_REPORTS:
        report_payload = load_json(report_spec.report_path)
        recomputed_metrics = compute_method_metrics(MUSIQUE_SPEC, report_spec.report_path)
        expand_assemble = dict(report_payload.get("expand_assemble_qa") or {})
        expand_assemble["method_retrieval_metrics"] = recomputed_metrics
        report_payload["expand_assemble_qa"] = expand_assemble
        report_spec.report_path.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        updated_metrics[report_spec.run_type] = recomputed_metrics
    return updated_metrics


def update_summary_json(summary_path: Path, recall_updates: Mapping[str, float]) -> None:
    payload = load_json(summary_path)
    rows = list(payload.get("rows") or [])
    for row in rows:
        if str(row.get("dataset")) != "musique":
            continue
        run_type = str(row.get("run_type"))
        if run_type not in recall_updates:
            continue
        row["recall_at_100"] = float(recall_updates[run_type])
    summary_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_float(value: object) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def build_musique_summary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Full-Scale Width-Matched Control: MuSiQue top-5 remaining",
        "",
        "Naming note:",
        "- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.",
        "- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.",
        "- `random3_deep_plus_ce` = matched random-append CE control.",
        "- This is the valid full-scale family tagged `20260409fullfix`, method-matched to the earlier `limit=100` width-matched controls.",
        "",
        "| Run | EM | F1 | R@5 | R@20 | R@100 | num_queries | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta_baseline = row.get("delta_vs_baseline_em")
        delta_top10 = row.get("delta_vs_baseline_top10_plus_ce_em")
        delta_baseline_str = "—" if delta_baseline is None else f"{float(delta_baseline):+.4f}"
        delta_top10_str = "—" if delta_top10 is None else f"{float(delta_top10):+.4f}"
        lines.append(
            f"| {row['run_type']} | {format_float(row.get('em'))} | {format_float(row.get('f1'))} | "
            f"{format_float(row.get('recall_at_5'))} | {format_float(row.get('recall_at_20'))} | "
            f"{format_float(row.get('recall_at_100'))} | {row.get('num_queries', '—')} | "
            f"{delta_baseline_str} | {delta_top10_str} |"
        )
    return "\n".join(lines) + "\n"


def build_fullfix_summary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Full-Scale Width-Matched Control",
        "",
        "Naming note:",
        "- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.",
        "- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.",
        "- `random3_deep_plus_ce` = matched random-append CE control.",
        "- This `20260409fullfix` family is the first valid full-scale version of the width-matched CE controls and is method-matched to the earlier `limit=100` runs.",
        "",
        "| Phase | Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | ΔEM vs baseline | ΔEM vs baseline top10+CE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta_baseline = row.get("delta_vs_baseline_em")
        delta_top10 = row.get("delta_vs_baseline_top10_plus_ce_em")
        delta_baseline_str = "—" if delta_baseline is None else f"{float(delta_baseline):+.4f}"
        delta_top10_str = "—" if delta_top10 is None else f"{float(delta_top10):+.4f}"
        lines.append(
            f"| {row.get('phase', '—')} | {row['dataset']} | {row['run_type']} | "
            f"{format_float(row.get('em'))} | {format_float(row.get('f1'))} | "
            f"{format_float(row.get('recall_at_5'))} | {format_float(row.get('recall_at_20'))} | "
            f"{format_float(row.get('recall_at_100'))} | {delta_baseline_str} | {delta_top10_str} |"
        )
    return "\n".join(lines) + "\n"


def rewrite_summary_markdown(summary_json_path: Path, summary_md_path: Path, formatter) -> None:
    rows = list((load_json(summary_json_path).get("rows") or []))
    summary_md_path.write_text(formatter(rows), encoding="utf-8")


def main() -> None:
    for dataset_spec, report_spec in VALIDATION_CASES:
        verify_existing_metrics(dataset_spec, report_spec)

    updated_metrics = backfill_musique_reports()
    recall100_updates = {
        run_type: metrics["Recall@100"]
        for run_type, metrics in updated_metrics.items()
    }

    update_summary_json(MUSIQUE_SUMMARY_JSON, recall100_updates)
    rewrite_summary_markdown(MUSIQUE_SUMMARY_JSON, MUSIQUE_SUMMARY_MD, build_musique_summary_markdown)

    update_summary_json(FULLFIX_SUMMARY_JSON, recall100_updates)
    rewrite_summary_markdown(FULLFIX_SUMMARY_JSON, FULLFIX_SUMMARY_MD, build_fullfix_summary_markdown)

    print("Validated existing fullfix reconstruction on HotpotQA and 2Wiki.")
    print("Backfilled MuSiQue method recall metrics:")
    for run_type in ("baseline_top10_plus_ce", "random3_deep_plus_ce", "bridge_append_plus_ce"):
        metrics = updated_metrics[run_type]
        print(
            f"  {run_type}: "
            f"R@5={metrics['Recall@5']:.4f} "
            f"R@20={metrics['Recall@20']:.4f} "
            f"R@100={metrics['Recall@100']:.4f}"
        )


if __name__ == "__main__":
    main()
