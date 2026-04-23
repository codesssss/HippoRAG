#!/usr/bin/env python3
"""Audit PropRAG-vs-DtC protocol alignment and compile pilot-100 tables.

The audit intentionally separates two questions:

1. Are the broad runtime conditions aligned?  Same datasets, reader family,
   embedding endpoint, top-k budgets, and first-N query order.
2. Is the comparison a same fixed-pool composition comparison?  For PropRAG vs
   DtC the expected answer is no unless PropRAG is explicitly adapted to consume
   HippoRAG's top-100 candidate pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DISPLAY_NAMES = {
    "2wikimultihopqa": "2Wiki",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
}


@dataclass(frozen=True)
class DatasetPaths:
    prop_json: Path
    dtc_json: Path
    prop_log: Path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _round4(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    rounded = _round4(value)
    if rounded is None:
        return "n/a"
    return f"{rounded:.4f}"


def _norm_question(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _sha1_json(items: Sequence[str]) -> str:
    payload = json.dumps(list(items), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _dataset_questions(root: Path, dataset: str, limit: int) -> List[str]:
    path = root / "reproduce" / "dataset" / f"{dataset}.json"
    samples = _load_json(path)
    return [_norm_question(row.get("question")) for row in samples[:limit]]


def _examples_questions(report: Mapping[str, Any]) -> List[str]:
    return [_norm_question(row.get("question")) for row in report.get("examples", [])]


def _trace_questions(report: Mapping[str, Any]) -> List[str]:
    return [
        _norm_question(row.get("question"))
        for row in report.get("setwise_selector_query_traces", [])
    ]


def _contains_think_tag(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return bool(re.search(r"</?think>", text, flags=re.IGNORECASE))


def _scan_text_artifacts_for_think(paths: Iterable[Path]) -> List[str]:
    hits: List[str] = []
    for path in paths:
        if _contains_think_tag(path):
            hits.append(str(path))
    return hits


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _line_no(path: Path, pattern: str) -> int | None:
    regex = re.compile(pattern)
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if regex.search(line):
            return idx
    return None


def _qa_function_body(proprag_py: Path) -> str:
    text = _read_text(proprag_py)
    match = re.search(r"\n    def qa\(.*?(?=\n    def |\nclass |\Z)", text, flags=re.DOTALL)
    return match.group(0) if match else ""


def _examples_doc_counts(examples: Sequence[Mapping[str, Any]]) -> List[int]:
    return [len(row.get("docs") or []) for row in examples]


def _dtc_top_counts(traces: Sequence[Mapping[str, Any]], field: str) -> List[int]:
    return [len(row.get(field) or []) for row in traces]


def _status(level: str, check: str, evidence: str) -> Dict[str, str]:
    return {"status": level, "check": check, "evidence": evidence}


def build_paths(args: argparse.Namespace) -> Dict[str, DatasetPaths]:
    prop_run = Path(args.prop_100_dir)
    hippo_root = Path(args.hippo_root)
    return {
        "2wikimultihopqa": DatasetPaths(
            prop_json=prop_run / "2wikimultihopqa.json",
            prop_log=prop_run / "2wikimultihopqa.log",
            dtc_json=hippo_root
            / "outputs_step0_general_nvembed_2wikimultihopqa"
            / "eval_reports"
            / "dtc_embed_nvembed_rankw0p2_repairable_filter_pilot100_anchor2_8041.json",
        ),
        "hotpotqa": DatasetPaths(
            prop_json=prop_run / "hotpotqa.json",
            prop_log=prop_run / "hotpotqa.log",
            dtc_json=hippo_root
            / "outputs_step0_general_nvembed_hotpotqa"
            / "eval_reports"
            / "dtc_embed_nvembed_rankw0p2_repairable_filter_pilot100_anchor2_8042.json",
        ),
        "musique": DatasetPaths(
            prop_json=prop_run / "musique.json",
            prop_log=prop_run / "musique.log",
            dtc_json=hippo_root
            / "outputs_step0_general_nvembed_musique"
            / "eval_reports"
            / "dtc_embed_nvembed_rankw0p2_repairable_filter_pilot100_anchor2_8043.json",
        ),
    }


def audit_dataset_alignment(
    *,
    dataset: str,
    prop_root: Path,
    hippo_root: Path,
    prop_report: Mapping[str, Any],
    dtc_report: Mapping[str, Any],
    limit: int,
) -> Dict[str, Any]:
    prop_dataset_q = _dataset_questions(prop_root, dataset, limit)
    hippo_dataset_q = _dataset_questions(hippo_root, dataset, limit)
    dataset_match = prop_dataset_q == hippo_dataset_q

    prop_examples_q = _examples_questions(prop_report)
    dtc_trace_q = _trace_questions(dtc_report)
    saved_n = min(len(prop_examples_q), len(dtc_trace_q))
    saved_match = prop_examples_q[:saved_n] == dtc_trace_q[:saved_n]

    return {
        "dataset": dataset,
        "display": DISPLAY_NAMES[dataset],
        "dataset_file_first_n_match": dataset_match,
        "dataset_file_first_n": limit,
        "dataset_file_sha1_prop": _sha1_json(prop_dataset_q),
        "dataset_file_sha1_hippo": _sha1_json(hippo_dataset_q),
        "saved_examples_compared": saved_n,
        "prop_saved_examples": len(prop_examples_q),
        "dtc_trace_examples": len(dtc_trace_q),
        "saved_examples_match": saved_match,
        "first_mismatch": _first_mismatch(prop_dataset_q, hippo_dataset_q),
    }


def _first_mismatch(left: Sequence[str], right: Sequence[str]) -> Dict[str, Any] | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": idx, "left": a, "right": b}
    if len(left) != len(right):
        return {"index": min(len(left), len(right)), "left_len": len(left), "right_len": len(right)}
    return None


def collect_metrics(
    dataset: str,
    prop_report: Mapping[str, Any],
    dtc_report: Mapping[str, Any],
) -> Dict[str, Any]:
    dtc = dtc_report.get("setwise_selector_qa") or {}
    base = dtc_report.get("overall_recomputed") or {}
    prop_qa = prop_report.get("qa") or {}
    prop_retr = prop_report.get("retrieval") or {}
    dtc_retr = dtc.get("selector_retrieval_metrics") or {}
    return {
        "dataset": dataset,
        "display": DISPLAY_NAMES[dataset],
        "hippo_baseline_em": _round4(dtc.get("baseline_EM", base.get("ExactMatch"))),
        "hippo_baseline_f1": _round4(dtc.get("baseline_F1", base.get("F1"))),
        "dtc_em": _round4(dtc.get("selector_EM")),
        "dtc_f1": _round4(dtc.get("selector_F1")),
        "dtc_delta_em": _round4(dtc.get("EM_delta")),
        "dtc_delta_f1": _round4(dtc.get("F1_delta")),
        "dtc_recall5": _round4(dtc_retr.get("Recall@5")),
        "dtc_recall20": _round4(dtc_retr.get("Recall@20")),
        "prop_em": _round4(prop_qa.get("ExactMatch")),
        "prop_f1": _round4(prop_qa.get("F1")),
        "prop_recall5": _round4(prop_retr.get("Recall@5")),
        "prop_recall20": _round4(prop_retr.get("Recall@20")),
        "prop_recall100": _round4(prop_retr.get("Recall@100")),
    }


def audit_protocol(
    *,
    prop_root: Path,
    hippo_root: Path,
    paths: Mapping[str, DatasetPaths],
    reports: Mapping[str, Dict[str, Mapping[str, Any]]],
) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []

    for dataset in DATASETS:
        prop = reports[dataset]["prop"]
        dtc = reports[dataset]["dtc"]
        prop_config = prop.get("config") or {}
        dtc_config = dtc.get("config") or {}
        dtc_traces = dtc.get("setwise_selector_query_traces") or []
        prop_examples = prop.get("examples") or []

        label = DISPLAY_NAMES[dataset]
        if prop.get("limit") == dtc.get("limit") == 100:
            checks.append(_status("PASS", f"{label}: pilot limit", "both reports use limit=100"))
        else:
            checks.append(
                _status("FAIL", f"{label}: pilot limit", f"PropRAG={prop.get('limit')} DtC={dtc.get('limit')}")
            )

        prop_reader = prop.get("llm_name")
        dtc_reader = dtc.get("llm_request_name") or dtc.get("llm_name")
        if prop_reader == dtc_reader == "qwen3-8b-train":
            checks.append(_status("PASS", f"{label}: reader model", prop_reader))
        else:
            checks.append(_status("WARN", f"{label}: reader model", f"PropRAG={prop_reader} DtC={dtc_reader}"))

        if prop.get("embedding_name") == dtc.get("embedding_name") == "VLLM/nvidia/NV-Embed-v2":
            checks.append(_status("PASS", f"{label}: embedding model", str(prop.get("embedding_name"))))
        else:
            checks.append(
                _status(
                    "FAIL",
                    f"{label}: embedding model",
                    f"PropRAG={prop.get('embedding_name')} DtC={dtc.get('embedding_name')}",
                )
            )

        prop_emb_url = prop.get("embedding_base_url")
        dtc_emb_url = dtc.get("embedding_base_url")
        if prop_emb_url == dtc_emb_url == "http://localhost:8019/v1/embeddings":
            checks.append(_status("PASS", f"{label}: embedding endpoint", str(prop_emb_url)))
        else:
            checks.append(_status("WARN", f"{label}: embedding endpoint", f"PropRAG={prop_emb_url} DtC={dtc_emb_url}"))

        prop_pool_k = prop_config.get("retrieval_top_k")
        dtc_pool_k = dtc_config.get("setwise_pool_k")
        if prop_pool_k == dtc_pool_k == 100:
            checks.append(_status("PASS", f"{label}: top-100 budget", "retrieval_top_k/setwise_pool_k=100"))
        else:
            checks.append(_status("FAIL", f"{label}: top-100 budget", f"PropRAG={prop_pool_k} DtC={dtc_pool_k}"))

        prop_qa_top_k = prop_config.get("qa_top_k")
        prop_doc_counts = _examples_doc_counts(prop_examples)
        baseline_counts = _dtc_top_counts(dtc_traces, "baseline_top_titles")
        selector_counts = _dtc_top_counts(dtc_traces, "selector_top_titles")
        if prop_qa_top_k == 5 and all(x == 5 for x in prop_doc_counts) and all(x == 5 for x in baseline_counts) and all(x == 5 for x in selector_counts):
            checks.append(_status("PASS", f"{label}: QA context width", "PropRAG docs and DtC trace top titles are width 5"))
        else:
            checks.append(
                _status(
                    "WARN",
                    f"{label}: QA context width",
                    f"PropRAG qa_top_k={prop_qa_top_k}, prop_doc_counts={sorted(set(prop_doc_counts))}, "
                    f"dtc_baseline_counts={sorted(set(baseline_counts))}, dtc_selector_counts={sorted(set(selector_counts))}",
                )
            )

        prop_max_tokens = prop_config.get("max_new_tokens")
        if prop_max_tokens == 2048:
            checks.append(_status("PASS", f"{label}: PropRAG max_new_tokens", "2048"))
        else:
            checks.append(_status("WARN", f"{label}: PropRAG max_new_tokens", str(prop_max_tokens)))

        checks.append(
            _status(
                "WARN",
                f"{label}: DtC QA max_new_tokens",
                "DtC report does not serialize qa max_new_tokens; build_config currently sets BaseConfig.max_new_tokens=None.",
            )
        )

    checks.append(
        _status(
            "WARN",
            "fixed-pool comparability",
            "PropRAG retrieves from its own proposition graph; DtC selects from HippoRAG top-100. This is not same fixed-pool composition.",
        )
    )
    return checks


def audit_gold_and_nothink(
    *,
    prop_root: Path,
    paths: Mapping[str, DatasetPaths],
    reports: Mapping[str, Dict[str, Mapping[str, Any]]],
) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []
    proprag_py = prop_root / "src" / "proprag" / "PropRAG.py"
    main_py = prop_root / "main.py"
    enhanced_openie = prop_root / "src" / "proprag" / "information_extraction" / "enhanced_openie.py"
    prop_extract = prop_root / "src" / "proprag" / "information_extraction" / "proposition_extraction.py"
    openai_gpt = prop_root / "src" / "proprag" / "llm" / "openai_gpt.py"

    qa_body = _qa_function_body(proprag_py)
    if "gold_docs" not in qa_body and "gold_answers" not in qa_body:
        checks.append(_status("PASS", "PropRAG QA prompt source", "qa() body uses query_solution.docs[:qa_top_k], not gold fields"))
    else:
        checks.append(_status("FAIL", "PropRAG QA prompt source", "qa() body mentions gold fields"))

    top_k_line = _line_no(proprag_py, r"top_k_docs\s*=")
    eval_line = _line_no(proprag_py, r"calculate_metric_scores\(gold_docs=gold_docs")
    if top_k_line and eval_line and top_k_line < eval_line:
        checks.append(
            _status(
                "PASS",
                "PropRAG gold_docs retrieval use",
                f"retrieved docs are built before recall evaluation: top_k_docs line {top_k_line}, recall line {eval_line}",
            )
        )
    else:
        checks.append(
            _status(
                "WARN",
                "PropRAG gold_docs retrieval use",
                f"manual check needed: top_k_docs line {top_k_line}, recall line {eval_line}",
            )
        )

    for dataset in DATASETS:
        prop_examples = reports[dataset]["prop"].get("examples") or []
        label = DISPLAY_NAMES[dataset]
        bad_gold_keys = []
        for idx, row in enumerate(prop_examples):
            keys = set(row.keys())
            forbidden = sorted(k for k in keys if "gold" in k.lower() and k != "gold_answers")
            if forbidden:
                bad_gold_keys.append({"idx": idx, "keys": forbidden})
        if bad_gold_keys:
            checks.append(_status("FAIL", f"{label}: PropRAG examples gold leakage fields", str(bad_gold_keys[:3])))
        else:
            checks.append(_status("PASS", f"{label}: PropRAG examples gold leakage fields", "only gold_answers is serialized; no gold_docs in examples"))

    source_checks = {
        "PropRAG QA no-think": proprag_py,
        "EnhancedOpenIE no-think": enhanced_openie,
        "Proposition extraction no-think": prop_extract,
        "OpenAI wrapper strips think envelope": openai_gpt,
    }
    for label, path in source_checks.items():
        text = _read_text(path)
        if "/no_think" in text or "_strip_qwen_think_envelope" in text or "_strip_think_blocks" in text:
            checks.append(_status("PASS", label, str(path)))
        else:
            checks.append(_status("WARN", label, f"no no-think/strip marker found in {path}"))

    text_artifacts: List[Path] = []
    for dataset, ds_paths in paths.items():
        prop_report = reports[dataset]["prop"]
        save_dir = Path(prop_report.get("config", {}).get("save_dir", ""))
        if not save_dir.is_absolute():
            save_dir = prop_root / save_dir
        text_artifacts.extend([ds_paths.prop_json, ds_paths.prop_log])
        text_artifacts.extend(save_dir.glob("openie_results_ner_*.json"))

    think_hits = _scan_text_artifacts_for_think(text_artifacts)
    if think_hits:
        checks.append(_status("FAIL", "PropRAG clean text artifacts contain think tags", "; ".join(think_hits[:10])))
    else:
        checks.append(_status("PASS", "PropRAG clean text artifacts contain think tags", "no <think> / </think> tags found in JSON/log/OpenIE text artifacts"))

    return checks


def render_status_table(checks: Sequence[Mapping[str, str]]) -> List[str]:
    lines = ["| Status | Check | Evidence |", "| --- | --- | --- |"]
    for row in checks:
        lines.append(f"| {row['status']} | {row['check']} | {row['evidence']} |")
    return lines


def render_alignment_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| Dataset | Dataset File First-100 | Saved Examples Compared | Saved Examples Match | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        notes = "PropRAG output serializes only first 10 examples" if row["prop_saved_examples"] < row["dataset_file_first_n"] else ""
        lines.append(
            f"| {row['display']} | {row['dataset_file_first_n_match']} | "
            f"{row['saved_examples_compared']} | {row['saved_examples_match']} | {notes} |"
        )
    return lines


def render_dtc_same_pool_table(metrics: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| Dataset | HippoRAG Baseline EM | HippoRAG Baseline F1 | DtC EM | DtC F1 | Delta EM | Delta F1 | DtC R@5 | DtC R@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['display']} | {_fmt(row['hippo_baseline_em'])} | {_fmt(row['hippo_baseline_f1'])} | "
            f"{_fmt(row['dtc_em'])} | {_fmt(row['dtc_f1'])} | {_fmt(row['dtc_delta_em'])} | "
            f"{_fmt(row['dtc_delta_f1'])} | {_fmt(row['dtc_recall5'])} | {_fmt(row['dtc_recall20'])} |"
        )
    return lines


def render_cross_system_table(metrics: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| Dataset | DtC EM | DtC F1 | PropRAG EM | PropRAG F1 | F1: DtC-PropRAG | DtC R@5 | PropRAG R@5 | Fixed-Pool Comparable? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in metrics:
        f1_delta = None
        if row["dtc_f1"] is not None and row["prop_f1"] is not None:
            f1_delta = row["dtc_f1"] - row["prop_f1"]
        lines.append(
            f"| {row['display']} | {_fmt(row['dtc_em'])} | {_fmt(row['dtc_f1'])} | "
            f"{_fmt(row['prop_em'])} | {_fmt(row['prop_f1'])} | {_fmt(f1_delta)} | "
            f"{_fmt(row['dtc_recall5'])} | {_fmt(row['prop_recall5'])} | No: different retrieved pools |"
        )
    return lines


def render_audit_md(payload: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# PropRAG / DtC Protocol Audit Report",
        "",
        "Date: 2026-04-23",
        "",
        "## Bottom Line",
        "",
        "- HippoRAG baseline vs DtC inside each DtC JSON is a valid same-pool comparison.",
        "- PropRAG clean no-think 100 vs DtC is aligned on broad runtime conditions, but not on fixed candidate pool.",
        "- PropRAG should be labeled as a cross-system same-reader/same-embedding baseline unless it is adapted to consume HippoRAG's top-100 pool.",
        "",
        "## Query Alignment",
        "",
    ]
    lines.extend(render_alignment_table(payload["query_alignment"]))
    lines.extend(["", "## Protocol Checks", ""])
    lines.extend(render_status_table(payload["protocol_checks"]))
    lines.extend(["", "## Gold-Leakage / No-Think Checks", ""])
    lines.extend(render_status_table(payload["gold_nothink_checks"]))
    lines.extend(["", "## Key Caveats", ""])
    lines.extend(
        [
            "- PropRAG JSON currently saves only 10 examples, so saved-example query alignment is partial. Dataset-file first-100 alignment is the stronger check.",
            "- PropRAG uses its own proposition graph retrieval. This is intentionally stronger/different than a same-pool selector baseline.",
            "- DtC reports do not serialize QA `max_new_tokens`; the current HippoRAG `build_config` sets `BaseConfig.max_new_tokens=None`, while PropRAG explicitly uses 2048.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_comparison_md(payload: Mapping[str, Any]) -> str:
    metrics = payload["metrics"]
    lines: List[str] = [
        "# PropRAG-100 vs DtC-100 Comparison",
        "",
        "Date: 2026-04-23",
        "",
        "## Comparison Labels",
        "",
        "- `HippoRAG baseline -> DtC`: same-pool composition comparison. DtC selects from the same HippoRAG top-100 used to evaluate the baseline.",
        "- `PropRAG -> DtC`: cross-system comparison. Same broad runtime budget, but different retriever/pool.",
        "",
        "## Same-Pool DtC Result",
        "",
    ]
    lines.extend(render_dtc_same_pool_table(metrics))
    lines.extend(["", "## Cross-System PropRAG Comparison", ""])
    lines.extend(render_cross_system_table(metrics))
    lines.extend(["", "## Interpretation", ""])
    lines.extend(
        [
            "- DtC repairable-filter pilot100 is three-dataset positive against its own HippoRAG baseline.",
            "- PropRAG is much stronger on 2Wiki at pilot100, roughly tied/slightly lower on HotpotQA, and slightly stronger on MuSiQue F1.",
            "- Because PropRAG does not use the same fixed top-100 pool, this table cannot be used to claim that PropRAG is a stronger/weaker evidence composer under the fixed-pool protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hippo-root", default="/mnt/nvme/code/HippoRAG")
    parser.add_argument("--prop-root", default="/mnt/nvme/code/PropRAG")
    parser.add_argument(
        "--prop-100-dir",
        default="/mnt/nvme/code/PropRAG/run_logs/proprag_clean_nothink_rebuild3_20260423",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", default="/mnt/nvme/code/HippoRAG/docs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hippo_root = Path(args.hippo_root)
    prop_root = Path(args.prop_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = build_paths(args)
    reports: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for dataset, ds_paths in paths.items():
        reports[dataset] = {
            "prop": _load_json(ds_paths.prop_json),
            "dtc": _load_json(ds_paths.dtc_json),
        }

    query_alignment = [
        audit_dataset_alignment(
            dataset=dataset,
            prop_root=prop_root,
            hippo_root=hippo_root,
            prop_report=reports[dataset]["prop"],
            dtc_report=reports[dataset]["dtc"],
            limit=int(args.limit),
        )
        for dataset in DATASETS
    ]
    protocol_checks = audit_protocol(
        prop_root=prop_root,
        hippo_root=hippo_root,
        paths=paths,
        reports=reports,
    )
    gold_nothink_checks = audit_gold_and_nothink(
        prop_root=prop_root,
        paths=paths,
        reports=reports,
    )
    metrics = [
        collect_metrics(dataset, reports[dataset]["prop"], reports[dataset]["dtc"])
        for dataset in DATASETS
    ]

    audit_payload = {
        "query_alignment": query_alignment,
        "protocol_checks": protocol_checks,
        "gold_nothink_checks": gold_nothink_checks,
    }
    comparison_payload = {"metrics": metrics}

    audit_json = output_dir / "proprag_protocol_audit_report_20260423.json"
    audit_md = output_dir / "proprag_protocol_audit_report_20260423.md"
    comparison_json = output_dir / "proprag_dtc_pilot100_comparison_20260423.json"
    comparison_md = output_dir / "proprag_dtc_pilot100_comparison_20260423.md"

    audit_json.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_md.write_text(render_audit_md(audit_payload), encoding="utf-8")
    comparison_json.write_text(json.dumps(comparison_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_md.write_text(render_comparison_md(comparison_payload), encoding="utf-8")

    print(f"Wrote {audit_md}")
    print(f"Wrote {audit_json}")
    print(f"Wrote {comparison_md}")
    print(f"Wrote {comparison_json}")


if __name__ == "__main__":
    main()
