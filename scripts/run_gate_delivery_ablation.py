#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


VARIANT_SPECS: Dict[str, Dict[str, Any]] = {
    "satguard": {},
    "ungated": {
        "gate_mode": "none",
    },
    "no_satguard": {
        "gate_mode": "suffix_bridge",
    },
    "lower_structure_threshold": {
        "gate_mode": "suffix_bridge_saturation_guard",
        "gate_min_structure_score": 0.0,
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stringify_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return "true" if text in {"1", "true", "yes"} else "false"


def append_arg(command: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def normalize_variant_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if normalized not in VARIANT_SPECS:
        raise ValueError(f"Unknown gate ablation variant: {name}")
    return normalized


def build_eval_command(base_report: Dict[str, Any],
                       output_json: Path,
                       save_dir_root: str,
                       limit: int,
                       overrides: Mapping[str, Any] | None = None) -> List[str]:
    cfg = dict(base_report.get("config", {}) or {})
    qa_block = dict(base_report.get("setwise_selector_qa", {}) or {})
    overrides = dict(overrides or {})

    def _value(key: str, default: Any) -> Any:
        if key in overrides:
            return overrides[key]
        if key in qa_block:
            return qa_block[key]
        return cfg.get(key, default)

    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "eval_causal_qwen3.py"),
        "--dataset", str(base_report.get("dataset", cfg.get("dataset", "musique"))),
        "--limit", str(int(limit)),
        "--save_dir", str(save_dir_root),
        "--llm_name", str(base_report.get("llm_name", "qwen3-8b")),
        "--llm_request_name", str(base_report.get("llm_request_name", base_report.get("llm_name", "qwen3-8b"))),
        "--llm_base_url", str(base_report.get("llm_base_url", "http://localhost:8043/v1")),
        "--embedding_name", str(base_report.get("embedding_name", "VLLM//mnt/nvme/Qwen3-Embedding-8B")),
        "--embedding_base_url", str(base_report.get("embedding_base_url", "http://localhost:8018/v1/embeddings")),
        "--openie_mode", "online",
        "--causal_enabled", stringify_bool(cfg.get("causal_enabled", False)),
        "--causal_engine_version", str(cfg.get("causal_engine_version", "v2")),
        "--causal_v2_graph_mode", str(cfg.get("causal_v2_graph_mode", "causal")),
        "--causal_v2_base_retrieval_mode", str(cfg.get("causal_v2_base_retrieval_mode", "legacy_fact_graph")),
        "--qa_top_k", str(int(cfg.get("qa_top_k", qa_block.get("qa_top_k", 5)) or 5)),
        "--setwise_selector", str(qa_block.get("selector", cfg.get("setwise_selector", "bridge_beam"))),
        "--setwise_score_mode", str(qa_block.get("score_mode", cfg.get("setwise_score_mode", "bridge"))),
        "--setwise_pool_k", str(int(qa_block.get("pool_k", cfg.get("setwise_pool_k", 100)) or 100)),
        "--setwise_anchor_count", str(int(qa_block.get("anchor_count", cfg.get("setwise_anchor_count", 2)) or 2)),
        "--setwise_reserve_top_m", str(int(_value("reserve_top_m", cfg.get("setwise_reserve_top_m", 3)) or 3)),
        "--setwise_max_bridge_slots", str(int(_value("max_bridge_slots", cfg.get("setwise_max_bridge_slots", 0)) or 0)),
        "--setwise_structure_max_hops", str(int(cfg.get("setwise_structure_max_hops", 2) or 2)),
        "--setwise_base_weight", str(float(cfg.get("setwise_base_weight", 0.25) or 0.25)),
        "--setwise_structure_weight", str(float(cfg.get("setwise_structure_weight", 0.60) or 0.60)),
        "--setwise_novelty_weight", str(float(cfg.get("setwise_novelty_weight", 0.15) or 0.15)),
        "--setwise_non_anchor_title_dedup", stringify_bool(_value("non_anchor_title_dedup", True)),
        "--setwise_query_entity_source", str(_value("query_entity_source", "seed")),
        "--setwise_gate_mode", str(_value("gate_mode", "none")),
        "--setwise_gate_min_structure_score", str(float(_value("gate_min_structure_score", 0.15) or 0.0)),
        "--setwise_gate_min_combined_margin", str(float(_value("gate_min_combined_margin", 0.0) or 0.0)),
        "--setwise_gate_min_closure_score", str(float(_value("gate_min_closure_score", 0.0) or 0.0)),
        "--setwise_gate_min_novelty_score", str(float(_value("gate_min_novelty_score", 0.0) or 0.0)),
        "--setwise_gate_min_frontier_gain", str(float(_value("gate_min_frontier_gain", 0.0) or 0.0)),
        "--setwise_gate_min_path_coherence", str(float(_value("gate_min_path_coherence", 0.0) or 0.0)),
        "--setwise_gate_max_avg_local_structure", str(float(_value("gate_max_avg_local_structure", 0.95) or 0.95)),
        "--setwise_gate_min_suffix_base_mean", str(float(_value("gate_min_suffix_base_mean", 0.15) or 0.15)),
        "--setwise_beam_width", str(int(cfg.get("setwise_beam_width", 4) or 4)),
        "--setwise_beam_expand_per_state", str(int(cfg.get("setwise_beam_expand_per_state", 4) or 4)),
        "--setwise_beam_projected_shortlist_factor", str(int(cfg.get("setwise_beam_projected_shortlist_factor", 1) or 1)),
        "--setwise_late_rerank_enabled", stringify_bool(cfg.get("setwise_late_rerank_enabled", False)),
        "--setwise_late_rerank_candidate_count", str(int(cfg.get("setwise_late_rerank_candidate_count", 4) or 4)),
        "--setwise_late_rerank_include_baseline", stringify_bool(cfg.get("setwise_late_rerank_include_baseline", True)),
        "--setwise_late_rerank_doc_char_limit", str(int(cfg.get("setwise_late_rerank_doc_char_limit", 280) or 280)),
        "--setwise_late_rerank_policy", str(cfg.get("setwise_late_rerank_policy", "always")),
        "--setwise_late_rerank_max_state_score_gap", str(float(cfg.get("setwise_late_rerank_max_state_score_gap", 0.0) or 0.0)),
        "--setwise_late_rerank_judge_backend", str(cfg.get("setwise_late_rerank_judge_backend", "inherit")),
        "--setwise_late_rerank_judge_model", str(cfg.get("setwise_late_rerank_judge_model", base_report.get("llm_request_name", base_report.get("llm_name", "qwen3-8b")))),
        "--setwise_late_rerank_judge_base_url", str(cfg.get("setwise_late_rerank_judge_base_url", base_report.get("llm_base_url", "http://localhost:8043/v1"))),
        "--structure_relation_probe_mode", str(cfg.get("structure_relation_probe_mode", "off")),
        "--structure_continuity_probe_mode", str(cfg.get("structure_continuity_probe_mode", "off")),
        "--structure_seed_target_bridge_mode", str(cfg.get("structure_seed_target_bridge_mode", "off")),
        "--structure_rerank_enabled", stringify_bool(cfg.get("structure_rerank_enabled", False)),
        "--structure_rerank_top_n", str(int(cfg.get("structure_rerank_top_n", 40) or 40)),
        "--structure_rerank_bonus_weight", str(float(cfg.get("structure_rerank_bonus_weight", 0.08) or 0.08)),
        "--structure_rerank_min_edge_support", str(int(cfg.get("structure_rerank_min_edge_support", 2) or 2)),
        "--structure_rerank_max_top5_swaps", str(int(cfg.get("structure_rerank_max_top5_swaps", 2) or 2)),
        "--structure_rerank_seed_top_k", str(int(cfg.get("structure_rerank_seed_top_k", 4) or 4)),
        "--structure_rerank_max_hops", str(int(cfg.get("structure_rerank_max_hops", 2) or 2)),
        "--structure_rerank_margin_threshold", str(float(cfg.get("structure_rerank_margin_threshold", 0.02) or 0.02)),
        "--max_retry_attempts", "12",
        "--output_json", str(output_json),
    ]
    append_arg(command, "--setwise_late_rerank_judge_reasoning_effort", cfg.get("setwise_late_rerank_judge_reasoning_effort"))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal gate-delivery ablations from a frozen bridge-beam report.")
    parser.add_argument("--base_report", required=True, help="Frozen candidate report JSON, usually the satguard candidate.")
    parser.add_argument("--baseline_report", default="", help="Optional original control report for later analysis.")
    parser.add_argument("--save_dir_root", default="outputs_step0_general")
    parser.add_argument("--limit", type=int, default=0, help="Override query limit. Defaults to report limit.")
    parser.add_argument("--tag", default="", help="Optional output tag. Defaults to current date-like suffix from caller.")
    parser.add_argument("--variants", nargs="+", default=["ungated", "no_satguard", "lower_structure_threshold"])
    parser.add_argument("--summary_json", default="", help="Path for the ablation summary JSON.")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    base_report_path = Path(args.base_report)
    base_report = load_json(base_report_path)
    baseline_report_path = Path(args.baseline_report) if args.baseline_report else None
    dataset = str(base_report.get("dataset", "unknown"))
    limit = int(args.limit or base_report.get("limit", 0) or 0)
    if limit <= 0:
        raise ValueError("A positive limit is required, either in the report or via --limit.")

    eval_dir = ROOT_DIR / f"{str(args.save_dir_root).rstrip('/')}_{dataset}" / "eval_reports"
    eval_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag or "gateablation")

    summary_json = Path(args.summary_json) if args.summary_json else (
        eval_dir / f"{dataset}_gate_delivery_ablation_{tag}.summary.json"
    )

    results: Dict[str, Any] = {
        "dataset": dataset,
        "limit": limit,
        "base_report": str(base_report_path),
        "baseline_report": str(baseline_report_path) if baseline_report_path else "",
        "variants": {},
    }

    results["variants"]["satguard"] = {
        "report": str(base_report_path),
        "command": None,
        "overrides": VARIANT_SPECS["satguard"],
        "reused_existing_report": True,
    }

    for variant_name in args.variants:
        normalized = normalize_variant_name(variant_name)
        output_json = eval_dir / f"{dataset}_gate_delivery_{normalized}_{tag}.json"
        command = build_eval_command(
            base_report=base_report,
            output_json=output_json,
            save_dir_root=str(args.save_dir_root),
            limit=limit,
            overrides=VARIANT_SPECS[normalized],
        )
        if args.force_rerun or not output_json.exists():
            subprocess.run(command, cwd=str(ROOT_DIR), check=True)
        results["variants"][normalized] = {
            "report": str(output_json),
            "command": command,
            "overrides": VARIANT_SPECS[normalized],
            "reused_existing_report": False,
        }

    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
