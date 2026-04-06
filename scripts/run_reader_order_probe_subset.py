#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import load_dataset, resolve_save_dir  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stringify_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return "true" if text in {"1", "true", "yes"} else "false"


def _question_key(question: str) -> str:
    return str(question or "").strip()


def materialize_subset_dataset(dataset: str,
                               subset_dataset: str,
                               selected_questions: Sequence[str]) -> Dict[str, Any]:
    _, samples = load_dataset(dataset, limit=0)
    selected_question_keys = {_question_key(question) for question in selected_questions}
    subset_samples = [
        sample
        for sample in samples
        if _question_key(sample.get("question", "")) in selected_question_keys
    ]
    subset_path = ROOT_DIR / "reproduce" / "dataset" / f"{subset_dataset}.json"
    subset_corpus_path = ROOT_DIR / "reproduce" / "dataset" / f"{subset_dataset}_corpus.json"
    subset_path.write_text(json.dumps(subset_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    if subset_corpus_path.exists() or subset_corpus_path.is_symlink():
        subset_corpus_path.unlink()
    base_corpus_path = ROOT_DIR / "reproduce" / "dataset" / f"{dataset}_corpus.json"
    subset_corpus_path.symlink_to(base_corpus_path.resolve())
    return {
        "subset_path": subset_path,
        "subset_corpus_path": subset_corpus_path,
        "subset_samples": subset_samples,
    }


def ensure_save_dir_symlink(dataset: str, subset_dataset: str, save_dir_root: str) -> Path:
    base_save_dir = ROOT_DIR / resolve_save_dir(save_dir_root, dataset)
    subset_save_dir = ROOT_DIR / resolve_save_dir(save_dir_root, subset_dataset)
    if subset_save_dir.exists() or subset_save_dir.is_symlink():
        return subset_save_dir
    subset_save_dir.symlink_to(base_save_dir.resolve())
    return subset_save_dir


def select_questions_from_gap_report(reader_gap_report: Dict[str, Any], quadrant_prefix: str) -> List[str]:
    questions = []
    for record in list(reader_gap_report.get("records", []) or []):
        quadrant = str(record.get("quadrant", ""))
        if quadrant.startswith(quadrant_prefix):
            questions.append(_question_key(record.get("question", "")))
    return [question for question in questions if question]


def append_arg(command: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_eval_command(base_report: Dict[str, Any],
                       subset_dataset: str,
                       subset_limit: int,
                       output_json: Path,
                       save_dir_root: str,
                       probe_mode: str) -> List[str]:
    cfg = dict(base_report.get("config", {}) or {})
    qa_block = dict(base_report.get("setwise_selector_qa", {}) or {})
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "eval_causal_qwen3.py"),
        "--dataset", subset_dataset,
        "--limit", str(int(subset_limit)),
        "--save_dir", str(save_dir_root),
        "--llm_name", str(base_report.get("llm_name", "qwen3-8b")),
        "--llm_request_name", str(base_report.get("llm_request_name", base_report.get("llm_name", "qwen3-8b"))),
        "--llm_base_url", str(base_report.get("llm_base_url", "http://localhost:8043/v1")),
        "--embedding_name", str(base_report.get("embedding_name", "VLLM//mnt/nvme/Qwen3-Embedding-8B")),
        "--embedding_base_url", str(base_report.get("embedding_base_url", "http://localhost:8018/v1/embeddings")),
        "--openie_mode", "online",
        "--causal_enabled", _stringify_bool(cfg.get("causal_enabled", False)),
        "--causal_engine_version", str(cfg.get("causal_engine_version", "v2")),
        "--causal_v2_graph_mode", str(cfg.get("causal_v2_graph_mode", "causal")),
        "--causal_v2_base_retrieval_mode", str(cfg.get("causal_v2_base_retrieval_mode", "legacy_fact_graph")),
        "--qa_top_k", str(int(cfg.get("qa_top_k", 5) or 5)),
        "--setwise_selector", str(qa_block.get("selector", cfg.get("setwise_selector", "bridge_beam"))),
        "--setwise_score_mode", str(qa_block.get("score_mode", cfg.get("setwise_score_mode", "bridge"))),
        "--setwise_pool_k", str(int(qa_block.get("pool_k", cfg.get("setwise_pool_k", 100)) or 100)),
        "--setwise_anchor_count", str(int(qa_block.get("anchor_count", cfg.get("setwise_anchor_count", 2)) or 2)),
        "--setwise_reserve_top_m", str(int(qa_block.get("reserve_top_m", cfg.get("setwise_reserve_top_m", 3)) or 3)),
        "--setwise_max_bridge_slots", str(int(qa_block.get("max_bridge_slots", cfg.get("setwise_max_bridge_slots", 0)) or 0)),
        "--setwise_structure_max_hops", str(int(cfg.get("setwise_structure_max_hops", 2) or 2)),
        "--setwise_base_weight", str(float(cfg.get("setwise_base_weight", 0.25) or 0.25)),
        "--setwise_structure_weight", str(float(cfg.get("setwise_structure_weight", 0.60) or 0.60)),
        "--setwise_novelty_weight", str(float(cfg.get("setwise_novelty_weight", 0.15) or 0.15)),
        "--setwise_non_anchor_title_dedup", _stringify_bool(qa_block.get("non_anchor_title_dedup", True)),
        "--setwise_query_entity_source", str(qa_block.get("query_entity_source", "seed")),
        "--setwise_gate_mode", str(qa_block.get("gate_mode", "none")),
        "--setwise_gate_min_structure_score", str(float(qa_block.get("gate_min_structure_score", 0.15) or 0.15)),
        "--setwise_gate_min_combined_margin", str(float(qa_block.get("gate_min_combined_margin", 0.0) or 0.0)),
        "--setwise_gate_min_closure_score", str(float(qa_block.get("gate_min_closure_score", 0.0) or 0.0)),
        "--setwise_gate_min_novelty_score", str(float(qa_block.get("gate_min_novelty_score", 0.0) or 0.0)),
        "--setwise_gate_min_frontier_gain", str(float(qa_block.get("gate_min_frontier_gain", 0.0) or 0.0)),
        "--setwise_gate_min_path_coherence", str(float(qa_block.get("gate_min_path_coherence", 0.0) or 0.0)),
        "--setwise_gate_max_avg_local_structure", str(float(qa_block.get("gate_max_avg_local_structure", 0.95) or 0.95)),
        "--setwise_gate_min_suffix_base_mean", str(float(qa_block.get("gate_min_suffix_base_mean", 0.15) or 0.15)),
        "--setwise_beam_width", str(int(cfg.get("setwise_beam_width", 4) or 4)),
        "--setwise_beam_expand_per_state", str(int(cfg.get("setwise_beam_expand_per_state", 4) or 4)),
        "--setwise_beam_projected_shortlist_factor", str(int(cfg.get("setwise_beam_projected_shortlist_factor", 1) or 1)),
        "--setwise_late_rerank_enabled", _stringify_bool(cfg.get("setwise_late_rerank_enabled", False)),
        "--setwise_late_rerank_candidate_count", str(int(cfg.get("setwise_late_rerank_candidate_count", 4) or 4)),
        "--setwise_late_rerank_include_baseline", _stringify_bool(cfg.get("setwise_late_rerank_include_baseline", True)),
        "--setwise_late_rerank_doc_char_limit", str(int(cfg.get("setwise_late_rerank_doc_char_limit", 280) or 280)),
        "--setwise_late_rerank_policy", str(cfg.get("setwise_late_rerank_policy", "always")),
        "--setwise_late_rerank_max_state_score_gap", str(float(cfg.get("setwise_late_rerank_max_state_score_gap", 0.0) or 0.0)),
        "--setwise_late_rerank_judge_backend", str(cfg.get("setwise_late_rerank_judge_backend", "inherit")),
        "--setwise_late_rerank_judge_model", str(cfg.get("setwise_late_rerank_judge_model", base_report.get("llm_request_name", base_report.get("llm_name", "qwen3-8b")))),
        "--setwise_late_rerank_judge_base_url", str(cfg.get("setwise_late_rerank_judge_base_url", base_report.get("llm_base_url", "http://localhost:8043/v1"))),
        "--structure_relation_probe_mode", str(cfg.get("structure_relation_probe_mode", "off")),
        "--structure_continuity_probe_mode", str(cfg.get("structure_continuity_probe_mode", "off")),
        "--structure_seed_target_bridge_mode", str(cfg.get("structure_seed_target_bridge_mode", "off")),
        "--structure_rerank_enabled", _stringify_bool(cfg.get("structure_rerank_enabled", False)),
        "--structure_rerank_top_n", str(int(cfg.get("structure_rerank_top_n", 40) or 40)),
        "--structure_rerank_bonus_weight", str(float(cfg.get("structure_rerank_bonus_weight", 0.08) or 0.08)),
        "--structure_rerank_min_edge_support", str(int(cfg.get("structure_rerank_min_edge_support", 2) or 2)),
        "--structure_rerank_max_top5_swaps", str(int(cfg.get("structure_rerank_max_top5_swaps", 2) or 2)),
        "--structure_rerank_seed_top_k", str(int(cfg.get("structure_rerank_seed_top_k", 4) or 4)),
        "--structure_rerank_max_hops", str(int(cfg.get("structure_rerank_max_hops", 2) or 2)),
        "--structure_rerank_margin_threshold", str(float(cfg.get("structure_rerank_margin_threshold", 0.02) or 0.02)),
        "--setwise_reader_order_probe_mode", str(probe_mode),
        "--output_json", str(output_json),
    ]
    append_arg(command, "--setwise_late_rerank_judge_reasoning_effort", cfg.get("setwise_late_rerank_judge_reasoning_effort"))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired subset evals for the reader-order probe.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate_report", required=True, help="Frozen candidate report used as the control config.")
    parser.add_argument("--reader_gap_json", required=True, help="Reader-gap JSON with query-level quadrants.")
    parser.add_argument("--save_dir_root", default="outputs_step0_general")
    parser.add_argument("--subset_dataset", default="")
    parser.add_argument("--quadrant_prefix", default="selector_improved__")
    parser.add_argument("--probe_mode", default="promote_best_bridge_to_slot3")
    parser.add_argument("--control_output_json", default="")
    parser.add_argument("--probe_output_json", default="")
    parser.add_argument("--analysis_output_json", default="")
    parser.add_argument("--analysis_output_md", default="")
    parser.add_argument("--force_rerun_control", action="store_true")
    parser.add_argument("--force_rerun_probe", action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    args = parser.parse_args()

    candidate_report = load_json(Path(args.candidate_report))
    reader_gap_report = load_json(Path(args.reader_gap_json))
    selected_questions = select_questions_from_gap_report(
        reader_gap_report=reader_gap_report,
        quadrant_prefix=str(args.quadrant_prefix),
    )
    subset_dataset = str(
        args.subset_dataset
        or f"{args.dataset}_readerprobe_{str(args.probe_mode).replace('promote_best_bridge_to_', '')}"
    )
    subset_artifacts = materialize_subset_dataset(
        dataset=str(args.dataset),
        subset_dataset=subset_dataset,
        selected_questions=selected_questions,
    )
    ensure_save_dir_symlink(
        dataset=str(args.dataset),
        subset_dataset=subset_dataset,
        save_dir_root=str(args.save_dir_root),
    )

    base_eval_dir = ROOT_DIR / resolve_save_dir(str(args.save_dir_root), str(args.dataset)) / "eval_reports"
    base_eval_dir.mkdir(parents=True, exist_ok=True)
    control_output_json = Path(args.control_output_json) if args.control_output_json else (
        base_eval_dir / f"{subset_dataset}_candidate_control.json"
    )
    probe_output_json = Path(args.probe_output_json) if args.probe_output_json else (
        base_eval_dir / f"{subset_dataset}_{str(args.probe_mode)}.json"
    )
    analysis_output_json = Path(args.analysis_output_json) if args.analysis_output_json else (
        base_eval_dir / f"{subset_dataset}_{str(args.probe_mode)}.analysis.json"
    )
    analysis_output_md = Path(args.analysis_output_md) if args.analysis_output_md else (
        base_eval_dir / f"{subset_dataset}_{str(args.probe_mode)}.analysis.md"
    )

    control_command = build_eval_command(
        base_report=candidate_report,
        subset_dataset=subset_dataset,
        subset_limit=len(subset_artifacts["subset_samples"]),
        output_json=control_output_json,
        save_dir_root=str(args.save_dir_root),
        probe_mode="none",
    )
    probe_command = build_eval_command(
        base_report=candidate_report,
        subset_dataset=subset_dataset,
        subset_limit=len(subset_artifacts["subset_samples"]),
        output_json=probe_output_json,
        save_dir_root=str(args.save_dir_root),
        probe_mode=str(args.probe_mode),
    )

    if selected_questions and (args.force_rerun_control or not control_output_json.exists()):
        subprocess.run(control_command, cwd=str(ROOT_DIR), check=True)
    if selected_questions and (args.force_rerun_probe or not probe_output_json.exists()):
        subprocess.run(probe_command, cwd=str(ROOT_DIR), check=True)

    if not args.skip_analysis and selected_questions:
        analysis_command = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "analyze_reader_order_probe.py"),
            "--control_report", str(control_output_json),
            "--probe_report", str(probe_output_json),
            "--output_json", str(analysis_output_json),
            "--output_md", str(analysis_output_md),
        ]
        subprocess.run(analysis_command, cwd=str(ROOT_DIR), check=True)

    result = {
        "dataset": str(args.dataset),
        "subset_dataset": subset_dataset,
        "subset_dataset_path": str(subset_artifacts["subset_path"]),
        "subset_corpus_path": str(subset_artifacts["subset_corpus_path"]),
        "selected_question_count": len(selected_questions),
        "control_output_json": str(control_output_json),
        "probe_output_json": str(probe_output_json),
        "analysis_output_json": str(analysis_output_json),
        "analysis_output_md": str(analysis_output_md),
        "control_command": control_command,
        "probe_command": probe_command,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
