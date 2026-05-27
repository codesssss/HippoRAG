#!/usr/bin/env python3
"""Run the frozen ETv3 candidate pool with the baseline-stable DBEC selector.

This wrapper is intentionally thin.  It first converts an ETv3 retrieval report
to the external-pool format expected by ``scripts/eval_causal_qwen3.py`` and
then runs DBEC/DAEC over that fixed pool.  The default selector starts from the
ETv3 top5 and applies strict positive DBEC-objective local edits.  It does not
modify ETv3 retrieval, shared AG-STO code, or DAEC/DBEC selector internals.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence_transition_graphragv3_dbec_latest.contract import (  # noqa: E402
    DEFAULT_SAFE_MAX_SWAPS,
    DEFAULT_SAFE_MIN_OBJECTIVE_GAIN,
    DEFAULT_SAFE_MIN_SWAP_GAIN,
    DEFAULT_SAFE_PRESERVE_TOP_M,
    POOL_SOURCE_NAME,
    SELECTOR_NAME,
)


def str_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def default_pool_path(output_root: Path, dataset: str, pool_k: int, limit: int) -> Path:
    return output_root / "pools" / f"{dataset}_etv3_pool{pool_k}_limit{limit}.json"


def default_eval_path(output_root: Path, dataset: str, pool_k: int, limit: int, selector: str) -> Path:
    label = "dbec_stable" if str(selector) == "daec_noisyor_safe_llm" else "dbec_full_rebuild"
    return output_root / "evals" / f"{dataset}_etv3_pool{pool_k}_{label}_limit{limit}.json"


def default_binding_cache_path(output_root: Path, dataset: str, pool_k: int) -> Path:
    return output_root / "evals" / f"{dataset}_etv3_pool{pool_k}_dbec_latest.binding_cache.json"


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    repo = str(_REPO_ROOT)
    env["PYTHONPATH"] = repo if not existing else f"{repo}:{existing}"
    subprocess.run(list(command), cwd=str(_REPO_ROOT), env=env, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--setwise-pool-k", type=int, default=100)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--qa-doc-max-chars", type=int, default=2048)
    parser.add_argument(
        "--selector",
        choices=["daec_noisyor_safe_llm", "daec_noisyor_llm"],
        default=SELECTOR_NAME,
        help=(
            "DBEC selector mode. The default is baseline-stable local edit; "
            "daec_noisyor_llm reproduces the full-rebuild diagnostic."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("run_logs/etv3_dbec_latest"))
    parser.add_argument("--retrieval-report", type=Path, default=None)
    parser.add_argument("--openie-results", type=Path, default=None)
    parser.add_argument("--pool-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--binding-cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-pool", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--save-dir", default="outputs_step0_general_nvembed")
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-request-name", default="qwen3-8b-train")
    parser.add_argument("--llm-base-url", default="http://localhost:8043/v1")
    parser.add_argument("--max-retry-attempts", type=int, default=20)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--llm-binding-url", default="")
    parser.add_argument("--llm-binding-model", default="")
    parser.add_argument("--daec-safe-min-objective-gain", type=float, default=DEFAULT_SAFE_MIN_OBJECTIVE_GAIN)
    parser.add_argument("--daec-safe-min-swap-gain", type=float, default=DEFAULT_SAFE_MIN_SWAP_GAIN)
    parser.add_argument("--daec-safe-max-swaps", type=int, default=DEFAULT_SAFE_MAX_SWAPS)
    parser.add_argument("--daec-safe-preserve-top-m", type=int, default=DEFAULT_SAFE_PRESERVE_TOP_M)
    parser.add_argument(
        "--llm-binding-title-match-mode",
        default="wiki_title",
        choices=[
            "exact",
            "substring",
            "substring_guarded",
            "wiki_title",
            "title_link",
            "normalized_title",
            "entity_title",
            "wiki_title_unique",
            "title_link_unique",
            "normalized_title_unique",
            "entity_title_unique",
        ],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, passthrough = parser.parse_known_args(argv)

    output_root = Path(args.output_root).expanduser()
    pool_json = (
        Path(args.pool_json).expanduser()
        if args.pool_json is not None
        else default_pool_path(output_root, str(args.dataset), int(args.pool_k), int(args.limit))
    )
    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json is not None
        else default_eval_path(
            output_root,
            str(args.dataset),
            int(args.pool_k),
            int(args.limit),
            str(args.selector),
        )
    )
    binding_cache_path = (
        Path(args.binding_cache_path).expanduser()
        if args.binding_cache_path is not None
        else default_binding_cache_path(output_root, str(args.dataset), int(args.pool_k))
    )

    if args.overwrite_pool or not pool_json.exists():
        if args.retrieval_report is None or args.openie_results is None:
            raise SystemExit(
                "--retrieval-report and --openie-results are required when the pool JSON "
                "does not exist or --overwrite-pool is set."
            )
        export_cmd = [
            sys.executable,
            "scripts/export_evidence_transition_pool.py",
            "--dataset",
            str(args.dataset),
            "--limit",
            str(int(args.limit)),
            "--pool_k",
            str(int(args.pool_k)),
            "--retrieval_report",
            str(Path(args.retrieval_report).expanduser()),
            "--openie_results",
            str(Path(args.openie_results).expanduser()),
            "--output_json",
            str(pool_json),
        ]
        run_command(export_cmd, dry_run=bool(args.dry_run))
    else:
        print(f"Reusing existing pool JSON: {pool_json}", flush=True)

    eval_cmd = [
        sys.executable,
        "scripts/eval_causal_qwen3.py",
        "--dataset",
        str(args.dataset),
        "--limit",
        str(int(args.limit)),
        "--save_dir",
        str(args.save_dir),
        "--llm_name",
        str(args.llm_name),
        "--llm_request_name",
        str(args.llm_request_name),
        "--max_retry_attempts",
        str(int(args.max_retry_attempts)),
        "--llm_base_url",
        str(args.llm_base_url),
        "--embedding_name",
        str(args.embedding_name),
        "--embedding_base_url",
        str(args.embedding_base_url),
        "--external_pool_json",
        str(pool_json),
        "--external_pool_source_name",
        f"{POOL_SOURCE_NAME}",
        "--external_pool_strict_questions",
        "true",
        "--setwise_selector",
        str(args.selector),
        "--setwise_pool_k",
        str(int(args.setwise_pool_k)),
        "--qa_top_k",
        str(int(args.qa_top_k)),
        "--qa_doc_max_chars",
        str(int(args.qa_doc_max_chars)),
        "--dtc_decomposition_mode",
        "llm",
        "--dtc_binding_max_candidates",
        "5",
        "--llm_binding_url",
        str(args.llm_binding_url or args.llm_base_url),
        "--llm_binding_model",
        str(args.llm_binding_model or args.llm_request_name),
        "--llm_binding_cache_path",
        str(binding_cache_path),
        "--llm_binding_title_match_mode",
        str(args.llm_binding_title_match_mode),
        "--daec_safe_min_objective_gain",
        str(float(args.daec_safe_min_objective_gain)),
        "--daec_safe_min_swap_gain",
        str(float(args.daec_safe_min_swap_gain)),
        "--daec_safe_max_swaps",
        str(int(args.daec_safe_max_swaps)),
        "--daec_safe_preserve_top_m",
        str(int(args.daec_safe_preserve_top_m)),
        "--causal_enabled",
        "false",
        "--causal_engine_version",
        "v2",
        "--causal_v2_base_retrieval_mode",
        "dense",
        "--structure_rerank_enabled",
        "false",
        "--output_json",
        str(output_json),
    ]
    if bool(args.qwen_disable_thinking):
        eval_cmd.append("--qwen_disable_thinking")
    eval_cmd.extend(passthrough)
    run_command(eval_cmd, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
