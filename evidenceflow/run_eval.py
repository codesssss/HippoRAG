#!/usr/bin/env python3
"""Run the legacy integrated PCEC evaluator.

This is the integrated PCEC runner.  It keeps ETv3 variable-flow expansion as
the expander and replaces top-k readout with prefix-residual admission:

    query -> ET expansion pool -> PrefixResidualReadout -> reader top5

The implementation intentionally reuses the existing DBEC frozen-binding
noisy-OR selector path, but exposes only the PCEC hard-constraint interface.
It is retained for legacy parity; the main EvidenceLink protocol is ETv4
fact-witnessed STO pool plus PCEC native-pool readout.
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

from evidenceflow.contract import (  # noqa: E402
    DEFAULT_DTC_BINDING_MAX_CANDIDATES,
    DEFAULT_POOL_K,
    DEFAULT_PREFIX_BUDGET_M,
    DEFAULT_READER_BUDGET_K,
    DEFAULT_SAFE_MIN_OBJECTIVE_GAIN,
    DEFAULT_SAFE_MIN_SWAP_GAIN,
    METHOD_NAME,
    SELECTOR_NAME,
    residual_budget,
    validate_budgets,
)
from evidenceflow.readout import (  # noqa: E402
    enrich_payload,
    read_json,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path("run_logs/evidenceflow")
DEFAULT_HISTORICAL_POOL_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/pools")
DEFAULT_HISTORICAL_BINDING_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/evals")
DEFAULT_SAVE_DIR = "outputs_step0_general_nvembed"
DATASET_PORTS = {
    "musique": 8041,
    "hotpotqa": 8042,
    "2wikimultihopqa": 8043,
}


def default_pool_path(pool_root: Path, dataset: str, pool_k: int, limit: int) -> Path:
    return pool_root / f"{dataset}_etv3_pool{pool_k}_limit{limit}.json"


def default_output_path(
    output_root: Path,
    dataset: str,
    pool_k: int,
    reader_budget_k: int,
    prefix_budget_m: int,
    limit: int,
) -> Path:
    residual = residual_budget(reader_budget_k, prefix_budget_m)
    return (
        output_root
        / "evals"
        / f"{dataset}_pcec_prefix{prefix_budget_m}_residual{residual}_pool{pool_k}_limit{limit}.json"
    )


def default_binding_cache_path(binding_root: Path, dataset: str, pool_k: int) -> Path:
    return binding_root / f"{dataset}_etv3_pool{pool_k}_dbec_latest.binding_cache.json"


def dataset_default_llm_base_url(dataset: str) -> str:
    return f"http://localhost:{DATASET_PORTS.get(str(dataset), 8043)}/v1"


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
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool-k", type=int, default=DEFAULT_POOL_K)
    parser.add_argument("--setwise-pool-k", type=int, default=0)
    parser.add_argument("--reader-budget-k", type=int, default=DEFAULT_READER_BUDGET_K)
    parser.add_argument("--prefix-budget-m", type=int, default=DEFAULT_PREFIX_BUDGET_M)
    parser.add_argument("--qa-doc-max-chars", type=int, default=2048)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_HISTORICAL_POOL_ROOT)
    parser.add_argument("--binding-root", type=Path, default=DEFAULT_HISTORICAL_BINDING_ROOT)
    parser.add_argument("--retrieval-report", type=Path, default=None)
    parser.add_argument("--openie-results", type=Path, default=None)
    parser.add_argument("--pool-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--binding-cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-pool", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")

    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-request-name", default="qwen3-8b-train")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--max-retry-attempts", type=int, default=20)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--llm-binding-url", default="")
    parser.add_argument("--llm-binding-model", default="")
    parser.add_argument("--daec-safe-min-objective-gain", type=float, default=DEFAULT_SAFE_MIN_OBJECTIVE_GAIN)
    parser.add_argument("--daec-safe-min-swap-gain", type=float, default=DEFAULT_SAFE_MIN_SWAP_GAIN)
    parser.add_argument("--dtc-binding-max-candidates", type=int, default=DEFAULT_DTC_BINDING_MAX_CANDIDATES)
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


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    pool_k = int(args.pool_k)
    limit = int(args.limit)
    prefix_budget_m = int(args.prefix_budget_m)
    pool_json = (
        Path(args.pool_json).expanduser()
        if args.pool_json is not None
        else default_pool_path(Path(args.pool_root).expanduser(), str(args.dataset), pool_k, limit)
    )
    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json is not None
        else default_output_path(
            Path(args.output_root).expanduser(),
            str(args.dataset),
            pool_k,
            int(args.reader_budget_k),
            prefix_budget_m,
            limit,
        )
    )
    binding_cache_path = (
        Path(args.binding_cache_path).expanduser()
        if args.binding_cache_path is not None
        else default_binding_cache_path(Path(args.binding_root).expanduser(), str(args.dataset), pool_k)
    )
    return pool_json, output_json, binding_cache_path


def maybe_export_pool(args: argparse.Namespace, pool_json: Path) -> None:
    if pool_json.exists() and not bool(args.overwrite_pool):
        print(f"Reusing existing ET pool JSON: {pool_json}", flush=True)
        return
    if args.retrieval_report is None or args.openie_results is None:
        raise SystemExit(
            "Missing ET pool JSON. Provide --pool-json/--pool-root or pass "
            "--retrieval-report and --openie-results to export one."
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


def build_eval_command(
    args: argparse.Namespace,
    *,
    pool_json: Path,
    output_json: Path,
    binding_cache_path: Path,
    passthrough: Sequence[str],
) -> list[str]:
    reader_budget_k = int(args.reader_budget_k)
    prefix_budget_m = int(args.prefix_budget_m)
    residual = residual_budget(reader_budget_k, prefix_budget_m)
    llm_base_url = str(args.llm_base_url or dataset_default_llm_base_url(str(args.dataset)))
    setwise_pool_k = int(args.setwise_pool_k) if int(args.setwise_pool_k) > 0 else int(args.pool_k)
    command = [
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
        llm_base_url,
        "--embedding_name",
        str(args.embedding_name),
        "--embedding_base_url",
        str(args.embedding_base_url),
        "--external_pool_json",
        str(pool_json),
        "--external_pool_source_name",
        f"{METHOD_NAME}_etv3_pool{int(args.pool_k)}",
        "--external_pool_strict_questions",
        "true",
        "--setwise_selector",
        SELECTOR_NAME,
        "--setwise_pool_k",
        str(setwise_pool_k),
        "--qa_top_k",
        str(reader_budget_k),
        "--qa_doc_max_chars",
        str(int(args.qa_doc_max_chars)),
        "--dtc_decomposition_mode",
        "llm",
        "--dtc_binding_max_candidates",
        str(int(args.dtc_binding_max_candidates)),
        "--llm_binding_url",
        str(args.llm_binding_url or llm_base_url),
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
        str(residual),
        "--daec_safe_preserve_top_m",
        str(prefix_budget_m),
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
        command.append("--qwen_disable_thinking")
    command.extend(passthrough)
    return command


def postprocess_output(args: argparse.Namespace, output_json: Path) -> None:
    if bool(args.dry_run) or bool(args.skip_postprocess):
        return
    payload = read_json(output_json)
    enriched = enrich_payload(
        payload,
        reader_budget_k=int(args.reader_budget_k),
        prefix_budget_m=int(args.prefix_budget_m),
        pool_k=int(args.pool_k),
    )
    write_json(enriched, output_json)
    summary = enriched.get("pcec_summary", {})
    print(
        "[PCEC] postprocessed {path} changed={changed} swaps={swaps} title-all@5={title_all}".format(
            path=output_json,
            changed=summary.get("changed_count", "NA"),
            swaps=summary.get("total_swaps", "NA"),
            title_all=summary.get("pcec_title_all_gold_top5", "NA"),
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, passthrough = parser.parse_known_args(argv)
    validate_budgets(reader_budget_k=int(args.reader_budget_k), prefix_budget_m=int(args.prefix_budget_m))
    pool_json, output_json, binding_cache_path = resolve_paths(args)
    maybe_export_pool(args, pool_json)
    eval_cmd = build_eval_command(
        args,
        pool_json=pool_json,
        output_json=output_json,
        binding_cache_path=binding_cache_path,
        passthrough=passthrough,
    )
    if not bool(args.dry_run):
        output_json.parent.mkdir(parents=True, exist_ok=True)
    run_command(eval_cmd, dry_run=bool(args.dry_run))
    postprocess_output(args, output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
