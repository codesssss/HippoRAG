#!/usr/bin/env python3
"""Check whether Evidence Transition GraphRAG is installed in a target repo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PathCheck:
    path: str
    required: bool
    reason: str


CHECKS: tuple[PathCheck, ...] = (
    PathCheck("evidence_transition_graphrag/contract.py", True, "method identity"),
    PathCheck("evidence_transition_graphrag/run_fresh_e2e.py", True, "public runner"),
    PathCheck("run_query_grounded_sto_fresh_e2e.py", True, "fresh indexing runner"),
    PathCheck("evidence_transition_top5_qa_adapter.py", True, "non-overwriting QA adapter"),
    PathCheck(
        "source_authorized_vocab_strict_retrieval/e2e_pipeline.py",
        True,
        "retrieval pipeline",
    ),
    PathCheck(
        "source_authorized_vocab_strict_retrieval/candidate_generator.py",
        True,
        "candidate generator",
    ),
    PathCheck("src/agsto/local_graph.py", True, "STO local graph"),
    PathCheck("src/agsto/role_transition.py", True, "STO transition graph"),
    PathCheck("src/hipporag", True, "target HippoRAG base implementation"),
    PathCheck("reproduce/dataset", False, "benchmark data root"),
    PathCheck("run_transition_top5_qa.py", False, "target QA runner for --run-qa"),
)


def run_checks(target_root: Path) -> tuple[list[PathCheck], list[PathCheck]]:
    target = target_root.expanduser().resolve()
    missing_required: list[PathCheck] = []
    missing_optional: list[PathCheck] = []
    for check in CHECKS:
        exists = (target / check.path).exists()
        if exists:
            continue
        if check.required:
            missing_required.append(check)
        else:
            missing_optional.append(check)
    return missing_required, missing_optional


def print_table(
    target_root: Path,
    *,
    missing_required: Sequence[PathCheck],
    missing_optional: Sequence[PathCheck],
) -> None:
    target = target_root.expanduser().resolve()
    missing_required_paths = {check.path for check in missing_required}
    missing_optional_paths = {check.path for check in missing_optional}

    print(f"Target: {target}")
    print()
    print("| path | status | required | reason |")
    print("|---|---:|---:|---|")
    for check in CHECKS:
        if check.path in missing_required_paths:
            status = "missing"
        elif check.path in missing_optional_paths:
            status = "warning"
        else:
            status = "ok"
        required = "yes" if check.required else "no"
        print(f"| `{check.path}` | {status} | {required} | {check.reason} |")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=".", help="Target HippoRAG repository root.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    target = Path(args.target)
    missing_required, missing_optional = run_checks(target)
    print_table(
        target,
        missing_required=missing_required,
        missing_optional=missing_optional,
    )
    if missing_required:
        print()
        print("Result: incomplete")
        return 1
    print()
    if missing_optional:
        print("Result: complete for retrieval; optional QA/data components are missing.")
        return 0
    print("Result: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
