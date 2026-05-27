#!/usr/bin/env python3
"""Export EvLink fresh-adapter pools for controlled ablations.

This utility materializes the same external-pool JSON shape consumed by
``evidenceflow/run_native_pool.py``, but obtains the pool from
``FrozenETv3Expander`` so that role-graph edge policies can be changed while
the downstream PCEC readout remains the native-pool protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evidenceflow.expander import FrozenETv3Expander  # noqa: E402
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.candidate_generator import (  # noqa: E402
    DEFAULT_ROLE_GRAPH_EDGE_POLICY,
    ROLE_GRAPH_EDGE_POLICIES,
)
from evidenceflow.readout import write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--et-candidate-pool-k", type=int, default=200)
    parser.add_argument("--reader-budget-k", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=Path("reproduce/dataset"))
    parser.add_argument("--frozen-runs-root", type=Path, required=True)
    parser.add_argument("--llm-name", default="qwen3-32b-judge")
    parser.add_argument("--embedding-name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument(
        "--role-graph-edge-policy",
        choices=ROLE_GRAPH_EDGE_POLICIES,
        default=DEFAULT_ROLE_GRAPH_EDGE_POLICY,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def export_pool(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    dataset = str(args.dataset)
    max_queries = int(args.max_queries)
    expander = FrozenETv3Expander(
        dataset=dataset,
        candidate_pool_k=int(args.pool_k),
        reader_budget_k=int(args.reader_budget_k),
        et_candidate_pool_k=int(args.et_candidate_pool_k),
        data_root=Path(args.data_root),
        run_root=Path(args.frozen_runs_root) / dataset,
        llm_name=str(args.llm_name),
        embedding_name=str(args.embedding_name),
        embedding_base_url=str(args.embedding_base_url),
        role_graph_edge_policy=str(args.role_graph_edge_policy),
    )
    limit = len(expander.samples) if max_queries <= 0 else min(max_queries, len(expander.samples))
    expander.prepare_queries(max_queries=limit)
    records: list[dict[str, Any]] = []
    progress_every = max(int(args.progress_every), 0)
    for query_idx in range(limit):
        if progress_every and (query_idx == 0 or query_idx % progress_every == 0):
            print(
                f"[PROGRESS] dataset={dataset} query={query_idx}/{limit} elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
        records.append(expander.expand_query_record(query_idx))
    if progress_every:
        print(
            f"[PROGRESS] dataset={dataset} query={limit}/{limit} elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )

    metadata = expander.metadata()
    return {
        "dataset": dataset,
        "limit": int(limit),
        "pool_k": int(args.pool_k),
        "source": "evidencelink_fresh_pool_export",
        "retrieval_report": "",
        "openie_path": str(metadata.get("openie_path") or ""),
        "retrieval": {
            "input_method": f"evidencelink_{args.role_graph_edge_policy}_fresh_pool",
            "input_source_variant": str(metadata.get("variant") or ""),
            "role_graph_edge_policy": str(args.role_graph_edge_policy),
            "candidate_generator_mode": str(metadata.get("candidate_generator_mode") or ""),
            "runner": str(metadata.get("runner") or ""),
            "enable_variable_flow_traversal": bool(metadata.get("enable_variable_flow_traversal")),
        },
        "fresh_expander": metadata,
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = export_pool(args)
    write_json(payload, Path(args.output_json))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "dataset": payload.get("dataset"),
                "limit": payload.get("limit"),
                "pool_k": payload.get("pool_k"),
                "role_graph_edge_policy": payload.get("retrieval", {}).get("role_graph_edge_policy"),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
