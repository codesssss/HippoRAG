#!/usr/bin/env python3
"""Build a minimal fixed-pool JSONL cache for D-PathRAG smoke runs."""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.cache import build_cache_jsonl
from src.dpathrag.data import load_2wiki_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--data_root", default="reproduce/dataset")
    parser.add_argument("--pool_json", default="run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json")
    parser.add_argument("--source", default="dense")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    args = parser.parse_args()

    bundle = load_2wiki_bundle(args.data_root, args.dataset)
    rows = build_cache_jsonl(bundle.samples, args.pool_json, args.output_jsonl, source=args.source, limit=int(args.limit))
    print(f"Wrote {rows} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()

