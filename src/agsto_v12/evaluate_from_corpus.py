#!/usr/bin/env python3
"""Run AG-STO v12 from a raw corpus.

Pipeline:

    raw corpus -> OpenIE JSON -> STO graph -> native STO proposals -> AG-STO v12 retrieval

OpenIE extraction is an explicit preprocessing step. Use ``--openie-mode llm``
for real OpenIE extraction via an OpenAI-compatible chat endpoint, or
``--openie-mode heuristic`` for dependency-free smoke tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .build_openie import build_openie_from_corpus
from .evaluate_from_openie import evaluate_agsto_v12_from_openie, write_markdown


def default_openie_path(output_json: str) -> Path:
    output_path = Path(output_json).resolve()
    return output_path.parent / f"{output_path.stem}_openie.json"


def evaluate_agsto_v12_from_corpus(args: argparse.Namespace) -> dict:
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    openie_json = Path(args.openie_json).resolve() if args.openie_json else default_openie_path(str(output_json))
    openie_built = False

    if bool(args.force_openie) or not openie_json.exists():
        build_args = argparse.Namespace(
            corpus_json=str(Path(args.corpus_json).resolve()),
            output_openie_json=str(openie_json),
            mode=str(args.openie_mode),
            limit_docs=int(args.limit_docs),
            workers=int(args.openie_workers),
            continue_on_error=bool(args.continue_on_openie_error),
            heuristic_max_entities=int(args.heuristic_max_entities),
            heuristic_max_triples=int(args.heuristic_max_triples),
            llm_base_url=str(args.llm_base_url),
            llm_model=str(args.llm_model),
            api_key=getattr(args, "api_key", None),
            api_key_file=getattr(args, "api_key_file", None),
            allow_empty_api_key=bool(args.allow_empty_api_key),
            timeout=float(args.openie_timeout),
            max_tokens=int(args.openie_max_tokens),
            retries=int(args.openie_retries),
        )
        build_openie_from_corpus(build_args)
        openie_built = True

    eval_args = argparse.Namespace(
        openie_results=str(openie_json),
        queries_json=str(Path(args.queries_json).resolve()),
        dataset=args.dataset,
        policy=str(args.policy),
        max_queries=int(args.max_queries),
        retrieval_top_k=int(args.retrieval_top_k),
        evidence_set_size=int(args.evidence_set_size),
        stable_anchor_k=int(args.stable_anchor_k),
        proposal_candidate_depth=int(args.proposal_candidate_depth),
        support_proposal_depth=int(args.support_proposal_depth),
        candidate_limit=int(args.candidate_limit),
        beam_size=int(args.beam_size),
        set_search_policy=str(args.set_search_policy),
        max_endpoint_degree=int(args.max_endpoint_degree),
        semantic_residual_weight=float(args.semantic_residual_weight),
        support_realization=str(args.support_realization),
        support_mct_iterations=int(args.support_mct_iterations),
        support_mct_depth=int(args.support_mct_depth),
        support_mct_exploration=float(args.support_mct_exploration),
        output_json=str(output_json),
        output_md=str(output_md),
    )
    summary = evaluate_agsto_v12_from_openie(eval_args)
    summary["raw_corpus_path"] = str(Path(args.corpus_json).resolve())
    summary["openie_json_path"] = str(openie_json)
    summary["openie_built"] = bool(openie_built)
    summary["openie_mode"] = str(args.openie_mode)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AG-STO v12 from a raw corpus.")
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--queries-json", required=True)
    parser.add_argument("--openie-json", default=None)
    parser.add_argument("--force-openie", action="store_true")
    parser.add_argument("--openie-mode", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--openie-workers", type=int, default=4)
    parser.add_argument("--continue-on-openie-error", action="store_true")
    parser.add_argument("--heuristic-max-entities", type=int, default=24)
    parser.add_argument("--heuristic-max-triples", type=int, default=32)
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--allow-empty-api-key", action="store_true")
    parser.add_argument("--openie-timeout", type=float, default=120.0)
    parser.add_argument("--openie-max-tokens", type=int, default=1024)
    parser.add_argument("--openie-retries", type=int, default=3)

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--policy", default="graph", choices=["base", "graph"])
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--evidence-set-size", type=int, default=5)
    parser.add_argument("--stable-anchor-k", type=int, default=2)
    parser.add_argument("--proposal-candidate-depth", type=int, default=10)
    parser.add_argument("--support-proposal-depth", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--beam-size", type=int, default=12)
    parser.add_argument("--set-search-policy", default="beam", choices=["beam", "mct"])
    parser.add_argument("--max-endpoint-degree", type=int, default=30)
    parser.add_argument("--semantic-residual-weight", type=float, default=8.0)
    parser.add_argument("--support-realization", default="direct", choices=["direct", "mct"])
    parser.add_argument("--support-mct-iterations", type=int, default=64)
    parser.add_argument("--support-mct-depth", type=int, default=2)
    parser.add_argument("--support-mct-exploration", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    summary = evaluate_agsto_v12_from_corpus(args)
    compact = {
        "openie_built": bool(summary.get("openie_built", False)),
        "openie_mode": summary.get("openie_mode"),
        "datasets": {
            dataset["dataset"]: {
                key: value
                for key, value in dataset.get("metrics", {}).items()
                if key.startswith("agsto_v12_") or key == "gold_row_count"
            }
            for dataset in summary.get("datasets", []) or []
        },
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
