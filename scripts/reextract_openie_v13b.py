#!/usr/bin/env python3
"""Re-extract OpenIE triples for limit100 documents using the v13b evidence-frame prompt.

This script is a lightweight alternative to running the full main_dpr.py pipeline.
It reads the existing OpenIE JSON, identifies documents referenced by the limit100
structural source report, re-extracts triples for those documents using the v13b
prompt template, and writes a new OpenIE JSON with the updated triples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.information_extraction.openie_openai import OpenIE
from src.hipporag.llm.openai_gpt import CacheOpenAI


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def doc_indices_from_source_report(source_report: Mapping[str, Any], *, gold_only: bool = False) -> Set[int]:
    indices: Set[int] = set()
    keys = ("gold_doc_indices",) if gold_only else ("candidate_doc_indices", "dense_pool_doc_indices", "gold_doc_indices")
    for dataset_payload in source_report.get("datasets", []) or []:
        for row in dataset_payload.get("rows", []) or []:
            for key in keys:
                for idx in row.get(key, []) or []:
                    try:
                        indices.add(int(idx))
                    except (TypeError, ValueError):
                        pass
    return indices

def reextract_docs(
    *,
    openie_docs: List[Dict[str, Any]],
    target_indices: Set[int],
    openie: OpenIE,
) -> List[Dict[str, Any]]:
    updated = list(openie_docs)
    total = len(target_indices)
    done = 0
    for doc_index in sorted(target_indices):
        if doc_index < 0 or doc_index >= len(updated):
            continue
        doc = dict(updated[doc_index])
        passage = str(doc.get("passage") or "")
        if not passage.strip():
            continue
        existing_entities = doc.get("extracted_entities", []) or []
        ner_result = openie.ner(chunk_key=str(doc.get("idx", doc_index)), passage=passage)
        entities = ner_result.unique_entities if ner_result.unique_entities else existing_entities
        triple_result = openie.triple_extraction(
            chunk_key=str(doc.get("idx", doc_index)),
            passage=passage,
            named_entities=entities,
        )
        doc["extracted_entities"] = entities
        doc["extracted_triples"] = triple_result.triples
        doc["v13b_reextracted"] = True
        updated[doc_index] = doc
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  reextracted {done}/{total} docs", flush=True)
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-extract OpenIE with v13b evidence-frame prompt.")
    parser.add_argument("--openie-json", required=True, help="Existing OpenIE JSON to update.")
    parser.add_argument("--source-report", required=True, help="Structural source report (identifies target docs).")
    parser.add_argument("--llm-base-url", required=True, help="Qwen vLLM endpoint URL.")
    parser.add_argument("--llm-model", default="qwen3-8b", help="Model name for the LLM.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-json", required=True, help="Output path for updated OpenIE JSON.")
    parser.add_argument("--gold-only", action="store_true", help="Only reextract gold docs (much faster).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    openie_payload = load_json(Path(args.openie_json))
    source_report = load_json(Path(args.source_report))
    openie_docs = openie_payload.get("docs", []) if isinstance(openie_payload, dict) else openie_payload

    target_indices = doc_indices_from_source_report(source_report, gold_only=args.gold_only)
    print(f"Target docs: {len(target_indices)} from source report, corpus size: {len(openie_docs)}")

    cache_dir = str(Path(args.output_json).parent / "llm_cache")

    from src.hipporag.utils.config_utils import BaseConfig
    config = BaseConfig(
        llm_name=args.llm_model,
        llm_base_url=args.llm_base_url,
        max_new_tokens=args.max_new_tokens,
        save_dir=str(Path(args.output_json).parent),
    )
    llm = CacheOpenAI(cache_dir=cache_dir, global_config=config)
    openie = OpenIE(llm_model=llm, triple_extraction_template="triple_extraction_v13b")

    updated_docs = reextract_docs(
        openie_docs=openie_docs,
        target_indices=target_indices,
        openie=openie,
    )

    output_payload = dict(openie_payload) if isinstance(openie_payload, dict) else {}
    output_payload["docs"] = updated_docs
    output_payload["v13b_reextracted_count"] = sum(1 for d in updated_docs if d.get("v13b_reextracted"))
    write_json(Path(args.output_json), output_payload)
    print(f"Wrote {args.output_json} ({output_payload['v13b_reextracted_count']} docs reextracted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
