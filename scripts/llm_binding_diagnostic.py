"""LLM-extraction binding diagnostic: compare LLM vs hand-crafted binding gold hit rate.

For each dependent requirement in DAEC traces:
1. Take upstream top-k docs (cosine baseline)
2. Ask Qwen3-8B to extract candidate answer entities from each doc
3. Check if extracted entities match any pool doc title
4. Compare gold hit rate against hand-crafted binding candidates
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def load_pool_records(pool_json: str) -> List[Dict[str, Any]]:
    with open(pool_json) as f:
        return json.load(f)["records"]


def load_daec_traces(trace_json: str) -> List[Dict[str, Any]]:
    with open(trace_json) as f:
        return json.load(f).get("setwise_selector_query_traces", [])


def llm_extract_entities(
    subquery: str,
    doc_text: str,
    llm_url: str,
    model_name: str,
    max_tokens: int = 80,
    temperature: float = 0.0,
) -> List[str]:
    prompt = (
        "/no_think\n"
        f"Passage:\n\"{doc_text[:1500]}\"\n\n"
        f"Question: {subquery}\n\n"
        "Extract all entity names from the passage that could answer this question. "
        "Return one entity per line, nothing else. If no entity answers the question, reply NONE."
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(f"{llm_url}/chat/completions", json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = strip_think(raw)
        entities = []
        for line in raw.strip().split("\n"):
            line = line.strip().strip("-•").strip()
            if line and line.upper() != "NONE":
                entities.append(line)
        return entities
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
        return []
def title_match_score(entity: str, pool_titles: List[str]) -> Optional[int]:
    entity_lower = entity.lower().strip()
    for idx, title in enumerate(pool_titles):
        if entity_lower == title.lower().strip():
            return idx
    for idx, title in enumerate(pool_titles):
        if entity_lower in title.lower() or title.lower() in entity_lower:
            if min(len(entity_lower), len(title.lower())) >= 3:
                return idx
    return None


def run_diagnostic(
    pool_records: List[Dict[str, Any]],
    daec_traces: List[Dict[str, Any]],
    llm_url: str,
    model_name: str,
    limit: int = 20,
    qa_top_k: int = 5,
) -> Dict[str, Any]:
    limit = min(limit, len(pool_records), len(daec_traces))
    rows: List[Dict[str, Any]] = []

    for q_idx in range(limit):
        rec = pool_records[q_idx]
        trace = daec_traces[q_idx]
        question = str(rec.get("question", ""))
        gold_titles = set(str(t) for t in rec.get("gold_titles", []))
        pool_docs = list(rec.get("pool_docs", []))
        pool_titles = list(rec.get("pool_titles", []))

        st = trace.get("selector_trace", {})
        reqs = st.get("requirements", [])
        dep_reqs = [r for r in reqs if r.get("depends_on")]
        if not dep_reqs:
            continue

        hc_bcr = st.get("binding_candidates_by_requirement", {})

        llm_binding_results: Dict[str, Any] = {}
        for req in dep_reqs:
            uid = req["unit_id"]
            subquery = req["subquery"]
            upstream_ids = req.get("depends_on", [])
            upstream_reqs = [r for r in reqs if r["unit_id"] in upstream_ids]

            upstream_docs_indices = list(range(min(qa_top_k, len(pool_docs))))
            extracted_entities: List[Dict[str, Any]] = []
            for doc_idx in upstream_docs_indices:
                doc_text = pool_docs[doc_idx]
                entities = llm_extract_entities(
                    subquery, doc_text, llm_url, model_name,
                )
                for ent in entities:
                    match_pos = title_match_score(ent, pool_titles)
                    extracted_entities.append({
                        "entity": ent,
                        "source_doc_idx": doc_idx,
                        "source_title": pool_titles[doc_idx] if doc_idx < len(pool_titles) else "?",
                        "match_pool_pos": match_pos,
                        "match_title": pool_titles[match_pos] if match_pos is not None else None,
                        "is_gold": pool_titles[match_pos] in gold_titles if match_pos is not None else False,
                    })

            llm_binding_results[uid] = extracted_entities

        hc_cands = []
        hc_gold = 0
        for uid, cands in hc_bcr.items():
            for c in cands:
                hc_cands.append(c)
                if c.get("title", "") in gold_titles:
                    hc_gold += 1

        llm_cands_all = []
        llm_gold = 0
        llm_unique_entities: Dict[str, Dict[str, Any]] = {}
        for uid, ents in llm_binding_results.items():
            for e in ents:
                key = e["entity"].lower()
                if key not in llm_unique_entities:
                    llm_unique_entities[key] = e
                    llm_cands_all.append(e)
                    if e["is_gold"]:
                        llm_gold += 1

        rows.append({
            "q_idx": q_idx,
            "question": question,
            "gold_titles": sorted(gold_titles),
            "dep_req_count": len(dep_reqs),
            "hc_binding_count": len(hc_cands),
            "hc_gold_hits": hc_gold,
            "llm_binding_count": len(llm_cands_all),
            "llm_gold_hits": llm_gold,
            "llm_binding_details": llm_binding_results,
            "hc_binding_details": hc_bcr,
        })
        logger.info(
            "q=%d hc_gold=%d/%d llm_gold=%d/%d  %s",
            q_idx, hc_gold, len(hc_cands), llm_gold, len(llm_cands_all),
            question[:60],
        )

    hc_total = sum(r["hc_binding_count"] for r in rows)
    hc_gold_total = sum(r["hc_gold_hits"] for r in rows)
    llm_total = sum(r["llm_binding_count"] for r in rows)
    llm_gold_total = sum(r["llm_gold_hits"] for r in rows)
    hc_query_hit = sum(1 for r in rows if r["hc_gold_hits"] > 0)
    llm_query_hit = sum(1 for r in rows if r["llm_gold_hits"] > 0)

    summary = {
        "query_count": len(rows),
        "handcrafted": {
            "total_candidates": hc_total,
            "gold_hits": hc_gold_total,
            "candidate_gold_rate": round(hc_gold_total / max(hc_total, 1), 4),
            "query_gold_hit": hc_query_hit,
            "query_gold_rate": round(hc_query_hit / max(len(rows), 1), 4),
        },
        "llm_extraction": {
            "total_candidates": llm_total,
            "gold_hits": llm_gold_total,
            "candidate_gold_rate": round(llm_gold_total / max(llm_total, 1), 4),
            "query_gold_hit": llm_query_hit,
            "query_gold_rate": round(llm_query_hit / max(len(rows), 1), 4),
        },
    }
    return {"summary": summary, "rows": rows}
# PLACEHOLDER_MAIN


def main():
    parser = argparse.ArgumentParser(description="LLM-extraction binding diagnostic")
    parser.add_argument("--dataset", required=True, choices=["2wikimultihopqa", "hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--llm_url", default="http://localhost:8043/v1")
    parser.add_argument("--model_name", default="qwen3-8b-train")
    parser.add_argument("--pool_json", default=None)
    parser.add_argument("--daec_trace_json", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(__file__).resolve().parent.parent
    if args.pool_json is None:
        args.pool_json = str(root / f"run_logs/proprag_pool_exports_full1000_20260424/{args.dataset}_pool100.json")
    if args.daec_trace_json is None:
        ds_key = "2wiki" if args.dataset == "2wikimultihopqa" else args.dataset
        candidates = [
            root / f"run_logs/daec_noisyor_proprag_pool100_{ds_key}_fixclean_20260428.json",
            root / f"run_logs/daec_noisyor_dense_pool100_{args.dataset}_fixclean_20260428.json",
        ]
        for c in candidates:
            if c.exists():
                args.daec_trace_json = str(c)
                break
        if args.daec_trace_json is None:
            logger.error("No DAEC trace found for %s", args.dataset)
            sys.exit(1)
    if args.output_json is None:
        out_dir = root / "run_logs" / "llm_binding_diagnostic_20260502"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = str(out_dir / f"{args.dataset}_llm_binding_limit{args.limit}.json")

    logger.info("Loading pool from %s", args.pool_json)
    pool_records = load_pool_records(args.pool_json)
    logger.info("Loading DAEC traces from %s", args.daec_trace_json)
    daec_traces = load_daec_traces(args.daec_trace_json)
    logger.info("Pool records: %d, DAEC traces: %d", len(pool_records), len(daec_traces))

    result = run_diagnostic(
        pool_records=pool_records,
        daec_traces=daec_traces,
        llm_url=args.llm_url,
        model_name=args.model_name,
        limit=args.limit,
        qa_top_k=args.qa_top_k,
    )
    result["config"] = {
        "dataset": args.dataset,
        "limit": args.limit,
        "qa_top_k": args.qa_top_k,
        "llm_url": args.llm_url,
        "model_name": args.model_name,
        "pool_json": args.pool_json,
        "daec_trace_json": args.daec_trace_json,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Output written to %s", args.output_json)

    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
