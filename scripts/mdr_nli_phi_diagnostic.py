"""MDR NLI phi diagnostic: compare NLI vs cosine phi behavior on demands × pool.

Reports:
1. NLI unmet rate vs cosine unmet rate
2. Whether NLI-unmet queries correlate with baseline EM failures
3. Whether NLI-selected repair docs contain gold support
4. Qualitative edit inspection (are repairs adding missing evidence?)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtc_embed_utils import DTCRequirement

logger = logging.getLogger(__name__)


def load_pool_records(pool_json: str) -> List[Dict[str, Any]]:
    with open(pool_json) as f:
        data = json.load(f)
    return data["records"]


def load_cached_traces(trace_json: str) -> List[Dict[str, Any]]:
    with open(trace_json) as f:
        data = json.load(f)
    return data.get("setwise_selector_query_traces", [])


def extract_requirements(selector_trace: Dict[str, Any]) -> List[DTCRequirement]:
    raw = selector_trace.get("requirements", [])
    reqs = []
    for r in raw:
        reqs.append(DTCRequirement(
            unit_id=str(r.get("unit_id", "")),
            subquery=str(r.get("subquery", "")),
            depends_on=tuple(r.get("depends_on", ())),
            expected_answer_type=str(r.get("expected_answer_type", "unknown")),
            anchor_mentions=tuple(r.get("anchor_mentions", ())),
            role=str(r.get("role", "support")),
            satisfiable_by=str(r.get("satisfiable_by", "unknown")),
        ))
    return reqs
# PLACEHOLDER_CONTINUE


def build_nli_verifier(model_name: str = "cross-encoder/nli-deberta-v3-base", batch_size: int = 32, device: str = "cpu"):
    from transformers import pipeline as hf_pipeline
    from src.dpathrag.arec.verifier import NliVerifier
    verifier = NliVerifier.__new__(NliVerifier)
    verifier.name = model_name
    verifier.batch_size = int(batch_size)
    verifier._pipeline = hf_pipeline(
        "text-classification", model=model_name, tokenizer=model_name,
        top_k=None, device=device, truncation=True, max_length=512,
    )
    return verifier


def compute_nli_phi(
    requirements: List[DTCRequirement],
    pool_docs: List[str],
    verifier: Any,
) -> np.ndarray:
    pairs: List[Tuple[str, str]] = []
    for req in requirements:
        for doc in pool_docs:
            pairs.append((str(req.subquery), str(doc)))
    if not pairs:
        return np.zeros((len(requirements), len(pool_docs)), dtype=float)
    raw = verifier.score_many(pairs)
    phi = np.zeros((len(requirements), len(pool_docs)), dtype=float)
    idx = 0
    for r_idx in range(len(requirements)):
        for d_idx in range(len(pool_docs)):
            phi[r_idx, d_idx] = float(raw[idx])
            idx += 1
    return phi


def title_from_doc(doc_text: str) -> str:
    return doc_text.split("\n", 1)[0].strip()


def run_diagnostic(
    pool_records: List[Dict[str, Any]],
    cached_traces: List[Dict[str, Any]],
    verifier: Any,
    limit: int = 20,
    qa_top_k: int = 5,
    tau_percentile: float = 50.0,
    tau_absolute: float | None = None,
) -> Dict[str, Any]:
    limit = min(limit, len(pool_records), len(cached_traces))
    rows: List[Dict[str, Any]] = []

    for q_idx in range(limit):
        rec = pool_records[q_idx]
        trace = cached_traces[q_idx]
        question = str(rec.get("question", ""))
        gold_titles = set(str(t) for t in rec.get("gold_titles", []))
        pool_docs = list(rec.get("pool_docs", []))
        pool_titles = list(rec.get("pool_titles", []))
        pool_limit = len(pool_docs)
        if pool_limit == 0:
            continue

        st = trace.get("selector_trace", {})
        requirements = extract_requirements(st)
        active_reqs = [r for r in requirements if str(r.subquery or "").strip()]
        if not active_reqs:
            continue

        baseline_em = float(trace.get("baseline_metrics", {}).get("ExactMatch", 0.0))

        cosine_tau = float(st.get("tau", 0.0))
        cosine_unmet = set(str(uid) for uid in st.get("unmet_requirement_ids", []))
        cosine_baseline_max = st.get("baseline_max_by_requirement", {})

        nli_phi = compute_nli_phi(active_reqs, pool_docs, verifier)
        finite = nli_phi[np.isfinite(nli_phi)]
        if tau_absolute is not None:
            nli_tau = float(tau_absolute)
        else:
            nli_tau = float(np.percentile(finite, tau_percentile)) if finite.size else 1.0
        nli_tau = float(np.clip(nli_tau, 0.0, 1.0))

        baseline_positions = list(range(min(qa_top_k, pool_limit)))
        nli_baseline_max = {}
        nli_unmet = set()
        for r_idx, req in enumerate(active_reqs):
            bmax = float(np.max(nli_phi[r_idx, baseline_positions])) if baseline_positions else 0.0
            nli_baseline_max[req.unit_id] = bmax
            if bmax < nli_tau:
                nli_unmet.add(req.unit_id)

        nli_repair_candidates = {}
        for uid in nli_unmet:
            r_idx = next((i for i, r in enumerate(active_reqs) if r.unit_id == uid), None)
            if r_idx is None:
                continue
            selected_set = set(baseline_positions)
            cands = [(pos, float(nli_phi[r_idx, pos])) for pos in range(pool_limit) if pos not in selected_set]
            cands.sort(key=lambda x: -x[1])
            top3 = cands[:3]
            nli_repair_candidates[uid] = [
                {
                    "pos": pos,
                    "title": pool_titles[pos] if pos < len(pool_titles) else "?",
                    "nli_score": round(score, 4),
                    "is_gold": pool_titles[pos] in gold_titles if pos < len(pool_titles) else False,
                }
                for pos, score in top3
            ]

        gold_in_baseline = sum(1 for t in pool_titles[:qa_top_k] if t in gold_titles)
        gold_in_pool = sum(1 for t in pool_titles if t in gold_titles)
# PLACEHOLDER_ROW_APPEND

        repair_has_gold = any(
            any(c["is_gold"] for c in cands)
            for cands in nli_repair_candidates.values()
        )

        rows.append({
            "q_idx": q_idx,
            "question": question,
            "baseline_em": baseline_em,
            "requirement_count": len(active_reqs),
            "cosine_tau": round(cosine_tau, 6),
            "cosine_unmet_count": len(cosine_unmet),
            "cosine_unmet_ids": sorted(cosine_unmet),
            "nli_tau": round(nli_tau, 6),
            "nli_unmet_count": len(nli_unmet),
            "nli_unmet_ids": sorted(nli_unmet),
            "nli_phi_mean": round(float(np.mean(nli_phi)), 6),
            "nli_phi_std": round(float(np.std(nli_phi)), 6),
            "nli_phi_max": round(float(np.max(nli_phi)), 6),
            "nli_phi_median": round(float(np.median(nli_phi)), 6),
            "nli_baseline_max_by_req": {k: round(v, 4) for k, v in nli_baseline_max.items()},
            "cosine_baseline_max_by_req": {k: round(float(v), 4) for k, v in cosine_baseline_max.items()},
            "nli_repair_candidates": nli_repair_candidates,
            "repair_has_gold": repair_has_gold,
            "gold_in_baseline": gold_in_baseline,
            "gold_in_pool": gold_in_pool,
            "gold_titles": sorted(gold_titles),
        })
        logger.info(
            "q=%d nli_unmet=%d cosine_unmet=%d baseline_em=%.0f repair_has_gold=%s",
            q_idx, len(nli_unmet), len(cosine_unmet), baseline_em, repair_has_gold,
        )

    cosine_unmet_rates = [r["cosine_unmet_count"] / max(r["requirement_count"], 1) for r in rows]
    nli_unmet_rates = [r["nli_unmet_count"] / max(r["requirement_count"], 1) for r in rows]
    nli_unmet_queries = [r for r in rows if r["nli_unmet_count"] > 0]
    cosine_unmet_queries = [r for r in rows if r["cosine_unmet_count"] > 0]

    nli_unmet_em = np.mean([r["baseline_em"] for r in nli_unmet_queries]) if nli_unmet_queries else float("nan")
    nli_met_queries = [r for r in rows if r["nli_unmet_count"] == 0]
    nli_met_em = np.mean([r["baseline_em"] for r in nli_met_queries]) if nli_met_queries else float("nan")

    repair_gold_count = sum(1 for r in nli_unmet_queries if r["repair_has_gold"])

    summary = {
        "query_count": len(rows),
        "metric_1_unmet_rate": {
            "cosine_avg_unmet_rate": round(float(np.mean(cosine_unmet_rates)), 4),
            "nli_avg_unmet_rate": round(float(np.mean(nli_unmet_rates)), 4),
            "cosine_unmet_query_count": len(cosine_unmet_queries),
            "nli_unmet_query_count": len(nli_unmet_queries),
        },
        "metric_2_unmet_vs_baseline_em": {
            "nli_unmet_avg_baseline_em": round(float(nli_unmet_em), 4) if not np.isnan(nli_unmet_em) else None,
            "nli_met_avg_baseline_em": round(float(nli_met_em), 4) if not np.isnan(nli_met_em) else None,
            "gap": round(float(nli_met_em - nli_unmet_em), 4) if not (np.isnan(nli_met_em) or np.isnan(nli_unmet_em)) else None,
        },
        "metric_3_repair_gold_hit": {
            "nli_unmet_query_count": len(nli_unmet_queries),
            "repair_candidates_contain_gold": repair_gold_count,
            "repair_gold_rate": round(repair_gold_count / max(len(nli_unmet_queries), 1), 4),
        },
        "metric_4_phi_distribution": {
            "nli_phi_global_mean": round(float(np.mean([r["nli_phi_mean"] for r in rows])), 6),
            "nli_phi_global_median": round(float(np.mean([r["nli_phi_median"] for r in rows])), 6),
            "nli_tau_mean": round(float(np.mean([r["nli_tau"] for r in rows])), 6),
            "cosine_tau_mean": round(float(np.mean([r["cosine_tau"] for r in rows])), 6),
        },
    }
    return {"summary": summary, "rows": rows}
# PLACEHOLDER_MAIN


def main():
    parser = argparse.ArgumentParser(description="MDR NLI phi diagnostic")
    parser.add_argument("--dataset", required=True, choices=["2wikimultihopqa", "hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--tau_percentile", type=float, default=50.0)
    parser.add_argument("--tau_absolute", type=float, default=None, help="Absolute NLI tau threshold (overrides percentile)")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--nli_batch_size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool_json", default=None)
    parser.add_argument("--trace_json", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(__file__).resolve().parent.parent
    if args.pool_json is None:
        args.pool_json = str(root / f"run_logs/proprag_pool_exports_full1000_20260424/{args.dataset}_pool100.json")
    if args.trace_json is None:
        args.trace_json = str(root / f"run_logs/minimal_demand_repair_limit100_aligned_20260501/{args.dataset}_minimal_demand_repair_limit100_aligned.json")
    if args.output_json is None:
        out_dir = root / "run_logs" / "mdr_nli_phi_diagnostic_20260501"
        out_dir.mkdir(parents=True, exist_ok=True)
        tau_tag = f"_tau{args.tau_absolute}" if args.tau_absolute is not None else ""
        args.output_json = str(out_dir / f"{args.dataset}_nli_phi_diagnostic_limit{args.limit}{tau_tag}.json")

    logger.info("Loading pool from %s", args.pool_json)
    pool_records = load_pool_records(args.pool_json)
    logger.info("Loading cached traces from %s", args.trace_json)
    cached_traces = load_cached_traces(args.trace_json)
    logger.info("Pool records: %d, Cached traces: %d", len(pool_records), len(cached_traces))

    logger.info("Building NLI verifier: %s", args.nli_model)
    verifier = build_nli_verifier(model_name=args.nli_model, batch_size=args.nli_batch_size, device=args.device)

    result = run_diagnostic(
        pool_records=pool_records,
        cached_traces=cached_traces,
        verifier=verifier,
        limit=args.limit,
        qa_top_k=args.qa_top_k,
        tau_percentile=args.tau_percentile,
        tau_absolute=args.tau_absolute,
    )
    result["config"] = {
        "dataset": args.dataset,
        "limit": args.limit,
        "qa_top_k": args.qa_top_k,
        "tau_percentile": args.tau_percentile,
        "tau_absolute": args.tau_absolute,
        "nli_model": args.nli_model,
        "pool_json": args.pool_json,
        "trace_json": args.trace_json,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Output written to %s", args.output_json)

    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()