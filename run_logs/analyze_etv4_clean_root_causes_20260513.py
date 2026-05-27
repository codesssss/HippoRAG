#!/usr/bin/env python3
"""Root-cause decomposition for ETv4 clean vs graph baselines.

This script is diagnostic-only. It uses gold labels to classify failures, not
to drive retrieval.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path("/mnt/nvme/code/HippoRAG")
METHOD = "evidence_transition_graphragv4_fact_witnessed_sto"
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")

ETV4_RETRIEVAL_ROOT = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
ETV4_QA_ROOT = ROOT / "run_logs/etv4_clean_mainline_qwen32b_reader_gpt4omini_20260512/reports"
PROP_ROOT = ROOT / "run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512"
HIPPO_ROOT = ROOT / "run_logs/hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2"
OUT_DIR = ROOT / "run_logs/etv4_clean_root_cause_20260513"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ints(values: Iterable[Any], *, limit: int | None = None) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values or []:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def title_from_passage(passage: str) -> str:
    return str(passage or "").splitlines()[0].strip()


def norm_title(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def doc_titles(dataset: str) -> dict[int, str]:
    path = ETV4_RETRIEVAL_ROOT / dataset / "index/openie_results_ner_qwen3-32b-judge.json"
    payload = read_json(path)
    titles: dict[int, str] = {}
    for idx, doc in enumerate(payload.get("docs", []) or []):
        doc_idx = int(doc.get("doc_index", doc.get("doc_idx", idx)) if isinstance(doc, Mapping) else idx)
        titles[doc_idx] = title_from_passage(str(doc.get("passage", "")))
    return titles


def exact_recall(top: Sequence[int], gold: Sequence[int]) -> float:
    gold_set = set(ints(gold))
    if not gold_set:
        return 0.0
    return len(set(ints(top)) & gold_set) / len(gold_set)


def exact_all(top: Sequence[int], gold: Sequence[int]) -> bool:
    gold_set = set(ints(gold))
    return bool(gold_set) and gold_set <= set(ints(top))


def title_recall(top_titles: Sequence[str], gold_titles: Sequence[str]) -> float:
    gold_set = {norm_title(t) for t in gold_titles if norm_title(t)}
    if not gold_set:
        return 0.0
    top_set = {norm_title(t) for t in top_titles if norm_title(t)}
    return len(top_set & gold_set) / len(gold_set)


def title_all(top_titles: Sequence[str], gold_titles: Sequence[str]) -> bool:
    gold_set = {norm_title(t) for t in gold_titles if norm_title(t)}
    top_set = {norm_title(t) for t in top_titles if norm_title(t)}
    return bool(gold_set) and gold_set <= top_set


def load_qa(path: Path) -> dict[int, Mapping[str, Any]]:
    payload = read_json(path)
    method_payload = next(iter(payload["datasets"][0]["methods"].values()))
    return {
        int(row.get("query_index", idx)): row
        for idx, row in enumerate(method_payload.get("per_query", []) or [])
    }


def qa_metrics(path: Path) -> Mapping[str, Any]:
    payload = read_json(path)
    return next(iter(payload["datasets"][0]["methods"].values())).get("metrics", {}) or {}


def etv4_rows(dataset: str, titles: Mapping[int, str]) -> dict[int, dict[str, Any]]:
    path = ETV4_RETRIEVAL_ROOT / dataset / "reports" / f"{dataset}_{METHOD}_retrieval.json"
    qa_path = ETV4_QA_ROOT / f"{dataset}_etv4_qwen32b_gpt4omini_reader_qa.json"
    retrieval = read_json(path)
    qa = load_qa(qa_path)
    rows: dict[int, dict[str, Any]] = {}
    for fallback_idx, row in enumerate(retrieval.get("rows", []) or []):
        qid = int(row.get("query_index", fallback_idx))
        top5 = ints(row.get("retrieved_doc_indices_top5", []), limit=5)
        gold = ints(row.get("gold_doc_indices", []))
        candidate_universe = (row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}
        pool200 = ints(candidate_universe.get("candidate_doc_indices", []), limit=200)
        gold_titles = [titles.get(doc, "") for doc in gold]
        top5_titles = [titles.get(doc, "") for doc in top5]
        pool_titles = [titles.get(doc, "") for doc in pool200]
        qa_row = qa.get(qid, {})
        rows[qid] = {
            "qid": qid,
            "question": str(row.get("question", "")),
            "top5": top5,
            "pool200": pool200,
            "gold": gold,
            "top5_titles": top5_titles,
            "pool_titles": pool_titles,
            "gold_titles": gold_titles,
            "em": float(qa_row.get("ExactMatch", 0.0) or 0.0),
            "f1": float(qa_row.get("F1", 0.0) or 0.0),
            "predicted_answer": str(qa_row.get("predicted_answer", "")),
            "gold_answers": qa_row.get("gold_answers", []),
        }
    return rows


def prop_rows(dataset: str) -> dict[int, dict[str, Any]]:
    pool_path = PROP_ROOT / "pools/proprag" / f"{dataset}_proprag_qwen32b_nothink_pool200.json"
    qa_path = PROP_ROOT / "reader_qa/proprag" / f"{dataset}_proprag_qwen32b_nothink_top200_gpt4omini_reader_top5.json"
    pool = read_json(pool_path)
    qa = load_qa(qa_path)
    rows: dict[int, dict[str, Any]] = {}
    for fallback_idx, record in enumerate(pool.get("records", []) or []):
        qid = int(record.get("query_idx", fallback_idx))
        top5 = ints(record.get("pool_doc_ids", []), limit=5)
        pool200 = ints(record.get("pool_doc_ids", []), limit=200)
        top5_titles = list(record.get("pool_titles", []) or [])[:5]
        pool_titles = list(record.get("pool_titles", []) or [])[:200]
        gold_titles = list(record.get("gold_titles", []) or [])
        gold = ints(qa.get(qid, {}).get("gold_doc_indices", []))
        qa_row = qa.get(qid, {})
        rows[qid] = {
            "qid": qid,
            "question": str(record.get("question", "")),
            "top5": top5,
            "pool200": pool200,
            "gold": gold,
            "top5_titles": top5_titles,
            "pool_titles": pool_titles,
            "gold_titles": gold_titles,
            "em": float(qa_row.get("ExactMatch", 0.0) or 0.0),
            "f1": float(qa_row.get("F1", 0.0) or 0.0),
            "predicted_answer": str(qa_row.get("predicted_answer", "")),
            "gold_answers": qa_row.get("gold_answers", []),
        }
    return rows


def hippo_rows(dataset: str) -> dict[int, dict[str, Any]]:
    pool_path = HIPPO_ROOT / "pools/hipporag" / f"{dataset}_hipporag_qwen32b_valid_pool200.json"
    qa_path = HIPPO_ROOT / "reader_qa/hipporag" / f"{dataset}_hipporag_qwen32b_valid_top200_gpt4omini_reader_top5.json"
    if not pool_path.exists() or not qa_path.exists():
        return {}
    pool = read_json(pool_path)
    qa = load_qa(qa_path)
    records = pool.get("records", []) if isinstance(pool, Mapping) else []
    rows: dict[int, dict[str, Any]] = {}
    for fallback_idx, record in enumerate(records):
        qid = int(record.get("query_idx", fallback_idx))
        qa_row = qa.get(qid, {})
        rows[qid] = {
            "qid": qid,
            "question": str(record.get("question", "")),
            "top5": ints(record.get("pool_doc_ids", []), limit=5),
            "pool200": ints(record.get("pool_doc_ids", []), limit=200),
            "gold": ints(qa_row.get("gold_doc_indices", [])),
            "top5_titles": list(record.get("pool_titles", []) or [])[:5],
            "pool_titles": list(record.get("pool_titles", []) or [])[:200],
            "gold_titles": list(record.get("gold_titles", []) or []),
            "em": float(qa_row.get("ExactMatch", 0.0) or 0.0),
            "f1": float(qa_row.get("F1", 0.0) or 0.0),
            "predicted_answer": str(qa_row.get("predicted_answer", "")),
            "gold_answers": qa_row.get("gold_answers", []),
        }
    return rows


def enrich(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["exact_r5"] = exact_recall(out["top5"], out["gold"])
    out["exact_all5"] = exact_all(out["top5"], out["gold"])
    out["exact_pool_all200"] = exact_all(out["pool200"], out["gold"])
    out["title_r5"] = title_recall(out["top5_titles"], out["gold_titles"])
    out["title_all5"] = title_all(out["top5_titles"], out["gold_titles"])
    out["title_pool_all200"] = title_all(out["pool_titles"], out["gold_titles"])
    return out


def mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(mean(float(row.get(key, 0.0) or 0.0) for row in rows)) if rows else 0.0


def classify_loss(etv4: Mapping[str, Any], other: Mapping[str, Any]) -> str:
    if not etv4["title_pool_all200"]:
        return "ETV4_pool200_missing_title_gold"
    if not etv4["title_all5"]:
        return "ETV4_readout_misses_title_gold"
    if float(etv4["em"]) < float(other["em"]):
        return "ETV4_reader_wrong_with_title_all_gold"
    return "other"


def dataset_compare(dataset: str, baseline_name: str, baseline_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    titles = doc_titles(dataset)
    etv4 = {qid: enrich(row) for qid, row in etv4_rows(dataset, titles).items()}
    baseline = {qid: enrich(row) for qid, row in baseline_rows.items()}
    qids = sorted(set(etv4) & set(baseline))
    paired = [(etv4[qid], baseline[qid]) for qid in qids]

    prop_wins = [(a, b) for a, b in paired if float(b["em"]) > float(a["em"])]
    etv4_wins = [(a, b) for a, b in paired if float(a["em"]) > float(b["em"])]
    ties = [(a, b) for a, b in paired if float(a["em"]) == float(b["em"])]

    loss_buckets = Counter(classify_loss(a, b) for a, b in prop_wins)
    readout_opportunities = [
        (a, b)
        for a, b in prop_wins
        if a["title_pool_all200"] and not a["title_all5"]
    ]
    reader_opportunities = [
        (a, b)
        for a, b in prop_wins
        if a["title_all5"] and float(a["em"]) < float(b["em"])
    ]
    pool_failures = [
        (a, b)
        for a, b in prop_wins
        if not a["title_pool_all200"]
    ]

    summary = {
        "dataset": dataset,
        "baseline": baseline_name,
        "count": len(qids),
        "metrics": {
            "ETV4_title_r5": mean_metric([a for a, _ in paired], "title_r5"),
            f"{baseline_name}_title_r5": mean_metric([b for _, b in paired], "title_r5"),
            "ETV4_title_all5": mean_metric([a for a, _ in paired], "title_all5"),
            f"{baseline_name}_title_all5": mean_metric([b for _, b in paired], "title_all5"),
            "ETV4_title_pool_all200": mean_metric([a for a, _ in paired], "title_pool_all200"),
            f"{baseline_name}_title_pool_all200": mean_metric([b for _, b in paired], "title_pool_all200"),
            "ETV4_EM": mean_metric([a for a, _ in paired], "em"),
            f"{baseline_name}_EM": mean_metric([b for _, b in paired], "em"),
            "ETV4_F1": mean_metric([a for a, _ in paired], "f1"),
            f"{baseline_name}_F1": mean_metric([b for _, b in paired], "f1"),
        },
        "pairwise_qa": {
            f"{baseline_name}_em_wins": len(prop_wins),
            "ETV4_em_wins": len(etv4_wins),
            "em_ties": len(ties),
            f"{baseline_name}_win_loss_buckets": dict(sorted(loss_buckets.items())),
        },
        "retrieval_pairwise": {
            f"{baseline_name}_title_all5_only": sum(
                1 for a, b in paired if b["title_all5"] and not a["title_all5"]
            ),
            "ETV4_title_all5_only": sum(
                1 for a, b in paired if a["title_all5"] and not b["title_all5"]
            ),
            f"{baseline_name}_title_pool_all200_only": sum(
                1 for a, b in paired if b["title_pool_all200"] and not a["title_pool_all200"]
            ),
            "ETV4_title_pool_all200_only": sum(
                1 for a, b in paired if a["title_pool_all200"] and not b["title_pool_all200"]
            ),
        },
        "examples": {
            "readout_opportunities": example_rows(readout_opportunities[:8]),
            "reader_opportunities": example_rows(reader_opportunities[:8]),
            "pool_failures": example_rows(pool_failures[:8]),
            "etv4_wins": example_rows(etv4_wins[:8]),
        },
    }
    return summary


def example_rows(items: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for etv4, other in items:
        rows.append(
            {
                "qid": etv4["qid"],
                "question": etv4["question"],
                "gold_titles": etv4["gold_titles"],
                "etv4_top5_titles": etv4["top5_titles"],
                "other_top5_titles": other["top5_titles"],
                "etv4_title_all5": etv4["title_all5"],
                "other_title_all5": other["title_all5"],
                "etv4_title_pool_all200": etv4["title_pool_all200"],
                "other_title_pool_all200": other["title_pool_all200"],
                "etv4_em": etv4["em"],
                "other_em": other["em"],
                "etv4_pred": etv4["predicted_answer"],
                "other_pred": other["predicted_answer"],
                "gold_answers": etv4["gold_answers"],
            }
        )
    return rows


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# ETv4 Clean Root-Cause Diagnostic", ""]
    lines.append("Diagnostic-only. Gold labels are used only for offline error classification.")
    lines.append("")
    for row in payload["comparisons"]:
        dataset = row["dataset"]
        baseline = row["baseline"]
        m = row["metrics"]
        q = row["pairwise_qa"]
        r = row["retrieval_pairwise"]
        lines.extend(
            [
                f"## {dataset} vs {baseline}",
                "",
                "```text",
                "Metric                  ETv4      Baseline",
                "----------------------  --------  --------",
                f"Title R@5               {m['ETV4_title_r5']:.4f}    {m[f'{baseline}_title_r5']:.4f}",
                f"Title all-gold@5        {m['ETV4_title_all5']:.4f}    {m[f'{baseline}_title_all5']:.4f}",
                f"Title pool all-gold@200 {m['ETV4_title_pool_all200']:.4f}    {m[f'{baseline}_title_pool_all200']:.4f}",
                f"EM                      {m['ETV4_EM']:.4f}    {m[f'{baseline}_EM']:.4f}",
                f"F1                      {m['ETV4_F1']:.4f}    {m[f'{baseline}_F1']:.4f}",
                "```",
                "",
                "```text",
                f"{baseline} EM wins: {q[f'{baseline}_em_wins']}",
                f"ETv4 EM wins: {q['ETV4_em_wins']}",
                f"EM ties: {q['em_ties']}",
                f"{baseline} win buckets: {q[f'{baseline}_win_loss_buckets']}",
                f"{baseline} title-all@5 only: {r[f'{baseline}_title_all5_only']}",
                f"ETv4 title-all@5 only: {r['ETV4_title_all5_only']}",
                f"{baseline} title-pool-all@200 only: {r[f'{baseline}_title_pool_all200_only']}",
                f"ETv4 title-pool-all@200 only: {r['ETV4_title_pool_all200_only']}",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, Any]] = []
    for dataset in DATASETS:
        comparisons.append(dataset_compare(dataset, "PropRAG", prop_rows(dataset)))
        hippo = hippo_rows(dataset)
        if hippo:
            comparisons.append(dataset_compare(dataset, "HippoRAG", hippo))
    payload = {"comparisons": comparisons}
    (OUT_DIR / "etv4_clean_root_cause.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "etv4_clean_root_cause.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(OUT_DIR / "etv4_clean_root_cause.json")
    print(OUT_DIR / "etv4_clean_root_cause.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
