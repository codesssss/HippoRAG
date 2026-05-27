#!/usr/bin/env python3
"""Build a v13b-compatible source report from exported retrieval pools.

v13b is a top-k evidence selector over an existing candidate universe.  This
adapter converts the current pool export format into the source-report schema
expected by ``evaluate_obligation_closed_sto_local_ppr.py``:

```
{
  "datasets": [
    {
      "dataset": "...",
      "report_path": "...",
      "openie_path": "...",
      "rows": [
        {
          "query_index": 0,
          "question": "...",
          "gold_doc_indices": [...],
          "candidate_doc_indices": [...],
          "retrieved_doc_indices_top5": [...],
          "retrieved_doc_indices_top10": [...]
        }
      ]
    }
  ]
}
```

The input pool exports used in this repository store integer document ids in
``pool_doc_ids``.  Those ids are the OpenIE doc indices, so the adapter keeps
them unchanged and only resolves gold titles against the OpenIE cache.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DATASETS = "2wikimultihopqa,musique,hotpotqa"
DEFAULT_POOL_TEMPLATE = "run_logs/proprag_pool_exports_full1000_20260424/{dataset}_pool100.json"
DEFAULT_OPENIE_TEMPLATE = "outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json"
DEFAULT_OUTPUT_JSON = "run_logs/v13b_source_report_from_proprag_pool100.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_passage(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique_ints(values: Iterable[Any], *, limit: int = 0) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item < 0 or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if limit > 0 and len(result) >= limit:
            break
    return result


def openie_docs_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        docs = payload.get("docs", [])
    else:
        docs = payload
    if not isinstance(docs, Sequence) or isinstance(docs, (str, bytes)):
        raise ValueError("OpenIE payload must be a mapping with `docs` or a list of docs")
    return [doc for doc in docs if isinstance(doc, Mapping)]


def title_for_doc(doc: Mapping[str, Any]) -> str:
    passage = str(doc.get("passage") or "")
    return passage.split("\n", 1)[0].strip() if passage else ""


def build_title_index(openie_docs: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for doc_idx, doc in enumerate(openie_docs):
        title = normalize_title(title_for_doc(doc))
        if not title:
            continue
        index.setdefault(title, []).append(int(doc_idx))
    return index


def build_passage_index(openie_docs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for doc_idx, doc in enumerate(openie_docs):
        passage = normalize_passage(doc.get("passage"))
        if passage and passage not in index:
            index[passage] = int(doc_idx)
    return index


def resolve_gold_doc_indices(
    *,
    record: Mapping[str, Any],
    passage_index: Mapping[str, int],
    title_index: Mapping[str, Sequence[int]],
) -> list[int]:
    gold_indices: list[int] = []
    for passage in record.get("gold_docs", []) or []:
        normalized_passage = normalize_passage(passage)
        if normalized_passage in passage_index:
            gold_indices.append(int(passage_index[normalized_passage]))
    if gold_indices:
        return unique_ints(gold_indices)

    for title in record.get("gold_titles", []) or []:
        normalized = normalize_title(title)
        if normalized in title_index:
            gold_indices.extend(int(idx) for idx in title_index[normalized])
    if gold_indices:
        return unique_ints(gold_indices)

    # Fallback for exports that include full gold passages but omit titles.
    for passage in record.get("gold_docs", []) or []:
        title = str(passage or "").split("\n", 1)[0].strip()
        normalized = normalize_title(title)
        if normalized in title_index:
            gold_indices.extend(int(idx) for idx in title_index[normalized])
    return unique_ints(gold_indices)


def records_from_pool_payload(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        records = payload.get("records", [])
    else:
        records = payload
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Pool payload must be a mapping with `records` or a list of records")
    return [record for record in records if isinstance(record, Mapping)]


def build_row(
    *,
    record: Mapping[str, Any],
    passage_index: Mapping[str, int],
    title_index: Mapping[str, Sequence[int]],
    candidate_limit: int,
) -> dict[str, Any]:
    candidates = unique_ints(record.get("pool_doc_ids", []) or [], limit=max(int(candidate_limit), 0))
    if not candidates:
        candidates = unique_ints(record.get("candidate_doc_indices", []) or [], limit=max(int(candidate_limit), 0))
    gold_doc_indices = resolve_gold_doc_indices(
        record=record,
        passage_index=passage_index,
        title_index=title_index,
    )
    query_index = record.get("query_index", record.get("query_idx", record.get("qid", 0)))
    row = {
        "query_index": int(query_index),
        "question": str(record.get("question") or record.get("query") or ""),
        "gold_answers": list(record.get("gold_answers", []) or []),
        "gold_titles": list(record.get("gold_titles", []) or []),
        "gold_doc_indices": gold_doc_indices,
        "candidate_doc_indices": candidates,
        "retrieved_doc_indices": candidates,
        "retrieved_doc_indices_top5": candidates[:5],
        "retrieved_doc_indices_top10": candidates[:10],
        "retrieved_doc_indices_top20": candidates[:20],
        "anchor_guided_evidence_doc_indices_top5": candidates[:5],
        "anchor_guided_evidence_doc_indices_top10": candidates[:10],
    }
    if record.get("pool_doc_scores"):
        row["candidate_doc_scores"] = list(record.get("pool_doc_scores", []) or [])[: len(candidates)]
    if record.get("pool_titles"):
        row["candidate_titles"] = list(record.get("pool_titles", []) or [])[: len(candidates)]
    return row


def build_dataset_payload(
    *,
    dataset: str,
    pool_path: Path,
    openie_path: Path,
    output_report_path: Path,
    max_queries: int,
    candidate_limit: int,
) -> dict[str, Any]:
    pool_payload = load_json(pool_path)
    openie_docs = openie_docs_from_payload(load_json(openie_path))
    passage_index = build_passage_index(openie_docs)
    title_index = build_title_index(openie_docs)
    records = records_from_pool_payload(pool_payload)
    if max_queries > 0:
        records = records[:max_queries]
    rows = [
        build_row(
            record=record,
            passage_index=passage_index,
            title_index=title_index,
            candidate_limit=candidate_limit,
        )
        for record in records
    ]
    missing_gold = sum(1 for row in rows if not row["gold_doc_indices"])
    return {
        "dataset": dataset,
        "report_path": str(output_report_path),
        "openie_path": str(openie_path),
        "rows": rows,
        "metrics": {
            "row_count": len(rows),
            "candidate_limit": int(candidate_limit),
            "missing_gold_doc_index_rows": int(missing_gold),
        },
    }


def build_report(
    *,
    datasets: Sequence[str],
    pool_template: str,
    openie_template: str,
    output_report_path: Path,
    max_queries: int,
    candidate_limit: int,
) -> dict[str, Any]:
    dataset_payloads: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        pool_path = Path(pool_template.format(dataset=dataset)).resolve()
        openie_path = Path(openie_template.format(dataset=dataset)).resolve()
        payload = build_dataset_payload(
            dataset=dataset,
            pool_path=pool_path,
            openie_path=openie_path,
            output_report_path=output_report_path.resolve(),
            max_queries=max_queries,
            candidate_limit=candidate_limit,
        )
        dataset_payloads.append(payload)
        variant_rows.extend(dict(row, dataset=dataset) for row in payload["rows"])

    return {
        "source": "v13b_make_source_report",
        "pool_template": pool_template,
        "openie_template": openie_template,
        "max_queries": int(max_queries),
        "candidate_limit": int(candidate_limit),
        "datasets": dataset_payloads,
        # Compatibility for run_transition_top5_qa when this file is used as
        # the source report.  Retrieval-side v13b ignores this section.
        "variants": {
            "hippohead_qgate_lexbeam_topkfact_stabilityfallback_guarded_local_ppr_gamma_0.3": variant_rows
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a v13b source report from exported pool100 files.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--pool-template", default=DEFAULT_POOL_TEMPLATE)
    parser.add_argument("--openie-template", default=DEFAULT_OPENIE_TEMPLATE)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args(argv)

    output_json = Path(args.output_json)
    report = build_report(
        datasets=parse_csv(args.datasets),
        pool_template=args.pool_template,
        openie_template=args.openie_template,
        output_report_path=output_json,
        max_queries=int(args.max_queries),
        candidate_limit=int(args.candidate_limit),
    )
    write_json(output_json, report)
    print(f"Wrote v13b source report: {output_json} ({sum(len(d['rows']) for d in report['datasets'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
