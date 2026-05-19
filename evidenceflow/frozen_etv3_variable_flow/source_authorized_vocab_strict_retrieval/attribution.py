"""Loss/gain attribution for source-authorized vocab-strict GraphRAG outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .evaluate_report import recall_at_k, unique_ints


def _load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _passage_title(text: object) -> str:
    raw = str(text or "")
    return raw.splitlines()[0].strip() if raw else ""


def _load_titles(openie_path: Path) -> Dict[int, str]:
    payload = _load_json(openie_path)
    docs = payload.get("docs", []) if isinstance(payload, Mapping) else []
    output: Dict[int, str] = {}
    for index, row in enumerate(docs or []):
        if not isinstance(row, Mapping):
            continue
        output[int(index)] = _passage_title(row.get("passage", ""))
    return output


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _certificate_for_doc(row: Mapping[str, Any], doc_index: int) -> Mapping[str, Any]:
    trace = row.get("route_trace", {}) or {}
    retrieval_trace = trace.get("retrieval", trace) if isinstance(trace, Mapping) else {}
    parent_certificates = (
        retrieval_trace.get("parent_certificates", {}) if isinstance(retrieval_trace, Mapping) else {}
    )
    certificate = parent_certificates.get(str(int(doc_index)))
    if isinstance(certificate, Mapping):
        return certificate
    certificate = parent_certificates.get(int(doc_index))
    return certificate if isinstance(certificate, Mapping) else {}


def _expanded_docs(row: Mapping[str, Any]) -> set[int]:
    trace = row.get("route_trace", {}) or {}
    candidate_expansion = (
        trace.get("candidate_expansion", {}) if isinstance(trace, Mapping) else {}
    )
    if not isinstance(candidate_expansion, Mapping):
        return set()
    return set(unique_ints(candidate_expansion.get("expanded_doc_indices", []) or []))


def _retrieval_trace(row: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = row.get("route_trace", {}) or {}
    if not isinstance(trace, Mapping):
        return {}
    retrieval = trace.get("retrieval", trace)
    return retrieval if isinstance(retrieval, Mapping) else {}


def _doc_title(titles: Mapping[int, str], doc_index: int) -> str:
    return str(titles.get(int(doc_index), ""))


def _certificate_key(certificate: Mapping[str, Any]) -> str:
    certificate_type = str(certificate.get("certificate_type") or "no_parent_certificate")
    roles = certificate.get("roles", []) or []
    if roles:
        return f"{certificate_type}:{','.join(str(role) for role in roles)}"
    return certificate_type


def _inserted_record(
    *,
    row: Mapping[str, Any],
    doc_index: int,
    dense_top5: Sequence[int],
    titles: Mapping[int, str],
) -> Dict[str, Any]:
    certificate = dict(_certificate_for_doc(row, int(doc_index)))
    source_doc_index = certificate.get("source_doc_index")
    try:
        source_doc_index = int(source_doc_index)
    except (TypeError, ValueError):
        source_doc_index = None
    expanded = _expanded_docs(row)
    candidate_docs = set(unique_ints(row.get("candidate_doc_indices", []) or []))
    if int(doc_index) in expanded:
        origin = "expanded_only"
    elif int(doc_index) in candidate_docs:
        origin = "candidate_pool"
    else:
        origin = "unknown"
    return {
        "doc_index": int(doc_index),
        "title": _doc_title(titles, int(doc_index)),
        "origin": origin,
        "was_in_dense_top5": int(doc_index) in set(unique_ints(dense_top5)),
        "parent_certificate": certificate,
        "parent_certificate_key": _certificate_key(certificate),
        "source_doc_index": source_doc_index,
        "source_title": _doc_title(titles, source_doc_index) if source_doc_index is not None else "",
        "source_in_dense_top5": (
            source_doc_index in set(unique_ints(dense_top5)) if source_doc_index is not None else False
        ),
    }


def analyze_output(payload: Mapping[str, Any]) -> Dict[str, Any]:
    openie_path = Path(str(payload.get("openie_path") or "")).expanduser()
    titles = _load_titles(openie_path) if str(openie_path) else {}
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, Mapping)]
    total_delta = 0.0
    changed = 0
    gain_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    neutral_changed_rows: List[Dict[str, Any]] = []
    inserted_by_outcome: Dict[str, Counter[str]] = {
        "gain": Counter(),
        "loss": Counter(),
        "neutral_changed": Counter(),
    }
    origin_by_outcome: Dict[str, Counter[str]] = {
        "gain": Counter(),
        "loss": Counter(),
        "neutral_changed": Counter(),
    }
    replacement_source_counts: Dict[str, Counter[str]] = {
        "gain": Counter(),
        "loss": Counter(),
        "neutral_changed": Counter(),
    }
    loss_displaced_gold = 0
    loss_inserted_non_gold = 0
    loss_expanded_insertions = 0
    loss_pair_insertions = 0

    examples: Dict[str, List[Dict[str, Any]]] = {
        "gain": [],
        "loss": [],
        "neutral_changed": [],
    }

    for fallback_idx, row in enumerate(rows):
        gold = unique_ints(row.get("gold_doc_indices", []) or [])
        dense_top5 = unique_ints(row.get("candidate_doc_indices", []) or [])[:5]
        current_top5 = unique_ints(row.get("retrieved_doc_indices_top5", []) or [])[:5]
        dense_r5 = recall_at_k(gold, dense_top5, 5)
        current_r5 = recall_at_k(gold, current_top5, 5)
        delta = float(current_r5) - float(dense_r5)
        total_delta += delta
        if dense_top5 != current_top5:
            changed += 1
        inserted = [doc for doc in current_top5 if int(doc) not in set(dense_top5)]
        removed = [doc for doc in dense_top5 if int(doc) not in set(current_top5)]
        if delta > 1e-12:
            outcome = "gain"
        elif delta < -1e-12:
            outcome = "loss"
        elif dense_top5 != current_top5:
            outcome = "neutral_changed"
        else:
            continue
        inserted_records = [
            _inserted_record(row=row, doc_index=doc, dense_top5=dense_top5, titles=titles)
            for doc in inserted
        ]
        for record in inserted_records:
            inserted_by_outcome[outcome][str(record["parent_certificate_key"])] += 1
            origin_by_outcome[outcome][str(record["origin"])] += 1
            replacement_source_counts[outcome][str(record["source_in_dense_top5"])] += 1
        removed_gold = [doc for doc in removed if int(doc) in set(gold)]
        inserted_gold = [doc for doc in inserted if int(doc) in set(gold)]
        record = {
            "query_index": int(row.get("query_index", fallback_idx)),
            "question": str(row.get("question") or ""),
            "gold_doc_indices": gold,
            "dense_top5": dense_top5,
            "current_top5": current_top5,
            "dense_r5": round(float(dense_r5), 6),
            "current_r5": round(float(current_r5), 6),
            "delta_r5": round(float(delta), 6),
            "inserted_docs": inserted_records,
            "removed_docs": [
                {
                    "doc_index": int(doc),
                    "title": _doc_title(titles, int(doc)),
                    "was_gold": int(doc) in set(gold),
                }
                for doc in removed
            ],
            "removed_gold_doc_indices": removed_gold,
            "inserted_gold_doc_indices": inserted_gold,
            "retrieval_trace_core": {
                key: _retrieval_trace(row).get(key)
                for key in (
                    "inserted_doc_indices",
                    "replacement_source_doc_indices",
                    "entry_doc_indices",
                    "protected_doc_indices",
                    "certified_doc_indices",
                    "closed_certificate_count_topk",
                    "dense_fill_count",
                )
            },
        }
        if outcome == "gain":
            gain_rows.append(record)
        elif outcome == "loss":
            loss_rows.append(record)
            if removed_gold:
                loss_displaced_gold += 1
            if any(int(doc) not in set(gold) for doc in inserted):
                loss_inserted_non_gold += 1
            if any(str(item["origin"]) == "expanded_only" for item in inserted_records):
                loss_expanded_insertions += 1
            if len(inserted) >= 2:
                loss_pair_insertions += 1
        else:
            neutral_changed_rows.append(record)
        if len(examples[outcome]) < 8:
            examples[outcome].append(record)

    row_count = len(rows)
    denom = float(max(row_count, 1))
    return {
        "dataset": str(payload.get("dataset") or ""),
        "method": str(payload.get("method") or ""),
        "row_count": row_count,
        "metrics": dict(payload.get("metrics", {}) or {}),
        "dense_baseline_r5": round(
            sum(
                recall_at_k(
                    unique_ints(row.get("gold_doc_indices", []) or []),
                    unique_ints(row.get("candidate_doc_indices", []) or [])[:5],
                    5,
                )
                for row in rows
            )
            / denom,
            6,
        ),
        "current_r5": round(float((payload.get("metrics", {}) or {}).get("r5", 0.0)), 6),
        "mean_delta_r5": round(total_delta / denom, 6),
        "changed_count": changed,
        "gain_query_count": len(gain_rows),
        "loss_query_count": len(loss_rows),
        "neutral_changed_query_count": len(neutral_changed_rows),
        "loss_displaced_gold_query_count": loss_displaced_gold,
        "loss_inserted_non_gold_query_count": loss_inserted_non_gold,
        "loss_expanded_insertion_query_count": loss_expanded_insertions,
        "loss_pair_insertion_query_count": loss_pair_insertions,
        "inserted_certificate_counts_by_outcome": {
            outcome: dict(counter.most_common())
            for outcome, counter in inserted_by_outcome.items()
        },
        "inserted_origin_counts_by_outcome": {
            outcome: dict(counter.most_common())
            for outcome, counter in origin_by_outcome.items()
        },
        "source_in_dense_top5_counts_by_outcome": {
            outcome: dict(counter.most_common())
            for outcome, counter in replacement_source_counts.items()
        },
        "examples": examples,
    }


def analyze_files(paths: Sequence[Path]) -> Dict[str, Any]:
    datasets = []
    for path in paths:
        payload = _load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"expected JSON object: {path}")
        summary = analyze_output(payload)
        summary["input_path"] = str(path)
        datasets.append(summary)
    return {"datasets": datasets}


def _format_counter(counter: Mapping[str, int], limit: int = 4) -> str:
    if not counter:
        return "-"
    items = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}={value}" for key, value in items)


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    datasets = [row for row in payload.get("datasets", []) or [] if isinstance(row, Mapping)]
    lines: List[str] = [
        "# Full1000 Loss/Gain Attribution",
        "",
        "This report compares the clean GraphRAG top5 against the incoming dense top5 in the same candidate universe.",
        "",
        "| Dataset | Dense R@5 | Current R@5 | Delta | Changed | GainQ | LossQ | Neutral changed | Loss displaced gold | Loss inserted non-gold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in datasets:
        lines.append(
            "| {dataset} | {dense:.4f} | {current:.4f} | {delta:+.4f} | {changed} | {gain} | {loss} | {neutral} | {loss_gold} | {loss_nongold} |".format(
                dataset=row.get("dataset", ""),
                dense=float(row.get("dense_baseline_r5", 0.0)),
                current=float(row.get("current_r5", 0.0)),
                delta=float(row.get("mean_delta_r5", 0.0)),
                changed=int(row.get("changed_count", 0)),
                gain=int(row.get("gain_query_count", 0)),
                loss=int(row.get("loss_query_count", 0)),
                neutral=int(row.get("neutral_changed_query_count", 0)),
                loss_gold=int(row.get("loss_displaced_gold_query_count", 0)),
                loss_nongold=int(row.get("loss_inserted_non_gold_query_count", 0)),
            )
        )
    lines.extend(
        [
            "",
            "| Dataset | Gain certificates | Loss certificates | Gain origins | Loss origins | Loss pair insertions | Loss expanded insertions |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in datasets:
        cert_counts = row.get("inserted_certificate_counts_by_outcome", {}) or {}
        origin_counts = row.get("inserted_origin_counts_by_outcome", {}) or {}
        lines.append(
            "| {dataset} | {gain_cert} | {loss_cert} | {gain_origin} | {loss_origin} | {loss_pair} | {loss_expanded} |".format(
                dataset=row.get("dataset", ""),
                gain_cert=_format_counter(cert_counts.get("gain", {}) or {}),
                loss_cert=_format_counter(cert_counts.get("loss", {}) or {}),
                gain_origin=_format_counter(origin_counts.get("gain", {}) or {}),
                loss_origin=_format_counter(origin_counts.get("loss", {}) or {}),
                loss_pair=int(row.get("loss_pair_insertion_query_count", 0)),
                loss_expanded=int(row.get("loss_expanded_insertion_query_count", 0)),
            )
        )
    lines.append("")
    for row in datasets:
        lines.extend(
            [
                f"## {row.get('dataset', '')} Loss Examples",
                "",
                "| QID | dR@5 | Question | Gold | Dense top5 | Current top5 | Inserted | Removed gold |",
                "| ---: | ---: | --- | --- | --- | --- | --- | --- |",
            ]
        )
        examples = ((row.get("examples", {}) or {}).get("loss", []) or [])[:8]
        if not examples:
            lines.append("| - | - | - | - | - | - | - | - |")
        for example in examples:
            inserted_text = "; ".join(
                "{doc}:{title} [{cert}] <- {src}:{src_title}".format(
                    doc=item.get("doc_index"),
                    title=str(item.get("title", ""))[:80],
                    cert=item.get("parent_certificate_key", ""),
                    src=item.get("source_doc_index", ""),
                    src_title=str(item.get("source_title", ""))[:60],
                )
                for item in example.get("inserted_docs", []) or []
            )
            lines.append(
                "| {qid} | {delta:+.3f} | {question} | {gold} | {dense} | {current} | {inserted} | {removed_gold} |".format(
                    qid=example.get("query_index"),
                    delta=float(example.get("delta_r5", 0.0)),
                    question=str(example.get("question", "")).replace("|", "/")[:180],
                    gold=example.get("gold_doc_indices", []),
                    dense=example.get("dense_top5", []),
                    current=example.get("current_top5", []),
                    inserted=inserted_text.replace("|", "/"),
                    removed_gold=example.get("removed_gold_doc_indices", []),
                )
            )
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = analyze_files([Path(path).expanduser() for path in args.inputs])
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(payload, Path(args.output_md).expanduser())
    compact = {
        row["dataset"]: {
            "dense_r5": row["dense_baseline_r5"],
            "current_r5": row["current_r5"],
            "gain_q": row["gain_query_count"],
            "loss_q": row["loss_query_count"],
        }
        for row in payload["datasets"]
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
