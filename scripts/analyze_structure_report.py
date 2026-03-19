import argparse
import json
from collections import defaultdict
from pathlib import Path


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _bucket_bridge_edges(trace: dict) -> str:
    scored_doc_count = int(trace.get("scored_doc_count", 0))
    return "scored_docs>=3" if scored_doc_count >= 3 else "scored_docs<3"


def _bucket_swaps(trace: dict) -> str:
    swaps = int(trace.get("num_swaps_top5", 0))
    return "swaps>=3" if swaps >= 3 else "swaps<3"


def _summarize_bucket(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "avg_delta_em": _avg([float(row["delta"]["ExactMatch"]) for row in rows]),
        "avg_delta_f1": _avg([float(row["delta"]["F1"]) for row in rows]),
        "improve_em_count": sum(1 for row in rows if float(row["delta"]["ExactMatch"]) > 0),
        "hurt_em_count": sum(1 for row in rows if float(row["delta"]["ExactMatch"]) < 0),
        "improve_f1_count": sum(1 for row in rows if float(row["delta"]["F1"]) > 0),
        "hurt_f1_count": sum(1 for row in rows if float(row["delta"]["F1"]) < 0),
    }


def _trim_question(question: str, max_len: int = 120) -> str:
    if len(question) <= max_len:
        return question
    return question[: max_len - 1] + "…"


def build_markdown_report(report: dict, source_path: Path) -> str:
    paired_examples = report.get("paired_examples", [])
    triggered = [row for row in paired_examples if (row.get("structure", {}).get("retrieval_trace") or {}).get("applied")]

    winners = sorted(triggered, key=lambda row: float(row["delta"]["F1"]), reverse=True)[:10]
    losers = sorted(triggered, key=lambda row: float(row["delta"]["F1"]))[:10]

    query_type_buckets: dict[str, list[dict]] = defaultdict(list)
    bridge_buckets: dict[str, list[dict]] = defaultdict(list)
    swap_buckets: dict[str, list[dict]] = defaultdict(list)

    for row in triggered:
        trace = row.get("structure", {}).get("retrieval_trace") or {}
        query_type_buckets[str(row.get("query_type", "unknown"))].append(row)
        bridge_buckets[_bucket_bridge_edges(trace)].append(row)
        swap_buckets[_bucket_swaps(trace)].append(row)

    lines: list[str] = []
    lines.append(f"# Structure Trigger Report: {report.get('dataset', 'unknown')} ({report.get('limit', 'unknown')})")
    lines.append("")
    lines.append(f"Source JSON: `{source_path}`")
    lines.append("")

    delta = report.get("delta", {})
    paired_summary = report.get("paired_summary", {})
    structure_analysis = report.get("structure", {}).get("structure_analysis", {})
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Overall delta: EM `{float(delta.get('ExactMatch', 0.0)):+.4f}`, "
        f"F1 `{float(delta.get('F1', 0.0)):+.4f}`, Recall@1 `{float(delta.get('Recall@1', 0.0)):+.4f}`, "
        f"Recall@2 `{float(delta.get('Recall@2', 0.0)):+.4f}`."
    )
    lines.append(
        f"- Triggering: `{structure_analysis.get('structure_enabled_count', 0)}` / "
        f"`{structure_analysis.get('num_queries', 0)}` queries "
        f"(`{100.0 * float(structure_analysis.get('structure_enabled_rate', 0.0)):.1f}%`)."
    )
    lines.append(
        f"- Pairwise outcomes among all queries: EM improve/hurt/tie = "
        f"`{paired_summary.get('improve_em_count', 0)}` / `{paired_summary.get('hurt_em_count', 0)}` / "
        f"`{paired_summary.get('tie_em_count', 0)}`, "
        f"F1 improve/hurt/tie = `{paired_summary.get('improve_f1_count', 0)}` / "
        f"`{paired_summary.get('hurt_f1_count', 0)}` / `{paired_summary.get('tie_f1_count', 0)}`."
    )
    lines.append(
        f"- Parse failures still high: `{structure_analysis.get('parse_failure_example_count', 0)}` examples had at least one rerank parse failure."
    )
    lines.append("")

    lines.append("## Trigger Bucket Summary")
    lines.append("")
    lines.append("| Bucket | Count | Avg ΔEM | Avg ΔF1 | EM + | EM - | F1 + | F1 - |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| all_triggered | "
        f"{len(triggered)} | {_avg([float(row['delta']['ExactMatch']) for row in triggered]):+.4f} | "
        f"{_avg([float(row['delta']['F1']) for row in triggered]):+.4f} | "
        f"{sum(1 for row in triggered if float(row['delta']['ExactMatch']) > 0)} | "
        f"{sum(1 for row in triggered if float(row['delta']['ExactMatch']) < 0)} | "
        f"{sum(1 for row in triggered if float(row['delta']['F1']) > 0)} | "
        f"{sum(1 for row in triggered if float(row['delta']['F1']) < 0)} |"
    )
    for bucket_name, rows in sorted(query_type_buckets.items()):
        summary = _summarize_bucket(rows)
        lines.append(
            f"| query_type={bucket_name} | {summary['count']} | {summary['avg_delta_em']:+.4f} | "
            f"{summary['avg_delta_f1']:+.4f} | {summary['improve_em_count']} | {summary['hurt_em_count']} | "
            f"{summary['improve_f1_count']} | {summary['hurt_f1_count']} |"
        )
    for bucket_name, rows in sorted(bridge_buckets.items()):
        summary = _summarize_bucket(rows)
        lines.append(
            f"| {bucket_name} | {summary['count']} | {summary['avg_delta_em']:+.4f} | "
            f"{summary['avg_delta_f1']:+.4f} | {summary['improve_em_count']} | {summary['hurt_em_count']} | "
            f"{summary['improve_f1_count']} | {summary['hurt_f1_count']} |"
        )
    for bucket_name, rows in sorted(swap_buckets.items()):
        summary = _summarize_bucket(rows)
        lines.append(
            f"| {bucket_name} | {summary['count']} | {summary['avg_delta_em']:+.4f} | "
            f"{summary['avg_delta_f1']:+.4f} | {summary['improve_em_count']} | {summary['hurt_em_count']} | "
            f"{summary['improve_f1_count']} | {summary['hurt_f1_count']} |"
        )
    lines.append("")

    def _append_examples(title: str, rows: list[dict]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| ΔEM | ΔF1 | Swaps | Scored Docs | Query Type | Question |")
        lines.append("|---:|---:|---:|---:|---|---|")
        for row in rows:
            trace = row.get("structure", {}).get("retrieval_trace") or {}
            lines.append(
                f"| {float(row['delta']['ExactMatch']):+.4f} | {float(row['delta']['F1']):+.4f} | "
                f"{int(trace.get('num_swaps_top5', 0))} | {int(trace.get('scored_doc_count', 0))} | "
                f"{row.get('query_type', 'unknown')} | {_trim_question(str(row.get('question', '')))} |"
            )
        lines.append("")

    _append_examples("Top Winners", winners)
    _append_examples("Top Losers", losers)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a structure-rerank comparison report and emit a Markdown summary.")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    report = json.loads(input_path.read_text())
    markdown = build_markdown_report(report, input_path)

    if args.output_md:
        output_path = Path(args.output_md)
    else:
        output_path = input_path.with_suffix(".trigger_report.md")
    output_path.write_text(markdown)
    print(json.dumps({"output_md": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
