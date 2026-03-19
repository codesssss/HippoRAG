import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _norm_status(value: object) -> str:
    if value is None:
        return "missing"
    return str(value)


def _trim(text: object, limit: int = 200) -> str:
    value = "" if text is None else str(text).replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reranker attempt quality from a struct_compare report.")
    parser.add_argument("--report", required=True, help="Path to struct_compare JSON report.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for preview sampling.")
    parser.add_argument("--preview_limit", type=int, default=2, help="Number of failed attempt1 previews to print.")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    paired_examples = report.get("paired_examples") or []

    attempt1_status = Counter()
    attempt2_status = Counter()
    attempt1_errors = Counter()
    attempt2_called = 0
    attempt2_success = 0
    attempt1_truncated_count = 0
    attempt2_truncated_count = 0
    prompt_changed_count = int((report.get("paired_summary") or {}).get("reader_prompt_changed_count", 0))
    improve = hurt = tie = 0
    failed_attempt1_rows: list[dict] = []

    for row in paired_examples:
        delta_f1 = float((row.get("delta") or {}).get("F1", 0.0))
        if delta_f1 > 0:
            improve += 1
        elif delta_f1 < 0:
            hurt += 1
        else:
            tie += 1

        trace = ((row.get("structure") or {}).get("retrieval_trace")) or {}
        a1 = _norm_status(trace.get("rerank_attempt1_status"))
        a2 = _norm_status(trace.get("rerank_attempt2_status"))
        attempt1_status[a1] += 1
        attempt2_status[a2] += 1
        if a2 != "missing":
            attempt2_called += 1
        if a2 == "ok":
            attempt2_success += 1
        if bool(trace.get("rerank_attempt1_truncated", False)):
            attempt1_truncated_count += 1
        if bool(trace.get("rerank_attempt2_truncated", False)):
            attempt2_truncated_count += 1
        a1_error = trace.get("rerank_attempt1_error")
        if a1_error:
            attempt1_errors[_trim(a1_error, 120)] += 1

        if a1 != "ok":
            failed_attempt1_rows.append({
                "question": row.get("question", ""),
                "status": a1,
                "error": trace.get("rerank_attempt1_error"),
                "preview": trace.get("rerank_attempt1_raw_output_preview"),
            })

    rng = random.Random(args.seed)
    rng.shuffle(failed_attempt1_rows)
    preview_rows = failed_attempt1_rows[: max(0, args.preview_limit)]

    output = {
        "report": args.report,
        "num_queries": len(paired_examples),
        "attempt1_status_counts": dict(sorted(attempt1_status.items())),
        "attempt2_status_counts": dict(sorted(attempt2_status.items())),
        "attempt1_parse_fail_example_count": attempt1_status.get("parse_failure", 0),
        "attempt1_schema_fail_example_count": attempt1_status.get("schema_failure", 0),
        "attempt2_called_count": attempt2_called,
        "attempt2_success_count": attempt2_success,
        "attempt1_truncated_count": attempt1_truncated_count,
        "attempt2_truncated_count": attempt2_truncated_count,
        "prompt_changed_count": prompt_changed_count,
        "improve_hurt_tie": {
            "improve": improve,
            "hurt": hurt,
            "tie": tie,
        },
        "top_attempt1_errors": attempt1_errors.most_common(10),
        "attempt1_failure_previews": [
            {
                "question": row["question"],
                "status": row["status"],
                "error": _trim(row["error"], 180),
                "preview": _trim(row["preview"], 220),
            }
            for row in preview_rows
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
