import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(max(variance, 0.0)))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return float(lower_value + (upper_value - lower_value) * weight)


def _bootstrap_mean_ci(values: list[float], num_samples: int, seed: int, ci: float) -> dict:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "ci_low": value, "ci_high": value}

    rng = random.Random(seed)
    sample_size = len(values)
    bootstrap_means = []
    for _ in range(num_samples):
        resample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        bootstrap_means.append(_mean(resample))
    bootstrap_means.sort()

    alpha = max(0.0, min(1.0, 1.0 - ci))
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    return {
        "mean": _mean(values),
        "ci_low": _quantile(bootstrap_means, lower_q),
        "ci_high": _quantile(bootstrap_means, upper_q),
    }


def _bootstrap_mean_difference(values_a: list[float],
                               values_b: list[float],
                               num_samples: int,
                               seed: int,
                               ci: float) -> dict:
    if not values_a or not values_b:
        return {
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_gt_zero": 0.0,
            "p_lt_zero": 0.0,
        }

    rng = random.Random(seed)
    sample_size_a = len(values_a)
    sample_size_b = len(values_b)
    diffs = []
    for _ in range(num_samples):
        resample_a = [values_a[rng.randrange(sample_size_a)] for _ in range(sample_size_a)]
        resample_b = [values_b[rng.randrange(sample_size_b)] for _ in range(sample_size_b)]
        diffs.append(_mean(resample_a) - _mean(resample_b))
    diffs.sort()

    alpha = max(0.0, min(1.0, 1.0 - ci))
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    mean_diff = _mean(values_a) - _mean(values_b)
    p_gt_zero = sum(1 for value in diffs if value > 0.0) / len(diffs)
    p_lt_zero = sum(1 for value in diffs if value < 0.0) / len(diffs)
    return {
        "mean_diff": mean_diff,
        "ci_low": _quantile(diffs, lower_q),
        "ci_high": _quantile(diffs, upper_q),
        "p_gt_zero": float(p_gt_zero),
        "p_lt_zero": float(p_lt_zero),
    }


def _load_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    paired_examples = report.get("paired_examples", [])
    per_query_em = [float(row.get("delta", {}).get("ExactMatch", 0.0)) for row in paired_examples]
    per_query_f1 = [float(row.get("delta", {}).get("F1", 0.0)) for row in paired_examples]
    return {
        "path": str(path),
        "name": path.name,
        "dataset": report.get("dataset"),
        "limit": int(report.get("limit", 0)),
        "delta_em": float(report.get("delta", {}).get("ExactMatch", 0.0)),
        "delta_f1": float(report.get("delta", {}).get("F1", 0.0)),
        "paired_summary": report.get("paired_summary", {}),
        "per_query_delta_em": per_query_em,
        "per_query_delta_f1": per_query_f1,
    }


def _infer_group(name: str, group_regex: str | None) -> str:
    lower_name = name.lower()
    if group_regex:
        match = re.search(group_regex, name)
        if match:
            if match.groups():
                return match.group(1)
            return match.group(0)

    if "nonempty" in lower_name:
        return "nonempty"
    if "edge2" in lower_name:
        return "edge2"
    return Path(name).stem


def _collect_paths(inputs: list[str], glob_patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        paths.append(Path(item))
    for pattern in glob_patterns:
        paths.extend(sorted(Path().glob(pattern)))

    deduped: list[Path] = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _summarize_group(group_name: str,
                     runs: list[dict],
                     num_bootstrap_samples: int,
                     seed: int,
                     ci: float) -> dict:
    run_delta_em = [run["delta_em"] for run in runs]
    run_delta_f1 = [run["delta_f1"] for run in runs]
    all_query_delta_em = [value for run in runs for value in run["per_query_delta_em"]]
    all_query_delta_f1 = [value for run in runs for value in run["per_query_delta_f1"]]

    return {
        "group": group_name,
        "num_runs": len(runs),
        "run_files": [run["name"] for run in runs],
        "run_summaries": [
            {
                "name": run["name"],
                "path": run["path"],
                "delta_em": run["delta_em"],
                "delta_f1": run["delta_f1"],
                "structure_triggered_count": int(run["paired_summary"].get("structure_triggered_count", 0)),
                "improve_f1_count": int(run["paired_summary"].get("improve_f1_count", 0)),
                "hurt_f1_count": int(run["paired_summary"].get("hurt_f1_count", 0)),
            }
            for run in runs
        ],
        "delta_em": {
            "run_mean": _mean(run_delta_em),
            "run_std": _std(run_delta_em),
            "run_values": run_delta_em,
            "pooled_query_bootstrap": _bootstrap_mean_ci(
                values=all_query_delta_em,
                num_samples=num_bootstrap_samples,
                seed=seed + 17,
                ci=ci,
            ),
        },
        "delta_f1": {
            "run_mean": _mean(run_delta_f1),
            "run_std": _std(run_delta_f1),
            "run_values": run_delta_f1,
            "pooled_query_bootstrap": _bootstrap_mean_ci(
                values=all_query_delta_f1,
                num_samples=num_bootstrap_samples,
                seed=seed + 31,
                ci=ci,
            ),
        },
        "structure_triggered_count_mean": _mean(
            [float(run["paired_summary"].get("structure_triggered_count", 0)) for run in runs]
        ),
        "improve_f1_count_mean": _mean(
            [float(run["paired_summary"].get("improve_f1_count", 0)) for run in runs]
        ),
        "hurt_f1_count_mean": _mean(
            [float(run["paired_summary"].get("hurt_f1_count", 0)) for run in runs]
        ),
        "pooled_query_delta_em": all_query_delta_em,
        "pooled_query_delta_f1": all_query_delta_f1,
    }


def _build_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Structure Run Aggregate Summary")
    lines.append("")
    lines.append(
        f"- Inputs: `{summary['num_reports']}` reports across `{len(summary['groups'])}` groups."
    )
    lines.append(
        f"- Bootstrap: `{summary['num_bootstrap_samples']}` samples, CI `{summary['ci']:.1%}`."
    )
    lines.append("")
    lines.append("## Groups")
    lines.append("")
    lines.append("| Group | Runs | Run Mean ΔEM | Run Std ΔEM | Bootstrap ΔEM CI | Run Mean ΔF1 | Run Std ΔF1 | Bootstrap ΔF1 CI | Avg Triggered | Avg Improve F1 | Avg Hurt F1 |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|")
    for group in summary["groups"]:
        em_ci = group["delta_em"]["pooled_query_bootstrap"]
        f1_ci = group["delta_f1"]["pooled_query_bootstrap"]
        lines.append(
            f"| {group['group']} | {group['num_runs']} | "
            f"{group['delta_em']['run_mean']:+.4f} | {group['delta_em']['run_std']:.4f} | "
            f"[{em_ci['ci_low']:+.4f}, {em_ci['ci_high']:+.4f}] | "
            f"{group['delta_f1']['run_mean']:+.4f} | {group['delta_f1']['run_std']:.4f} | "
            f"[{f1_ci['ci_low']:+.4f}, {f1_ci['ci_high']:+.4f}] | "
            f"{group['structure_triggered_count_mean']:.2f} | "
            f"{group['improve_f1_count_mean']:.2f} | "
            f"{group['hurt_f1_count_mean']:.2f} |"
        )
    lines.append("")

    for group in summary["groups"]:
        lines.append(f"## {group['group']}")
        lines.append("")
        lines.append("| Report | ΔEM | ΔF1 | Triggered | Improve F1 | Hurt F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for report in group["run_summaries"]:
            lines.append(
                f"| {report['name']} | {report['delta_em']:+.4f} | {report['delta_f1']:+.4f} | "
                f"{int(report['structure_triggered_count'])} | "
                f"{int(report['improve_f1_count'])} | "
                f"{int(report['hurt_f1_count'])} |"
            )
        lines.append("")

    if summary["comparisons"]:
        lines.append("## Comparisons")
        lines.append("")
        lines.append("| A | B | Δ(run mean EM) | Δ(run mean F1) | Bootstrap ΔEM CI | P(ΔEM>0) | Bootstrap ΔF1 CI | P(ΔF1>0) |")
        lines.append("|---|---|---:|---:|---|---:|---|---:|")
        for comparison in summary["comparisons"]:
            lines.append(
                f"| {comparison['group_a']} | {comparison['group_b']} | "
                f"{comparison['delta_run_mean_em']:+.4f} | "
                f"{comparison['delta_run_mean_f1']:+.4f} | "
                f"[{comparison['delta_em_bootstrap']['ci_low']:+.4f}, {comparison['delta_em_bootstrap']['ci_high']:+.4f}] | "
                f"{comparison['delta_em_bootstrap']['p_gt_zero']:.3f} | "
                f"[{comparison['delta_f1_bootstrap']['ci_low']:+.4f}, {comparison['delta_f1_bootstrap']['ci_high']:+.4f}] | "
                f"{comparison['delta_f1_bootstrap']['p_gt_zero']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _pairwise_groups(groups: Iterable[dict],
                     num_bootstrap_samples: int,
                     seed: int,
                     ci: float) -> list[dict]:
    groups = list(groups)
    comparisons: list[dict] = []
    for index, group_a in enumerate(groups):
        for group_b in groups[index + 1:]:
            comparisons.append({
                "group_a": group_a["group"],
                "group_b": group_b["group"],
                "delta_run_mean_em": group_a["delta_em"]["run_mean"] - group_b["delta_em"]["run_mean"],
                "delta_run_mean_f1": group_a["delta_f1"]["run_mean"] - group_b["delta_f1"]["run_mean"],
                "delta_em_bootstrap": _bootstrap_mean_difference(
                    values_a=group_a["pooled_query_delta_em"],
                    values_b=group_b["pooled_query_delta_em"],
                    num_samples=num_bootstrap_samples,
                    seed=seed + 101,
                    ci=ci,
                ),
                "delta_f1_bootstrap": _bootstrap_mean_difference(
                    values_a=group_a["pooled_query_delta_f1"],
                    values_b=group_b["pooled_query_delta_f1"],
                    num_samples=num_bootstrap_samples,
                    seed=seed + 211,
                    ci=ci,
                ),
            })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple struct_compare JSON reports and summarize delta stability."
    )
    parser.add_argument("--inputs", nargs="*", default=[], help="Explicit report JSON files.")
    parser.add_argument("--glob", dest="glob_patterns", nargs="*", default=[], help="Glob patterns for report JSON files.")
    parser.add_argument("--group_regex", type=str, default=None, help="Regex used to derive group labels from filenames.")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    args = parser.parse_args()

    report_paths = _collect_paths(args.inputs, args.glob_patterns)
    if not report_paths:
        raise SystemExit("No report files found. Provide --inputs and/or --glob.")

    reports = []
    grouped_reports: dict[str, list[dict]] = defaultdict(list)
    for path in report_paths:
        report = _load_report(path)
        group_name = _infer_group(report["name"], args.group_regex)
        report["group"] = group_name
        reports.append(report)
        grouped_reports[group_name].append(report)

    groups = []
    for group_name, group_reports in sorted(grouped_reports.items()):
        group_summary = _summarize_group(
            group_name=group_name,
            runs=group_reports,
            num_bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            ci=args.ci,
        )
        groups.append(group_summary)

    summary = {
        "num_reports": len(reports),
        "num_bootstrap_samples": args.bootstrap_samples,
        "ci": args.ci,
        "groups": groups,
        "comparisons": _pairwise_groups(
            groups,
            num_bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            ci=args.ci,
        ),
    }

    for group in summary["groups"]:
        group.pop("pooled_query_delta_em", None)
        group.pop("pooled_query_delta_f1", None)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.write_text(_build_markdown(summary))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
