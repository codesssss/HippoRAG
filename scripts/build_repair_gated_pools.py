#!/usr/bin/env python3
"""Materialize repair-gated DBEC pools for reader-only evaluation.

This consumes the offline arbitration report and writes external-pool JSONs
for a small set of selected repair-gated variants.  It does not call the
reader; the generated launcher runs ``eval_causal_qwen3.py`` with
``--setwise_selector none``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    SOURCE_POOL_DIR,
    TOP_K,
    list_from_json,
    read_json,
    safe_float,
    safe_int,
    source_pool_path,
)


OFFLINE_REPORT_DIR = Path("reports/repair_gated_arbitration_offline_20260507")
RUN_DIR = Path("run_logs/repair_gated_arbitration_20260507")
READER_REPORT_DIR = Path("reports/repair_gated_arbitration_reader_20260507")
SAVE_DIR_ROOT = "outputs_step0_general_nvembed"

DATASETS: tuple[dict[str, Any], ...] = (
    {
        "label": "2Wiki",
        "dataset": "2wikimultihopqa",
        "subset_dataset": "2wikimultihopqa_repair_gated_underselect_20260507",
        "port": 8041,
    },
    {
        "label": "HotpotQA",
        "dataset": "hotpotqa",
        "subset_dataset": "hotpotqa_repair_gated_underselect_20260507",
        "port": 8042,
    },
    {
        "label": "MuSiQue",
        "dataset": "musique",
        "subset_dataset": "musique_repair_gated_underselect_20260507",
        "port": 8043,
    },
)

POOL_LIST_KEYS = ("pool_docs", "pool_titles", "pool_doc_scores", "pool_doc_ids")


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataset_path(dataset: str) -> Path:
    return Path("reproduce/dataset") / f"{dataset}.json"


def corpus_path(dataset: str) -> Path:
    return Path("reproduce/dataset") / f"{dataset}_corpus.json"


def resolve_save_dir(save_dir_root: str, dataset: str) -> Path:
    if save_dir_root == "outputs":
        return Path(save_dir_root) / dataset
    return Path(f"{save_dir_root}_{dataset}")


def ensure_corpus_symlink(base_dataset: str, subset_dataset: str) -> None:
    target = corpus_path(base_dataset).resolve()
    link = corpus_path(subset_dataset)
    if link.is_symlink():
        if link.resolve() != target:
            link.unlink()
            link.symlink_to(target)
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink corpus path: {link}")
    link.symlink_to(target)


def ensure_save_dir_symlink(base_dataset: str, subset_dataset: str) -> None:
    target = resolve_save_dir(SAVE_DIR_ROOT, base_dataset).resolve()
    link = resolve_save_dir(SAVE_DIR_ROOT, subset_dataset)
    if link.is_symlink():
        if link.resolve() != target:
            link.unlink()
            link.symlink_to(target)
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink save_dir path: {link}")
    link.symlink_to(target)


def variant_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, int]:
    return (
        safe_float(row.get("delta_support_complete_vs_rank")),
        safe_float(row.get("delta_support_recall_vs_rank")),
        -safe_float(row.get("harmful_replacement_rate")),
        -safe_float(row.get("accepted_repair_count")),
        safe_float(row.get("min_gain")),
        -safe_int(row.get("rank_quota")),
    )


def best_for_dataset(summary_rows: Sequence[Mapping[str, Any]], dataset: str) -> Mapping[str, Any] | None:
    candidates = [row for row in summary_rows if str(row.get("dataset")) == dataset]
    if not candidates:
        return None
    return sorted(candidates, key=variant_sort_key, reverse=True)[0]


def cross_dataset_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for row in summary_rows:
        by_variant.setdefault(str(row.get("variant")), []).append(row)
    output: list[dict[str, Any]] = []
    for variant, rows in by_variant.items():
        if not rows:
            continue
        first = rows[0]
        output.append({
            "dataset": "ALL",
            "variant": variant,
            "n_dataset": len(rows),
            "rank_quota": safe_int(first.get("rank_quota")),
            "max_repairs": safe_int(first.get("max_repairs")),
            "min_gain": safe_float(first.get("min_gain")),
            "branch_policy": str(first.get("branch_policy") or ""),
            "delta_support_complete_vs_rank": sum(safe_float(row.get("delta_support_complete_vs_rank")) for row in rows) / len(rows),
            "delta_support_recall_vs_rank": sum(safe_float(row.get("delta_support_recall_vs_rank")) for row in rows) / len(rows),
            "harmful_replacement_rate": sum(safe_float(row.get("harmful_replacement_rate")) for row in rows) / len(rows),
            "accepted_repair_count": sum(safe_float(row.get("accepted_repair_count")) for row in rows) / len(rows),
        })
    return output


def select_variants(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    requested_variants: Sequence[str] | None = None,
    max_variants: int = 3,
) -> list[dict[str, Any]]:
    if requested_variants:
        by_variant = {str(row.get("variant")): row for row in summary_rows}
        selected: list[dict[str, Any]] = []
        for variant in requested_variants:
            if variant not in by_variant:
                raise ValueError(f"Requested variant not found in offline summary: {variant}")
            selected.append({"reason": "requested", **dict(by_variant[variant])})
        return selected[: int(max_variants)]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for dataset, reason in (("MuSiQue", "best_musique"), ("2Wiki", "best_2wiki")):
        row = best_for_dataset(summary_rows, dataset)
        if row and str(row.get("variant")) not in seen:
            selected.append({"reason": reason, **dict(row)})
            seen.add(str(row.get("variant")))

    cross_rows = sorted(cross_dataset_rows(summary_rows), key=variant_sort_key, reverse=True)
    for row in cross_rows:
        if len(selected) >= int(max_variants):
            break
        variant = str(row.get("variant"))
        if variant in seen:
            continue
        selected.append({"reason": "best_cross_dataset_average", **dict(row)})
        seen.add(variant)

    return selected[: int(max_variants)]


def materialize_subset_dataset(config: Mapping[str, Any], audit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_dataset = str(config["dataset"])
    subset_dataset = str(config["subset_dataset"])
    base_samples = read_json(dataset_path(base_dataset))
    selected_indices = sorted({safe_int(row.get("query_index")) for row in audit_rows})
    subset_samples = [base_samples[index] for index in selected_indices]
    write_json(subset_samples, dataset_path(subset_dataset))
    ensure_corpus_symlink(base_dataset, subset_dataset)
    ensure_save_dir_symlink(base_dataset, subset_dataset)
    return {
        "subset_dataset": subset_dataset,
        "subset_sample_count": len(subset_samples),
        "selected_indices": selected_indices,
        "subset_path": str(dataset_path(subset_dataset)),
        "subset_corpus_path": str(corpus_path(subset_dataset)),
        "subset_save_dir": str(resolve_save_dir(SAVE_DIR_ROOT, subset_dataset)),
    }


def reorder_record(
    source_record: Mapping[str, Any],
    *,
    order: Sequence[int],
    subset_query_idx: int,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(source_record)
    output["query_idx"] = int(subset_query_idx)
    for key in POOL_LIST_KEYS:
        values = list(source_record.get(key) or [])
        output[key] = [values[pos] for pos in order if 0 <= int(pos) < len(values)]
    output["repair_gated_trace"] = dict(trace)
    return output


def build_trace_from_audit_row(row: Mapping[str, Any], order: Sequence[int], source_record: Mapping[str, Any]) -> dict[str, Any]:
    pool_titles = list(source_record.get("pool_titles") or [])
    return {
        "source_query_idx": safe_int(row.get("query_index")),
        "variant": str(row.get("variant") or ""),
        "rank_quota": safe_int(row.get("rank_quota")),
        "max_repairs": safe_int(row.get("max_repairs")),
        "min_gain": safe_float(row.get("min_gain")),
        "branch_policy": str(row.get("branch_policy") or ""),
        "seed_positions": list_from_json(row.get("seed_positions_json")),
        "final_order_positions": [int(pos) for pos in order],
        "final_titles": [str(pool_titles[pos]) for pos in order],
        "accepted_repair_positions": list_from_json(row.get("accepted_repair_positions_json")),
        "accepted_repair_titles": list_from_json(row.get("accepted_repair_titles_json")),
        "accepted_repair_gains": list_from_json(row.get("accepted_repair_gains_json")),
        "accepted_repair_branch_status": list_from_json(row.get("accepted_repair_branch_status_json")),
        "rank_fill_positions": list_from_json(row.get("rank_fill_positions_json")),
        "offline_support": {
            "final_support_recall": safe_float(row.get("final_support_recall")),
            "final_support_complete": safe_int(row.get("final_support_complete")),
            "rank_support_recall": safe_float(row.get("rank_support_recall")),
            "rank_support_complete": safe_int(row.get("rank_support_complete")),
            "dbec_support_recall": safe_float(row.get("dbec_support_recall")),
            "dbec_support_complete": safe_int(row.get("dbec_support_complete")),
            "net_gold_delta_vs_rank": safe_int(row.get("net_gold_delta_vs_rank")),
            "harmful_replacement": safe_int(row.get("harmful_replacement")),
        },
    }


def build_variant_pool(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    base_dataset = str(config["dataset"])
    subset_dataset = str(config["subset_dataset"])
    source_payload = read_json(source_pool_path(base_dataset))
    source_records = list(source_payload.get("records") or [])
    variant_rows = [row for row in rows if str(row.get("variant")) == variant]
    variant_rows = sorted(variant_rows, key=lambda row: safe_int(row.get("query_index")))
    selected_indices = [safe_int(row.get("query_index")) for row in variant_rows]
    subset_index_by_source = {source_idx: subset_idx for subset_idx, source_idx in enumerate(selected_indices)}
    output_records: list[dict[str, Any]] = []
    for row in variant_rows:
        source_idx = safe_int(row.get("query_index"))
        source_record = source_records[source_idx]
        order = [int(pos) for pos in list_from_json(row.get("final_positions_json"))][:TOP_K]
        trace = build_trace_from_audit_row(row, order, source_record)
        output_records.append(
            reorder_record(
                source_record,
                order=order,
                subset_query_idx=subset_index_by_source[source_idx],
                trace=trace,
            )
        )

    output_payload = {
        "dataset": subset_dataset,
        "base_dataset": base_dataset,
        "limit": len(output_records),
        "pool_k": TOP_K,
        "source": f"repair_gated_{variant}",
        "source_pool_json": str(source_pool_path(base_dataset)),
        "offline_report_dir": str(OFFLINE_REPORT_DIR),
        "records": output_records,
        "repair_gated_arbitration": {
            "variant": variant,
            "top_k": TOP_K,
            "records": len(output_records),
            "avg_accepted_repair_count": (
                sum(safe_float(row.get("accepted_repair_count")) for row in variant_rows) / max(1, len(variant_rows))
            ),
            "avg_rank_fill_count": (
                sum(safe_float(row.get("rank_fill_count")) for row in variant_rows) / max(1, len(variant_rows))
            ),
            "avg_offline_delta_support_complete_vs_rank": (
                sum(safe_float(row.get("delta_support_complete_vs_rank")) for row in variant_rows) / max(1, len(variant_rows))
            ),
        },
    }
    pool_path = RUN_DIR / f"{subset_dataset}_{variant}.pool.json"
    write_json(output_payload, pool_path)
    return {
        "variant": variant,
        "pool_json": str(pool_path),
        "records": len(output_records),
        **output_payload["repair_gated_arbitration"],
    }


def write_launcher(manifest: Mapping[str, Any]) -> Path:
    launcher = RUN_DIR / "launch_repair_gated_reader_eval_20260507.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        "",
        'ROOT_DIR="/mnt/nvme/code/HippoRAG"',
        'PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"',
        f'RUN_DIR="${{ROOT_DIR}}/{RUN_DIR}"',
        f'REPORT_DIR="${{ROOT_DIR}}/{READER_REPORT_DIR}"',
        f'SAVE_DIR="{SAVE_DIR_ROOT}"',
        'LLM_MODEL="${REPAIR_GATED_LLM_MODEL:-qwen3-8b-train}"',
        'EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"',
        'EMBEDDING_BASE_URL="${REPAIR_GATED_EMBEDDING_BASE_URL:-http://localhost:8019/v1/embeddings}"',
        'PORTS_CSV="${REPAIR_GATED_QWEN_PORTS:-8041,8042,8043}"',
        'MAX_PARALLEL="${REPAIR_GATED_MAX_PARALLEL:-3}"',
        'IFS="," read -r -a PORTS <<< "${PORTS_CSV}"',
        "",
        'mkdir -p "${RUN_DIR}" "${REPORT_DIR}"',
        "",
        "run_cmd() {",
        '  local label="$1"',
        "  shift",
        '  local log_path="${RUN_DIR}/${label}.log"',
        '  local status_path="${RUN_DIR}/${label}.status"',
        '  if [[ -f "${status_path}" ]] && grep -q "^DONE " "${status_path}"; then',
        '    echo "[SKIP] ${label}"',
        "    return 0",
        "  fi",
        '  echo "START ${label} $(date -Is)" > "${status_path}"',
        '  echo "[START] ${label}"',
        '  (cd "${ROOT_DIR}" && "$@") > "${log_path}" 2>&1',
        "  local code=$?",
        '  if [[ "${code}" -eq 0 ]]; then',
        '    echo "DONE ${label} $(date -Is)" > "${status_path}"',
        '    echo "[DONE] ${label}"',
        "  else",
        '    echo "FAILED ${label} code=${code} $(date -Is)" > "${status_path}"',
        '    echo "[FAILED] ${label} code=${code}"',
        '    return "${code}"',
        "  fi",
        "}",
        "",
        "run_eval() {",
        '  local dataset="$1"',
        '  local variant="$2"',
        '  local pool_json="$3"',
        '  local port="$4"',
        '  local output_json="${REPORT_DIR}/${dataset}_${variant}.eval.json"',
        '  run_cmd "eval_${dataset}_${variant}" \\',
        '    "${PYTHON_BIN}" scripts/eval_causal_qwen3.py \\',
        '      --dataset "${dataset}" \\',
        '      --limit 0 \\',
        '      --save_dir "${SAVE_DIR}" \\',
        '      --llm_name qwen3-8b \\',
        '      --llm_request_name "${LLM_MODEL}" \\',
        '      --max_retry_attempts 20 \\',
        '      --llm_base_url "http://localhost:${port}/v1" \\',
        '      --embedding_name "${EMBEDDING_NAME}" \\',
        '      --embedding_base_url "${EMBEDDING_BASE_URL}" \\',
        '      --qa_top_k 5 \\',
        '      --qa_doc_max_chars 2048 \\',
        '      --external_pool_json "${pool_json}" \\',
        '      --external_pool_source_name "repair_gated_${variant}" \\',
        '      --external_pool_strict_questions true \\',
        '      --setwise_selector none \\',
        '      --output_json "${output_json}"',
        "}",
        "",
        "throttle() {",
        '  while [[ "$(jobs -pr | wc -l)" -ge "${MAX_PARALLEL}" ]]; do',
        "    wait -n || RC=1",
        "  done",
        "}",
        "",
        "main() {",
        '  echo "[START] repair_gated_reader_eval $(date -Is)" > "${RUN_DIR}/launcher.status"',
        "  RC=0",
        "  local job_index=0",
    ]
    for dataset_entry in manifest.get("datasets", []) or []:
        dataset = dataset_entry["subset_dataset"]
        variant_entries = list(dataset_entry.get("variants", []) or [])
        if not variant_entries:
            continue
        lines.extend([
            "  throttle",
            '  local port="${PORTS[$((job_index % ${#PORTS[@]}))]}"',
            "  (",
            "    group_rc=0",
        ])
        for variant_entry in variant_entries:
            variant = variant_entry["variant"]
            pool_json = variant_entry["pool_json"]
            lines.append(
                f'    run_eval "{dataset}" "{variant}" "{pool_json}" "${{port}}" || group_rc=1'
            )
        lines.extend([
            '    exit "${group_rc}"',
            "  ) &",
            "  job_index=$((job_index + 1))",
        ])
    lines.extend([
        '  while [[ "$(jobs -pr | wc -l)" -gt 0 ]]; do',
        "    wait -n || RC=1",
        "  done",
        '  if [[ "${RC}" -eq 0 ]]; then',
        '    echo "[DONE] repair_gated_reader_eval $(date -Is)" > "${RUN_DIR}/launcher.status"',
        "  else",
        '    echo "[FAILED] repair_gated_reader_eval rc=${RC} $(date -Is)" > "${RUN_DIR}/launcher.status"',
        "  fi",
        '  return "${RC}"',
        "}",
        "",
        'main "$@"',
        "",
    ])
    launcher.write_text("\n".join(lines), encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def main() -> None:
    global OFFLINE_REPORT_DIR, RUN_DIR, READER_REPORT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline_report_dir", type=Path, default=OFFLINE_REPORT_DIR)
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--reader_report_dir", type=Path, default=READER_REPORT_DIR)
    parser.add_argument("--max_variants", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    OFFLINE_REPORT_DIR = Path(args.offline_report_dir)
    RUN_DIR = Path(args.run_dir)
    READER_REPORT_DIR = Path(args.reader_report_dir)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    READER_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    audit_rows = read_csv_rows(OFFLINE_REPORT_DIR / "replacement_audit.csv")
    summary_rows = read_csv_rows(OFFLINE_REPORT_DIR / "variant_support_summary.csv")
    selected_variants = select_variants(
        summary_rows,
        requested_variants=args.variants,
        max_variants=int(args.max_variants),
    )
    selected_variant_names = [str(row["variant"]) for row in selected_variants]
    manifest: dict[str, Any] = {
        "run_dir": str(RUN_DIR),
        "reader_report_dir": str(READER_REPORT_DIR),
        "offline_report_dir": str(OFFLINE_REPORT_DIR),
        "source_pool_dir": str(SOURCE_POOL_DIR),
        "top_k": TOP_K,
        "selected_variants": selected_variants,
        "datasets": [],
    }

    for config in DATASETS:
        dataset_rows = [row for row in audit_rows if str(row.get("dataset")) == str(config["label"])]
        if not dataset_rows:
            continue
        subset_info = materialize_subset_dataset(config, dataset_rows)
        variant_infos = [
            build_variant_pool(config, dataset_rows, variant=variant)
            for variant in selected_variant_names
            if any(str(row.get("variant")) == variant for row in dataset_rows)
        ]
        manifest["datasets"].append({
            "label": config["label"],
            "dataset": config["dataset"],
            "subset_dataset": config["subset_dataset"],
            "port": config["port"],
            **subset_info,
            "variants": variant_infos,
        })

    launcher = write_launcher(manifest)
    manifest["launcher"] = str(launcher)
    write_json(manifest, RUN_DIR / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
