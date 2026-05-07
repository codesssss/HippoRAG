#!/usr/bin/env python3
"""Build subset datasets and external pools for counterfactual fill tests.

The generated pools evaluate whether SetR-faithful failures are caused by
missing support documents:

* rank_fill5: SetR selected docs + rank fill to 5.
* oracle_fill5: SetR selected docs + SetR-missing gold supports + rank fill to 5.
* dbec_repair_fill5: SetR selected docs + DBEC-selected non-SetR docs + rank fill to 5.

This script only materializes datasets/pools. It does not call the reader.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


RUN_DIR = Path("run_logs/counterfactual_fill_mechanism_20260507")
REPORT_DIR = Path("reports/counterfactual_fill_mechanism_20260507")
SUPPORT_ROWS = Path("reports/support_repair_mechanism_20260507/support_repair_rows.csv")
SOURCE_POOL_DIR = Path("run_logs/proprag_pool_exports_full1000_20260424")
SETR_FAITHFUL_DIR = Path("run_logs/setr_faithful_proprag_full1000_20260507")
SAVE_DIR_ROOT = "outputs_step0_general_nvembed"
TOP_K = 5

DATASETS: tuple[dict[str, Any], ...] = (
    {
        "label": "2Wiki",
        "dataset": "2wikimultihopqa",
        "subset_dataset": "2wikimultihopqa_dbec_cf_underselect_20260507",
        "slice": "gold_doc_count>=4 and SetR count-underselected",
        "port": 8041,
        "predicate": lambda row: (
            row["dataset"] == "2Wiki"
            and int(row["gold_doc_count"]) >= 4
            and int(row["count_under_selected"]) == 1
        ),
    },
    {
        "label": "MuSiQue",
        "dataset": "musique",
        "subset_dataset": "musique_dbec_cf_underselect_20260507",
        "slice": "gold_doc_count>=3 and SetR count-underselected",
        "port": 8043,
        "predicate": lambda row: (
            row["dataset"] == "MuSiQue"
            and int(row["gold_doc_count"]) >= 3
            and int(row["count_under_selected"]) == 1
        ),
    },
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_support_rows() -> list[dict[str, Any]]:
    with SUPPORT_ROWS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_title(value: Any) -> str:
    import re

    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def list_from_json(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def source_pool_path(dataset: str) -> Path:
    return SOURCE_POOL_DIR / f"{dataset}_pool100.json"


def setr_pool_path(dataset: str) -> Path:
    return SETR_FAITHFUL_DIR / f"{dataset}_proprag_setr_k20_doc768_faithful.selected_pool.json"


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


def unique_order(values: Sequence[int], *, limit: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        index = int(value)
        if index < 0 or index in seen:
            continue
        seen.add(index)
        output.append(index)
        if len(output) >= int(limit):
            break
    return output


def title_position_map(pool_titles: Sequence[Any]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for index, title in enumerate(pool_titles):
        key = normalize_title(title)
        if key:
            positions.setdefault(key, []).append(index)
    return positions


def positions_for_titles(
    titles: Sequence[Any],
    positions_by_title: Mapping[str, list[int]],
    *,
    already_used: set[int] | None = None,
) -> list[int]:
    used = set(already_used or set())
    output: list[int] = []
    for title in titles:
        key = normalize_title(title)
        if not key:
            continue
        for pos in positions_by_title.get(key, []):
            if int(pos) not in used:
                used.add(int(pos))
                output.append(int(pos))
                break
    return output


def build_order(
    variant: str,
    row: Mapping[str, Any],
    source_record: Mapping[str, Any],
    setr_record: Mapping[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    pool_titles = list(source_record.get("pool_titles") or [])
    positions_by_title = title_position_map(pool_titles)
    setr_trace = dict(setr_record.get("setr_selection_trace") or {})
    setr_positions = [
        int(pos)
        for pos in list(setr_trace.get("selected_positions") or [])
        if 0 <= int(pos) < len(pool_titles)
    ]
    setr_positions = unique_order(setr_positions, limit=TOP_K)

    used = set(setr_positions)
    if variant == "rank_fill5":
        add_titles = []
    elif variant == "oracle_fill5":
        add_titles = list_from_json(row.get("setr_missing_titles_json"))
    elif variant == "dbec_repair_fill5":
        add_titles = list_from_json(row.get("dbec_titles_json"))
    else:
        raise ValueError(f"unknown variant: {variant}")
    added_positions = positions_for_titles(add_titles, positions_by_title, already_used=used)
    used.update(added_positions)
    rank_fill_positions = [pos for pos in range(len(pool_titles)) if pos not in used]
    order = unique_order(setr_positions + added_positions + rank_fill_positions, limit=TOP_K)
    selected_set = set(setr_positions)
    actual_added_positions = [pos for pos in order if pos not in selected_set]
    trace = {
        "variant": variant,
        "source_query_idx": int(row["query_index"]),
        "setr_selected_positions": setr_positions,
        "requested_add_titles": [str(title) for title in add_titles],
        "candidate_added_positions": added_positions,
        "actual_added_positions": actual_added_positions,
        "rank_fill_positions": [pos for pos in order if pos not in set(setr_positions + added_positions)],
        "final_order_positions": order,
        "final_titles": [pool_titles[pos] for pos in order],
    }
    return order, trace


def reorder_record(
    source_record: Mapping[str, Any],
    *,
    order: Sequence[int],
    subset_query_idx: int,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(source_record)
    output["query_idx"] = int(subset_query_idx)
    for key in ("pool_docs", "pool_titles", "pool_doc_scores", "pool_doc_ids"):
        values = list(source_record.get(key) or [])
        output[key] = [values[pos] for pos in order if 0 <= int(pos) < len(values)]
    output["counterfactual_fill_trace"] = dict(trace)
    return output


def materialize_subset_dataset(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_dataset = str(config["dataset"])
    subset_dataset = str(config["subset_dataset"])
    base_samples = read_json(dataset_path(base_dataset))
    selected_indices = [int(row["query_index"]) for row in rows]
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


def build_variant_pool(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    base_dataset = str(config["dataset"])
    subset_dataset = str(config["subset_dataset"])
    source_payload = read_json(source_pool_path(base_dataset))
    setr_payload = read_json(setr_pool_path(base_dataset))
    source_records = list(source_payload.get("records") or [])
    setr_records = list(setr_payload.get("records") or [])
    output_records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for subset_query_idx, row in enumerate(rows):
        source_idx = int(row["query_index"])
        source_record = source_records[source_idx]
        setr_record = setr_records[source_idx]
        order, trace = build_order(variant, row, source_record, setr_record)
        output_records.append(
            reorder_record(
                source_record,
                order=order,
                subset_query_idx=subset_query_idx,
                trace=trace,
            )
        )
        trace_rows.append(dict(trace))

    output_payload = {
        "dataset": subset_dataset,
        "base_dataset": base_dataset,
        "subset_slice": str(config["slice"]),
        "limit": len(output_records),
        "pool_k": TOP_K,
        "source": f"counterfactual_{variant}",
        "source_pool_json": str(source_pool_path(base_dataset)),
        "setr_faithful_pool_json": str(setr_pool_path(base_dataset)),
        "records": output_records,
        "counterfactual_fill": {
            "variant": variant,
            "top_k": TOP_K,
            "records": len(output_records),
            "avg_original_setr_count": (
                sum(len(row["setr_selected_positions"]) for row in trace_rows) / max(1, len(trace_rows))
            ),
            "avg_actual_added_count": (
                sum(len(row["actual_added_positions"]) for row in trace_rows) / max(1, len(trace_rows))
            ),
        },
    }
    pool_path = RUN_DIR / f"{subset_dataset}_{variant}.pool.json"
    write_json(output_payload, pool_path)
    return {
        "variant": variant,
        "pool_json": str(pool_path),
        "records": len(output_records),
        "avg_original_setr_count": output_payload["counterfactual_fill"]["avg_original_setr_count"],
        "avg_actual_added_count": output_payload["counterfactual_fill"]["avg_actual_added_count"],
    }


def write_launcher(manifest: Mapping[str, Any]) -> Path:
    launcher = RUN_DIR / "launch_counterfactual_fill_mechanism_20260507.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        "",
        'ROOT_DIR="/mnt/nvme/code/HippoRAG"',
        'PYTHON_BIN="${ROOT_DIR}/.venv-hipporag/bin/python"',
        f'RUN_DIR="${{ROOT_DIR}}/{RUN_DIR}"',
        f'REPORT_DIR="${{ROOT_DIR}}/{REPORT_DIR}"',
        f'SAVE_DIR="{SAVE_DIR_ROOT}"',
        'LLM_MODEL="qwen3-8b-train"',
        'EMBEDDING_NAME="VLLM/nvidia/NV-Embed-v2"',
        'EMBEDDING_BASE_URL="http://localhost:8019/v1/embeddings"',
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
        '  local port="$2"',
        '  local variant="$3"',
        '  local pool_json="$4"',
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
        '      --external_pool_source_name "counterfactual_${variant}" \\',
        '      --external_pool_strict_questions true \\',
        '      --setwise_selector none \\',
        '      --output_json "${output_json}"',
        "}",
        "",
        "main() {",
        '  echo "[START] counterfactual_fill_mechanism $(date -Is)" > "${RUN_DIR}/launcher.status"',
        "  local rc=0",
    ]
    for dataset_entry in manifest["datasets"]:
        dataset = dataset_entry["subset_dataset"]
        port = dataset_entry["port"]
        for variant_entry in dataset_entry["variants"]:
            lines.append(
                f'  run_eval "{dataset}" "{port}" "{variant_entry["variant"]}" "{variant_entry["pool_json"]}" || rc=1'
            )
    lines.extend([
        '  if [[ "${rc}" -eq 0 ]]; then',
        '    echo "[DONE] counterfactual_fill_mechanism $(date -Is)" > "${RUN_DIR}/launcher.status"',
        "  else",
        '    echo "[FAILED] counterfactual_fill_mechanism rc=${rc} $(date -Is)" > "${RUN_DIR}/launcher.status"',
        "  fi",
        '  return "${rc}"',
        "}",
        "",
        'main "$@"',
        "",
    ])
    launcher.write_text("\n".join(lines), encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    support_rows = read_support_rows()
    manifest: dict[str, Any] = {
        "run_dir": str(RUN_DIR),
        "report_dir": str(REPORT_DIR),
        "top_k": TOP_K,
        "datasets": [],
    }
    for config in DATASETS:
        selected_rows = [row for row in support_rows if config["predicate"](row)]
        selected_rows = sorted(selected_rows, key=lambda row: int(row["query_index"]))
        subset_info = materialize_subset_dataset(config, selected_rows)
        variant_infos = [
            build_variant_pool(config, selected_rows, variant="rank_fill5"),
            build_variant_pool(config, selected_rows, variant="oracle_fill5"),
            build_variant_pool(config, selected_rows, variant="dbec_repair_fill5"),
        ]
        manifest["datasets"].append({
            "label": config["label"],
            "dataset": config["dataset"],
            "subset_dataset": config["subset_dataset"],
            "slice": config["slice"],
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
