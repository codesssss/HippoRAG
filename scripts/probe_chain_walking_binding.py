#!/usr/bin/env python3
"""Probe whether upstream-context binding can surface MuSiQue missing support.

This is a fail-fast diagnostic for the chain-walking binding hypothesis.  It
does not change DBEC selection or call the reader.  On the MuSiQue
``gold_source_only_not_rank_or_dbec`` slice, it asks:

* can a local LLM read an upstream/root document and extract latent entity or
  page-title anchors for each dependent demand?
* if we rerank the fixed pool100 with those anchors, do the missing gold support
  titles move from deep source ranks into top-5/top-10?

The target slice and baseline ranks come from
``reports/repair_candidate_quality_audit_20260507``.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    DBEC_SELECTIVE_DIR,
    SETR_FAITHFUL_DIR,
    SOURCE_POOL_DIR,
    TOP_K,
    extract_dbec_final_order,
    extract_setr_seed_positions,
    list_from_json,
    normalize_title,
    rank_fill_order,
    read_json,
    safe_float,
    safe_int,
)


REPORT_DIR = Path("reports/chain_walking_binding_probe_20260507")
CANDIDATE_AUDIT_DIR = Path("reports/repair_candidate_quality_audit_20260507")
MUSIQUE_DATASET = "musique"
MUSIQUE_LABEL = "MuSiQue"
TARGET_BUCKET = "gold_source_only_not_rank_or_dbec"
PROMPT_VERSION = "chain_walking_binding_probe_v1"
DEFAULT_LLM_BASE_URL = "http://localhost:8043/v1"
DEFAULT_LLM_MODEL = "qwen3-8b-train"
DEFAULT_MAX_DOC_CHARS = 800
DEFAULT_MAX_UPSTREAM_DOCS = 3
DEFAULT_SOURCE_TOP_N = 2

STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha1_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class JsonlCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.values: dict[str, Any] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.values[str(row["key"])] = row.get("value")

    def get(self, key: str) -> Any | None:
        return self.values.get(str(key))

    def set(self, key: str, value: Any, metadata: Mapping[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.values[str(key)] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"key": key, "value": value, "metadata": dict(metadata or {})}, ensure_ascii=False)
                + "\n"
            )


def strip_think_blocks(text: Any) -> str:
    raw = str(text or "")
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw)
    raw = raw.replace("```json", "```")
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return raw.strip()


def parse_json_payload(text: Any) -> Any | None:
    raw = strip_think_blocks(text)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start_candidates = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(raw), start, -1):
        snippet = raw[start:end].strip()
        if not snippet:
            continue
        try:
            return json.loads(snippet)
        except Exception:
            continue
    return None


def normalize_match_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_match_text(value).split()
        if token and token not in STOP_TOKENS
    }


def clean_entity(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[\\s\\-•*\"']+", "", text)
    text = re.sub(r"[\\s\"']+$", "", text)
    text = re.sub(r"\\s+", " ", text)
    if not text or text.upper() in {"NONE", "N/A", "NULL"}:
        return ""
    if len(text) > 120:
        return ""
    return text


def entities_from_parsed(parsed: Any) -> list[dict[str, str]]:
    raw_entities: list[Any]
    if isinstance(parsed, Mapping):
        raw_entities = list(parsed.get("entities") or parsed.get("titles") or parsed.get("anchors") or [])
    elif isinstance(parsed, list):
        raw_entities = list(parsed)
    else:
        raw_entities = []

    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_entities:
        support = ""
        why = ""
        if isinstance(item, Mapping):
            text = (
                item.get("text")
                or item.get("entity")
                or item.get("title")
                or item.get("name")
                or ""
            )
            support = str(item.get("support_span") or item.get("span") or "")
            why = str(item.get("why") or item.get("rationale") or "")
        else:
            text = item
        entity = clean_entity(text)
        key = normalize_match_text(entity)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append({"entity": entity, "support_span": support, "why": why})
    return output[:3]


def entities_from_raw(text: Any) -> list[dict[str, str]]:
    parsed = parse_json_payload(text)
    if parsed is not None:
        return entities_from_parsed(parsed)
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in strip_think_blocks(text).splitlines():
        entity = clean_entity(line)
        key = normalize_match_text(entity)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append({"entity": entity, "support_span": "", "why": ""})
        if len(output) >= 3:
            break
    return output


def chat_completion_raw(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }
    request = urllib.request.Request(
        str(base_url).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str((payload.get("choices") or [{}])[0].get("message", {}).get("content") or "")


def probe_llm(base_url: str, model: str, timeout: int) -> dict[str, Any]:
    try:
        raw = chat_completion_raw(
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": "/no_think\nReturn exactly: OK"}],
            max_tokens=16,
            timeout=timeout,
        )
        stripped = strip_think_blocks(raw)
        return {"available": "ok" in stripped.lower() or "ok" in raw.lower(), "raw": raw, "stripped": stripped, "error": ""}
    except Exception as exc:
        return {"available": False, "raw": "", "stripped": "", "error": str(exc)}


def source_pool_path(dataset: str) -> Path:
    return SOURCE_POOL_DIR / f"{dataset}_pool100.json"


def setr_pool_path(dataset: str) -> Path:
    return SETR_FAITHFUL_DIR / f"{dataset}_proprag_setr_k20_doc768_faithful.selected_pool.json"


def dbec_report_path(dataset: str) -> Path:
    return DBEC_SELECTIVE_DIR / f"{dataset}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def load_artifacts(dataset: str = MUSIQUE_DATASET) -> dict[str, Any]:
    return {
        "source_records": list(read_json(source_pool_path(dataset)).get("records") or []),
        "setr_records": list(read_json(setr_pool_path(dataset)).get("records") or []),
        "dbec_traces": list(read_json(dbec_report_path(dataset)).get("setwise_selector_query_traces") or []),
    }


def target_missing_gold_rows(
    *,
    audit_dir: str | Path = CANDIDATE_AUDIT_DIR,
    dataset_label: str = MUSIQUE_LABEL,
    bucket: str = TARGET_BUCKET,
    query_primary_only: bool = False,
) -> list[dict[str, Any]]:
    rows = read_csv_rows(Path(audit_dir) / "missing_gold_audit.csv")
    allowed_query_indices: set[int] | None = None
    if query_primary_only:
        query_rows = read_csv_rows(Path(audit_dir) / "query_audit.csv")
        allowed_query_indices = {
            safe_int(row.get("query_index"))
            for row in query_rows
            if str(row.get("dataset")) == dataset_label and str(row.get("primary_bucket")) == bucket
        }
    output = [
        row
        for row in rows
        if str(row.get("dataset")) == dataset_label and str(row.get("bucket")) == bucket
        and (allowed_query_indices is None or safe_int(row.get("query_index")) in allowed_query_indices)
    ]
    return sorted(output, key=lambda row: (safe_int(row.get("query_index")), str(row.get("gold_title"))))


def grouped_missing_gold(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[safe_int(row.get("query_index"))].append(dict(row))
    return dict(grouped)


def selector_trace_from_query_trace(query_trace: Mapping[str, Any]) -> Mapping[str, Any]:
    selector_trace = query_trace.get("selector_trace")
    return selector_trace if isinstance(selector_trace, Mapping) else {}


def dependent_requirements(selector_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    requirements = [req for req in selector_trace.get("requirements") or [] if isinstance(req, Mapping)]
    return [dict(req) for req in requirements if req.get("depends_on")]


def requirement_by_id(selector_trace: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(req.get("unit_id")): dict(req)
        for req in selector_trace.get("requirements") or []
        if isinstance(req, Mapping) and req.get("unit_id")
    }


def append_position(output: list[int], position: Any, *, pool_size: int) -> None:
    pos = safe_int(position, default=-1)
    if 0 <= pos < pool_size and pos not in output:
        output.append(pos)


def positions_for_title(title: Any, pool_titles: Sequence[Any]) -> list[int]:
    norm = normalize_title(title)
    if not norm:
        return []
    return [
        index
        for index, candidate in enumerate(pool_titles)
        if normalize_title(candidate) == norm
    ]


def upstream_positions_for_requirement(
    *,
    requirement: Mapping[str, Any],
    selector_trace: Mapping[str, Any],
    setr_record: Mapping[str, Any],
    pool_titles: Sequence[Any],
    source_top_n: int = DEFAULT_SOURCE_TOP_N,
    max_upstream_docs: int = DEFAULT_MAX_UPSTREAM_DOCS,
) -> list[int]:
    """Choose existing, non-oracle upstream contexts for one dependent demand."""

    pool_size = len(pool_titles)
    positions: list[int] = []
    requirement_id = str(requirement.get("unit_id") or "")
    depends_on = [str(item) for item in requirement.get("depends_on") or []]

    candidates_by_requirement = selector_trace.get("binding_candidates_by_requirement")
    candidates_by_requirement = candidates_by_requirement if isinstance(candidates_by_requirement, Mapping) else {}

    # DBEC's existing binding trace often records which upstream document was
    # read to propose candidates for this dependent requirement.  This is the
    # cleanest context signal for the probe.
    for candidate in candidates_by_requirement.get(requirement_id) or []:
        if isinstance(candidate, Mapping):
            append_position(positions, candidate.get("dep_position"), pool_size=pool_size)

    # If the upstream requirement already had candidates or an assignment, use
    # the upstream result title as context before generic rank docs.
    selected_binding = selector_trace.get("selected_binding")
    assignments = selected_binding.get("assignments") if isinstance(selected_binding, Mapping) else {}
    assignments = assignments if isinstance(assignments, Mapping) else {}
    for dep_id in depends_on:
        for candidate in candidates_by_requirement.get(dep_id) or []:
            if isinstance(candidate, Mapping):
                append_position(positions, candidate.get("title_pool_position"), pool_size=pool_size)
        if dep_id in assignments:
            for pos in positions_for_title(assignments.get(dep_id), pool_titles):
                append_position(positions, pos, pool_size=pool_size)

    # Add source/rank/DBEC front contexts.  These are method-visible, not gold
    # oracle docs, and keep the probe close to the current pipeline.
    for pos in range(min(max(0, int(source_top_n)), pool_size)):
        append_position(positions, pos, pool_size=pool_size)

    dbec_positions = extract_dbec_final_order(selector_trace, pool_size)
    setr_positions = extract_setr_seed_positions(setr_record, pool_size)
    rank_positions = rank_fill_order(setr_positions, pool_size=pool_size, top_k=TOP_K)
    for pos in list(dbec_positions) + list(setr_positions) + list(rank_positions):
        append_position(positions, pos, pool_size=pool_size)

    return positions[: max(1, int(max_upstream_docs))]


def build_messages(
    *,
    question: str,
    requirement: Mapping[str, Any],
    upstream_requirements: Sequence[Mapping[str, Any]],
    upstream_title: str,
    upstream_doc: str,
    max_doc_chars: int,
) -> list[dict[str, str]]:
    upstream_lines = [
        f"- {req.get('unit_id')}: {req.get('subquery')}"
        for req in upstream_requirements
    ]
    prompt = (
        "/no_think\n"
        "You are testing chain-aware retrieval binding. Read the upstream document and extract up to 3 short entity or Wikipedia page-title strings that should be used as retrieval anchors for the dependent demand.\n"
        "Rules:\n"
        "- Prefer exact entities or page titles explicitly supported by the upstream document.\n"
        "- If the dependent demand requires a directly implied country, continent, organization, person, or work, include that short implied anchor only when the upstream text gives enough evidence.\n"
        "- Do not answer the original question unless that answer is the next retrieval anchor.\n"
        "- Do not invent entities unsupported by the upstream document.\n"
        "- Output strict JSON only: {\"entities\": [{\"text\": string, \"support_span\": string, \"why\": string}]}.\n\n"
        f"Original question: {question}\n\n"
        "Upstream demand(s):\n"
        + ("\n".join(upstream_lines) if upstream_lines else "- unknown")
        + "\n\n"
        f"Dependent demand: {requirement.get('unit_id')}: {requirement.get('subquery')}\n\n"
        f"Upstream doc title: {upstream_title}\n"
        f"Upstream doc content:\n{str(upstream_doc)[: int(max_doc_chars)]}\n"
    )
    return [
        {"role": "system", "content": "You extract grounded retrieval anchors and output JSON only."},
        {"role": "user", "content": prompt},
    ]


def doc_entity_score(entity: str, title: Any, doc_text: Any) -> tuple[float, str]:
    entity_norm = normalize_match_text(entity)
    title_norm = normalize_match_text(title)
    doc_norm = normalize_match_text(doc_text)
    if not entity_norm or len(entity_norm) < 2:
        return 0.0, ""

    score = 0.0
    reason = ""
    if entity_norm == title_norm:
        return 100.0, "title_exact"
    if len(entity_norm) >= 4 and (entity_norm in title_norm or title_norm in entity_norm):
        score = 85.0
        reason = "title_substring"
    elif len(entity_norm) >= 4 and entity_norm in doc_norm:
        score = 55.0
        reason = "body_substring"

    entity_tok = entity_tokens(entity_norm)
    title_tok = entity_tokens(title_norm)
    doc_tok = entity_tokens(doc_norm)
    if entity_tok:
        title_overlap = len(entity_tok & title_tok) / len(entity_tok)
        body_overlap = len(entity_tok & doc_tok) / len(entity_tok)
        if title_overlap > 0:
            title_score = 35.0 * title_overlap
            if title_score > score:
                score = title_score
                reason = "title_token_overlap"
        if body_overlap > 0:
            body_score = 18.0 * body_overlap
            if body_score > score:
                score = body_score
                reason = "body_token_overlap"
    return score, reason


def rank_pool_by_entities(
    *,
    pool_titles: Sequence[Any],
    pool_docs: Sequence[Any],
    entity_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    best_by_position: dict[int, dict[str, Any]] = {}
    for pos, title in enumerate(pool_titles):
        doc = pool_docs[pos] if pos < len(pool_docs) else title
        best: dict[str, Any] = {"score": 0.0, "entity": "", "reason": ""}
        for entity_row in entity_rows:
            entity = str(entity_row.get("entity") or "")
            score, reason = doc_entity_score(entity, title, doc)
            if score > safe_float(best.get("score")):
                best = {
                    "score": float(score),
                    "entity": entity,
                    "reason": reason,
                    "requirement_id": str(entity_row.get("requirement_id") or ""),
                    "requirement_subquery": str(entity_row.get("requirement_subquery") or ""),
                    "upstream_position": safe_int(entity_row.get("upstream_position"), default=-1),
                    "upstream_title": str(entity_row.get("upstream_title") or ""),
                    "support_span": str(entity_row.get("support_span") or ""),
                    "why": str(entity_row.get("why") or ""),
                }
        best_by_position[pos] = best
    order = sorted(range(len(pool_titles)), key=lambda pos: (-safe_float(best_by_position[pos].get("score")), pos))
    return order, best_by_position


def rank_of_title(title: Any, order: Sequence[int], pool_titles: Sequence[Any]) -> int | None:
    norm = normalize_title(title)
    if not norm:
        return None
    for rank, pos in enumerate(order):
        if 0 <= int(pos) < len(pool_titles) and normalize_title(pool_titles[int(pos)]) == norm:
            return rank
    return None


def bucket_for_rank(rank: int) -> str:
    if rank <= 20:
        return "source_rank_0_20"
    if rank <= 50:
        return "source_rank_21_50"
    return "source_rank_51_99"


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return sum(vals) / len(vals) if vals else 0.0


def safe_rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def summarize_probe_rows(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groups: dict[str, list[Mapping[str, Any]]] = {"overall": list(probe_rows)}
    for row in probe_rows:
        source_rank = safe_int(row.get("source_best_gold_rank"), default=999)
        groups.setdefault(bucket_for_rank(source_rank), []).append(row)

    for name, rows in groups.items():
        total = len(rows)
        row = {
            "bucket": name,
            "missing_gold_titles": total,
            "baseline_hit_at_3": sum(safe_int(r.get("baseline_hit_at_3")) for r in rows),
            "baseline_hit_at_5": sum(safe_int(r.get("baseline_hit_at_5")) for r in rows),
            "baseline_hit_at_10": sum(safe_int(r.get("baseline_hit_at_10")) for r in rows),
            "baseline_hit_at_20": sum(safe_int(r.get("baseline_hit_at_20")) for r in rows),
            "probe_hit_at_3": sum(safe_int(r.get("probe_hit_at_3")) for r in rows),
            "probe_hit_at_5": sum(safe_int(r.get("probe_hit_at_5")) for r in rows),
            "probe_hit_at_10": sum(safe_int(r.get("probe_hit_at_10")) for r in rows),
            "probe_hit_at_20": sum(safe_int(r.get("probe_hit_at_20")) for r in rows),
            "new_hit_at_3": sum(safe_int(r.get("new_hit_at_3")) for r in rows),
            "new_hit_at_5": sum(safe_int(r.get("new_hit_at_5")) for r in rows),
            "new_hit_at_10": sum(safe_int(r.get("new_hit_at_10")) for r in rows),
            "new_hit_at_20": sum(safe_int(r.get("new_hit_at_20")) for r in rows),
            "mean_source_rank": safe_mean(safe_int(r.get("source_best_gold_rank"), default=0) for r in rows),
            "mean_probe_rank": safe_mean(safe_int(r.get("best_probe_rank"), default=0) for r in rows),
            "mean_rank_improvement": safe_mean(safe_float(r.get("rank_improvement")) for r in rows),
            "positive_score_titles": sum(1 for r in rows if safe_float(r.get("best_probe_score")) > 0),
        }
        for k in (3, 5, 10, 20):
            row[f"probe_recall_at_{k}"] = safe_rate(safe_int(row[f"probe_hit_at_{k}"]), total)
            row[f"new_recall_at_{k}"] = safe_rate(safe_int(row[f"new_hit_at_{k}"]), total)
        row["positive_score_rate"] = safe_rate(safe_int(row["positive_score_titles"]), total)
        summaries.append(row)

    prompt_count = len(prompt_rows)
    parse_ok = sum(safe_int(row.get("parse_ok")) for row in prompt_rows)
    entity_count = sum(safe_int(row.get("entity_count")) for row in prompt_rows)
    overall = next((row for row in summaries if row["bucket"] == "overall"), {})
    new_recall_at_5 = safe_float(overall.get("new_recall_at_5"))
    if new_recall_at_5 >= 0.30:
        decision = "strong_go"
    elif new_recall_at_5 >= 0.15:
        decision = "mixed_inspect"
    elif new_recall_at_5 >= 0.05:
        decision = "weak_future_work"
    else:
        decision = "stop_chain_walking"
    metadata = {
        "missing_gold_titles": len(probe_rows),
        "queries": len({safe_int(row.get("query_index")) for row in probe_rows}),
        "prompt_rows": prompt_count,
        "parse_ok_rate": safe_rate(parse_ok, prompt_count),
        "entity_outputs": entity_count,
        "entities_per_prompt": safe_rate(entity_count, prompt_count),
        "decision": decision,
    }
    return summaries, metadata


def build_markdown(
    *,
    metadata: Mapping[str, Any],
    rank_shift_summary: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    report_dir: Path,
) -> str:
    overall = next((row for row in rank_shift_summary if row.get("bucket") == "overall"), {})
    lines = [
        "# Chain-Walking Binding Probe",
        "",
        "This diagnostic tests whether upstream-document entity extraction can move MuSiQue missing gold support titles from deep fixed-pool ranks into top-k. It does not call the reader and does not change DBEC selection.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Target queries | {metadata.get('queries', 0)} |",
        f"| Target missing gold titles | {metadata.get('missing_gold_titles', 0)} |",
        f"| Prompt rows | {metadata.get('prompt_rows', 0)} |",
        f"| Parse ok rate | {100.0 * safe_float(metadata.get('parse_ok_rate')):.1f}% |",
        f"| Entity outputs | {metadata.get('entity_outputs', 0)} |",
        f"| New Recall@5 | {100.0 * safe_float(overall.get('new_recall_at_5')):.1f}% |",
        f"| New Recall@10 | {100.0 * safe_float(overall.get('new_recall_at_10')):.1f}% |",
        f"| New Recall@20 | {100.0 * safe_float(overall.get('new_recall_at_20')):.1f}% |",
        f"| Mean source rank | {safe_float(overall.get('mean_source_rank')):.1f} |",
        f"| Mean probe rank | {safe_float(overall.get('mean_probe_rank')):.1f} |",
        f"| Mean rank improvement | {safe_float(overall.get('mean_rank_improvement')):.1f} |",
        f"| Decision | {metadata.get('decision', '')} |",
        "",
        "## Rank Shift",
        "",
        "| Bucket | Titles | New @5 | New @10 | New @20 | mean source rank | mean probe rank | mean improvement | positive-score rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rank_shift_summary:
        lines.append(
            "| {bucket} | {n} | {r5:.1f}% | {r10:.1f}% | {r20:.1f}% | {sr:.1f} | {pr:.1f} | {imp:.1f} | {pos:.1f}% |".format(
                bucket=row.get("bucket"),
                n=row.get("missing_gold_titles"),
                r5=100.0 * safe_float(row.get("new_recall_at_5")),
                r10=100.0 * safe_float(row.get("new_recall_at_10")),
                r20=100.0 * safe_float(row.get("new_recall_at_20")),
                sr=safe_float(row.get("mean_source_rank")),
                pr=safe_float(row.get("mean_probe_rank")),
                imp=safe_float(row.get("mean_rank_improvement")),
                pos=100.0 * safe_float(row.get("positive_score_rate")),
            )
        )

    best_cases = sorted(probe_rows, key=lambda row: safe_float(row.get("rank_improvement")), reverse=True)[:12]
    lines.extend(["", "## Top Rank Improvements", ""])
    for row in best_cases:
        lines.append(
            "- q{qid} `{gold}` rank {src} -> {probe} via `{entity}` from `{upstream}` ({reason})".format(
                qid=row.get("query_index"),
                gold=row.get("gold_title"),
                src=row.get("source_best_gold_rank"),
                probe=row.get("best_probe_rank"),
                entity=row.get("best_probe_entity"),
                upstream=row.get("best_upstream_title"),
                reason=row.get("best_probe_reason"),
            )
        )

    lines.extend([
        "",
        "## Files",
        "",
        f"- Probe rows: `{report_dir / 'probe_rows.csv'}`",
        f"- Entity outputs: `{report_dir / 'entity_outputs.jsonl'}`",
        f"- Rank summary: `{report_dir / 'rank_shift_summary.csv'}`",
        f"- Full summary: `{report_dir / 'summary.json'}`",
        "",
    ])
    return "\n".join(lines)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    missing_rows = target_missing_gold_rows(
        audit_dir=args.audit_dir,
        dataset_label=MUSIQUE_LABEL,
        bucket=TARGET_BUCKET,
        query_primary_only=bool(args.query_primary_only),
    )
    grouped = grouped_missing_gold(missing_rows)
    query_indices = sorted(grouped)
    if int(args.limit_queries) > 0:
        query_indices = query_indices[: int(args.limit_queries)]

    artifacts = load_artifacts(MUSIQUE_DATASET)
    source_records = artifacts["source_records"]
    setr_records = artifacts["setr_records"]
    dbec_traces = artifacts["dbec_traces"]

    llm_probe = probe_llm(str(args.llm_base_url), str(args.llm_model), min(int(args.timeout), 15))
    run_llm = bool(args.run_llm)
    if run_llm and not llm_probe.get("available"):
        raise RuntimeError(f"LLM unavailable at {args.llm_base_url}: {llm_probe.get('error')}")

    cache_path = Path(args.cache_path) if args.cache_path else report_dir / "llm_cache.jsonl"
    cache = JsonlCache(cache_path)
    prompt_rows: list[dict[str, Any]] = []
    entity_rows_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    errors: Counter[str] = Counter()

    for query_index in query_indices:
        source_record = source_records[query_index]
        setr_record = setr_records[query_index]
        selector_trace = selector_trace_from_query_trace(dbec_traces[query_index])
        requirements = requirement_by_id(selector_trace)
        pool_titles = list(source_record.get("pool_titles") or [])
        pool_docs = list(source_record.get("pool_docs") or [])
        question = str(source_record.get("question") or selector_trace.get("query") or "")

        for requirement in dependent_requirements(selector_trace):
            requirement_id = str(requirement.get("unit_id") or "")
            depends_on = [str(item) for item in requirement.get("depends_on") or []]
            upstream_requirements = [requirements[dep_id] for dep_id in depends_on if dep_id in requirements]
            upstream_positions = upstream_positions_for_requirement(
                requirement=requirement,
                selector_trace=selector_trace,
                setr_record=setr_record,
                pool_titles=pool_titles,
                source_top_n=int(args.source_top_n),
                max_upstream_docs=int(args.max_upstream_docs),
            )
            for upstream_position in upstream_positions:
                upstream_title = str(pool_titles[upstream_position]) if upstream_position < len(pool_titles) else ""
                upstream_doc = str(pool_docs[upstream_position]) if upstream_position < len(pool_docs) else upstream_title
                messages = build_messages(
                    question=question,
                    requirement=requirement,
                    upstream_requirements=upstream_requirements,
                    upstream_title=upstream_title,
                    upstream_doc=upstream_doc,
                    max_doc_chars=int(args.max_doc_chars),
                )
                key = sha1_json(
                    {
                        "prompt_version": PROMPT_VERSION,
                        "query_index": query_index,
                        "requirement_id": requirement_id,
                        "upstream_position": upstream_position,
                        "upstream_title": upstream_title,
                        "model": str(args.llm_model),
                        "max_doc_chars": int(args.max_doc_chars),
                    }
                )
                raw = cache.get(key)
                cache_hit = raw is not None
                if raw is None and run_llm:
                    try:
                        raw = chat_completion_raw(
                            base_url=str(args.llm_base_url),
                            model=str(args.llm_model),
                            messages=messages,
                            max_tokens=int(args.max_tokens),
                            timeout=int(args.timeout),
                        )
                        cache.set(
                            key,
                            raw,
                            {
                                "query_index": query_index,
                                "requirement_id": requirement_id,
                                "upstream_position": upstream_position,
                                "upstream_title": upstream_title,
                            },
                        )
                    except Exception as exc:
                        raw = f"ERROR: {exc}"
                        errors["llm_call_failed"] += 1
                elif raw is None:
                    raw = ""

                parsed = parse_json_payload(raw)
                entities = entities_from_raw(raw)
                prompt_row = {
                    "query_index": query_index,
                    "question": question,
                    "requirement_id": requirement_id,
                    "requirement_subquery": str(requirement.get("subquery") or ""),
                    "depends_on_json": json.dumps(depends_on, ensure_ascii=False),
                    "upstream_position": upstream_position,
                    "upstream_title": upstream_title,
                    "raw": raw,
                    "parsed_json": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "",
                    "parse_ok": int(parsed is not None),
                    "entities_json": json.dumps(entities, ensure_ascii=False),
                    "entity_count": len(entities),
                    "cache_hit": int(cache_hit),
                }
                prompt_rows.append(prompt_row)
                for entity in entities:
                    entity_rows_by_query[query_index].append(
                        {
                            "query_index": query_index,
                            "entity": entity["entity"],
                            "support_span": entity.get("support_span", ""),
                            "why": entity.get("why", ""),
                            "requirement_id": requirement_id,
                            "requirement_subquery": str(requirement.get("subquery") or ""),
                            "upstream_position": upstream_position,
                            "upstream_title": upstream_title,
                        }
                    )

    probe_rows: list[dict[str, Any]] = []
    for query_index in query_indices:
        source_record = source_records[query_index]
        pool_titles = list(source_record.get("pool_titles") or [])
        pool_docs = list(source_record.get("pool_docs") or [])
        entity_rows = entity_rows_by_query.get(query_index, [])
        order, best_by_position = rank_pool_by_entities(
            pool_titles=pool_titles,
            pool_docs=pool_docs,
            entity_rows=entity_rows,
        )
        for missing_row in grouped[query_index]:
            gold_title = str(missing_row.get("gold_title") or "")
            source_rank = safe_int(missing_row.get("source_best_gold_rank"), default=999)
            probe_rank = rank_of_title(gold_title, order, pool_titles)
            if probe_rank is None:
                probe_rank = source_rank
            gold_positions = [
                pos
                for pos, title in enumerate(pool_titles)
                if normalize_title(title) == normalize_title(gold_title)
            ]
            gold_pos = gold_positions[0] if gold_positions else -1
            best = best_by_position.get(gold_pos, {"score": 0.0})
            if not entity_rows:
                failure_reason = "no_entity_outputs"
            elif safe_float(best.get("score")) <= 0:
                failure_reason = "no_positive_score_for_gold"
            elif probe_rank >= 20:
                failure_reason = "not_top20"
            elif probe_rank >= 10:
                failure_reason = "top20_not_top10"
            elif probe_rank >= 5:
                failure_reason = "top10_not_top5"
            else:
                failure_reason = "recovered_top5"

            row = {
                "dataset": MUSIQUE_LABEL,
                "base_dataset": MUSIQUE_DATASET,
                "query_index": query_index,
                "question": str(missing_row.get("question") or source_record.get("question") or ""),
                "gold_title": gold_title,
                "source_best_gold_rank": source_rank,
                "best_probe_rank": probe_rank,
                "rank_improvement": source_rank - probe_rank,
                "baseline_hit_at_3": int(source_rank < 3),
                "baseline_hit_at_5": int(source_rank < 5),
                "baseline_hit_at_10": int(source_rank < 10),
                "baseline_hit_at_20": int(source_rank < 20),
                "probe_hit_at_3": int(probe_rank < 3),
                "probe_hit_at_5": int(probe_rank < 5),
                "probe_hit_at_10": int(probe_rank < 10),
                "probe_hit_at_20": int(probe_rank < 20),
                "new_hit_at_3": int(source_rank >= 3 and probe_rank < 3),
                "new_hit_at_5": int(source_rank >= 5 and probe_rank < 5),
                "new_hit_at_10": int(source_rank >= 10 and probe_rank < 10),
                "new_hit_at_20": int(source_rank >= 20 and probe_rank < 20),
                "best_probe_score": safe_float(best.get("score")),
                "best_probe_entity": str(best.get("entity") or ""),
                "best_probe_reason": str(best.get("reason") or ""),
                "best_requirement_id": str(best.get("requirement_id") or ""),
                "best_requirement_subquery": str(best.get("requirement_subquery") or ""),
                "best_upstream_position": safe_int(best.get("upstream_position"), default=-1),
                "best_upstream_title": str(best.get("upstream_title") or ""),
                "best_support_span": str(best.get("support_span") or ""),
                "failure_reason": failure_reason,
                "entity_outputs_json": json.dumps(entity_rows, ensure_ascii=False),
            }
            probe_rows.append(row)

    rank_shift_summary, metadata = summarize_probe_rows(probe_rows, prompt_rows=prompt_rows)
    metadata = {
        **metadata,
        "dataset": MUSIQUE_LABEL,
        "base_dataset": MUSIQUE_DATASET,
        "target_bucket": TARGET_BUCKET,
        "audit_dir": str(args.audit_dir),
        "query_primary_only": bool(args.query_primary_only),
        "source_pool": str(source_pool_path(MUSIQUE_DATASET)),
        "setr_pool": str(setr_pool_path(MUSIQUE_DATASET)),
        "dbec_trace": str(dbec_report_path(MUSIQUE_DATASET)),
        "report_dir": str(report_dir),
        "cache_path": str(cache_path),
        "run_llm": run_llm,
        "llm_base_url": str(args.llm_base_url),
        "llm_model": str(args.llm_model),
        "llm_probe": llm_probe,
        "limit_queries": int(args.limit_queries),
        "max_upstream_docs": int(args.max_upstream_docs),
        "source_top_n": int(args.source_top_n),
        "max_doc_chars": int(args.max_doc_chars),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "errors": dict(errors),
    }
    payload = {
        "metadata": metadata,
        "rank_shift_summary": rank_shift_summary,
        "probe_rows": probe_rows,
    }
    write_csv(probe_rows, report_dir / "probe_rows.csv")
    write_jsonl(prompt_rows, report_dir / "entity_outputs.jsonl")
    write_csv(rank_shift_summary, report_dir / "rank_shift_summary.csv")
    write_json(payload, report_dir / "summary.json")
    (report_dir / "summary.md").write_text(
        build_markdown(
            metadata=metadata,
            rank_shift_summary=rank_shift_summary,
            probe_rows=probe_rows,
            report_dir=report_dir,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--audit_dir", type=Path, default=CANDIDATE_AUDIT_DIR)
    parser.add_argument("--limit_queries", type=int, default=0)
    parser.add_argument("--query_primary_only", action="store_true")
    parser.add_argument("--cache_path", type=Path, default=None)
    parser.add_argument("--run_llm", action="store_true")
    parser.add_argument("--llm_base_url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_tokens", type=int, default=220)
    parser.add_argument("--max_doc_chars", type=int, default=DEFAULT_MAX_DOC_CHARS)
    parser.add_argument("--max_upstream_docs", type=int, default=DEFAULT_MAX_UPSTREAM_DOCS)
    parser.add_argument("--source_top_n", type=int, default=DEFAULT_SOURCE_TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_probe(args)
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(f"Wrote chain-walking binding probe to {Path(args.report_dir)}")


if __name__ == "__main__":
    main()
