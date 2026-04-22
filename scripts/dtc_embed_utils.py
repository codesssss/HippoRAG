from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.hipporag.utils.causal_utils import normalize_structure_text


@dataclass(frozen=True)
class DTCRequirement:
    unit_id: str
    subquery: str
    depends_on: Tuple[str, ...] = ()
    expected_answer_type: str = "unknown"
    anchor_mentions: Tuple[str, ...] = ()
    role: str = "support"

    def to_trace(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["depends_on"] = list(self.depends_on)
        payload["anchor_mentions"] = list(self.anchor_mentions)
        return payload


def _strip_think_tags(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", str(text or "")).strip()


def _extract_json_payload(text: str) -> Any | None:
    cleaned = _strip_think_tags(text)
    candidates: List[str] = []
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_char)
        end = cleaned.rfind(close_char)
        if start >= 0 and end > start:
            candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _coerce_string_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        key = normalize_structure_text(text)
        if not text or not key or key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
    return tuple(cleaned)


def parse_dtc_decomposition_response(
    response_text: str,
    *,
    max_steps: int = 4,
) -> Tuple[List[DTCRequirement], Dict[str, object]]:
    payload = _extract_json_payload(response_text)
    trace: Dict[str, object] = {
        "parse_succeeded": False,
        "parse_error": None,
        "raw_preview": str(response_text or "")[:500],
        "parsed_step_count": 0,
        "active_step_count": 0,
    }
    if payload is None:
        trace["parse_error"] = "json_parse_failed"
        return [], trace

    if isinstance(payload, dict):
        steps = payload.get("steps", payload.get("requirements", []))
    else:
        steps = payload
    if not isinstance(steps, list):
        trace["parse_error"] = "payload_not_list"
        return [], trace

    requirements: List[DTCRequirement] = []
    used_ids: set[str] = set()
    trace["parsed_step_count"] = int(len(steps))
    for idx, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            continue
        subquery = str(
            raw_step.get("subquery")
            or raw_step.get("question")
            or raw_step.get("need")
            or ""
        ).strip()
        if not subquery:
            continue
        raw_id = str(
            raw_step.get("id")
            or raw_step.get("unit_id")
            or raw_step.get("step_id")
            or f"s{idx + 1}"
        ).strip()
        unit_id = normalize_structure_text(raw_id).replace(" ", "_") or f"s{idx + 1}"
        if unit_id in used_ids:
            unit_id = f"{unit_id}_{idx + 1}"
        used_ids.add(unit_id)
        requirements.append(
            DTCRequirement(
                unit_id=unit_id,
                subquery=subquery,
                depends_on=tuple(
                    normalize_structure_text(dep).replace(" ", "_")
                    for dep in _coerce_string_list(raw_step.get("depends_on", raw_step.get("dependencies", [])))
                    if normalize_structure_text(dep)
                ),
                expected_answer_type=str(raw_step.get("expected_answer_type", "unknown") or "unknown").strip().lower(),
                anchor_mentions=_coerce_string_list(raw_step.get("anchor_mentions", raw_step.get("anchors", []))),
                role=str(raw_step.get("role", raw_step.get("type", "support")) or "support").strip().lower(),
            )
        )
        if len(requirements) >= max(1, int(max_steps)):
            break

    valid_ids = {req.unit_id for req in requirements}
    normalized_requirements: List[DTCRequirement] = []
    for req in requirements:
        normalized_requirements.append(
            DTCRequirement(
                unit_id=req.unit_id,
                subquery=req.subquery,
                depends_on=tuple(dep for dep in req.depends_on if dep in valid_ids),
                expected_answer_type=req.expected_answer_type,
                anchor_mentions=req.anchor_mentions,
                role=req.role,
            )
        )

    trace["parse_succeeded"] = bool(normalized_requirements)
    trace["active_step_count"] = int(len(normalized_requirements))
    if not normalized_requirements:
        trace["parse_error"] = "no_active_steps"
    return normalized_requirements, trace


def build_fallback_dtc_requirements(query: str) -> List[DTCRequirement]:
    return [
        DTCRequirement(
            unit_id="q0",
            subquery=str(query or "").strip(),
            depends_on=(),
            expected_answer_type="unknown",
            anchor_mentions=(),
            role="query",
        )
    ] if str(query or "").strip() else []


def _normalize_vector(vector: Any) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def _min_max(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    finite = np.asarray(values, dtype=float)
    finite[~np.isfinite(finite)] = float(np.nan)
    if np.all(np.isnan(finite)):
        return np.zeros_like(values, dtype=float)
    min_value = float(np.nanmin(finite))
    max_value = float(np.nanmax(finite))
    if max_value <= min_value:
        return np.ones_like(values, dtype=float) if max_value > 0 else np.zeros_like(values, dtype=float)
    normalized = (finite - min_value) / (max_value - min_value)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(float)


def _doc_text_body(doc_text: str) -> str:
    return str(doc_text or "").split("\n", 1)[1] if "\n" in str(doc_text or "") else str(doc_text or "")


def _anchor_hit_score(anchor_mentions: Sequence[str],
                      doc_title: str,
                      doc_text: str,
                      doc_entities: set[str]) -> float:
    if not anchor_mentions:
        return 0.0
    normalized_title = normalize_structure_text(doc_title)
    normalized_text = normalize_structure_text(_doc_text_body(doc_text))
    normalized_entities = {normalize_structure_text(entity) for entity in doc_entities}
    for anchor in anchor_mentions:
        key = normalize_structure_text(anchor)
        if not key:
            continue
        if key in normalized_title or normalized_title in key:
            return 1.0
        if key in normalized_entities:
            return 1.0
        if key in normalized_text:
            return 0.5
    return 0.0


def _dependency_hit_score(dependency_entities: set[str], doc_entities: set[str], doc_title: str) -> float:
    if not dependency_entities:
        return 0.0
    normalized_title = normalize_structure_text(doc_title)
    normalized_doc_entities = {normalize_structure_text(entity) for entity in doc_entities}
    overlap = dependency_entities & normalized_doc_entities
    if overlap:
        return 1.0
    for entity in dependency_entities:
        if entity and (entity in normalized_title or normalized_title in entity):
            return 0.75
    return 0.0


def _contains_normalized_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def _phrase_occurrences(haystack: str, needle: str) -> Tuple[int, int]:
    if not haystack or not needle:
        return 0, 10**9
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    matches = list(re.finditer(pattern, haystack))
    if not matches:
        return 0, 10**9
    return len(matches), int(matches[0].start())


def _is_generic_binding_title(title: str) -> bool:
    key = normalize_structure_text(title)
    if not key:
        return True
    generic_prefixes = (
        "list of ",
        "outline of ",
        "index of ",
        "timeline of ",
        "category ",
    )
    if any(key.startswith(prefix) for prefix in generic_prefixes):
        return True
    if len(key) <= 2:
        return True
    return False


def _candidate_type_compatible(title: str, expected_answer_type: str) -> bool:
    if _is_generic_binding_title(title):
        return False
    expected = normalize_structure_text(expected_answer_type)
    if not expected or expected in {"unknown", "entity", "thing"}:
        return True
    key = normalize_structure_text(title)
    # Keep this conservative: only reject obvious document-type mismatches.
    work_markers = {" film", " song", " album", " episode", " season", " novel", " book"}
    if expected in {"person", "people", "human"}:
        return not any(marker in f" {key}" for marker in {" film", " song", " album", " episode", " season"})
    if expected in {"location", "place", "country", "city"}:
        return not any(marker in f" {key}" for marker in work_markers)
    if expected in {"organization", "organisation", "company", "team"}:
        return not any(marker in f" {key}" for marker in work_markers)
    return True


def _binding_candidate_hit_score(candidate_key: str,
                                 doc_entities: set[str],
                                 doc_title: str,
                                 doc_text: str) -> float:
    if not candidate_key:
        return 0.0
    title_key = normalize_structure_text(doc_title)
    if candidate_key and (candidate_key == title_key or candidate_key in title_key or title_key in candidate_key):
        return 1.0
    normalized_entities = {normalize_structure_text(entity) for entity in doc_entities}
    if candidate_key in normalized_entities:
        return 1.0
    normalized_text = normalize_structure_text(_doc_text_body(doc_text))
    if _contains_normalized_phrase(normalized_text, candidate_key):
        return 0.5
    return 0.0


def _build_bound_subquery(subquery: str, candidate_title: str) -> str:
    text = str(subquery or "").strip()
    title = str(candidate_title or "").strip()
    if not text or not title:
        return text
    replaced = re.sub(r"\[[^\]]+\]", title, text)
    referent_pattern = re.compile(
        r"\b(that|the)\s+("
        r"person|entity|director|performer|singer|composer|author|writer|actor|actress|"
        r"father|mother|parent|husband|wife|spouse|child|team|city|place|country|"
        r"organization|organisation|company|film|movie|work"
        r")\b",
        flags=re.IGNORECASE,
    )
    replaced = referent_pattern.sub(title, replaced, count=1)
    if replaced != text:
        return replaced
    return f"{text} Target entity: {title}."


def select_dtc_embed_positions(
    *,
    query: str,
    requirements: Sequence[DTCRequirement],
    requirement_embeddings: Dict[str, np.ndarray],
    pool_docs: Sequence[str],
    pool_doc_ids: Sequence[int | None],
    pool_doc_scores: Sequence[float],
    pool_doc_titles: Sequence[str],
    doc_idx_to_entities: Dict[int, set[str]],
    passage_embeddings: np.ndarray,
    qa_top_k: int,
    reserve_top_m: int = 2,
    match_threshold: float = 0.35,
    redundancy_weight: float = 0.10,
    base_weight: float = 0.05,
    anchor_bonus_weight: float = 0.10,
    dependency_bonus_weight: float = 0.10,
    non_anchor_title_dedup: bool = True,
    enable_dependency_binding: bool = False,
    enforce_dependencies: bool = True,
    binding_max_candidates: int = 4,
    binding_entity_hit_required: bool = True,
    require_new_crossing: bool = False,
    embed_texts_fn: Callable[[Sequence[str]], Dict[str, np.ndarray]] | None = None,
) -> Tuple[List[int], Dict[str, object]]:
    pool_limit = len(pool_docs)
    target_k = min(max(int(qa_top_k), 0), pool_limit)
    if target_k <= 0:
        return [], {"selector": "dtc_embed", "status": "empty_pool"}

    active_requirements = list(requirements)
    if not enforce_dependencies:
        active_requirements = [
            DTCRequirement(
                unit_id=req.unit_id,
                subquery=req.subquery,
                depends_on=(),
                expected_answer_type=req.expected_answer_type,
                anchor_mentions=req.anchor_mentions,
                role=req.role,
            )
            for req in active_requirements
        ]
    if not active_requirements:
        fallback_positions = list(range(target_k))
        return fallback_positions, {
            "selector": "dtc_embed",
            "status": "fallback_no_requirements",
            "selection_steps": [],
            "final_positions": fallback_positions,
        }

    doc_vectors: Dict[int, np.ndarray] = {}
    for pos, doc_id in enumerate(pool_doc_ids[:pool_limit]):
        if doc_id is None or int(doc_id) < 0 or int(doc_id) >= len(passage_embeddings):
            continue
        vector = _normalize_vector(passage_embeddings[int(doc_id)])
        if vector.size:
            doc_vectors[int(pos)] = vector

    req_vectors = {
        req.unit_id: _normalize_vector(requirement_embeddings.get(req.unit_id, np.array([])))
        for req in active_requirements
    }
    raw_match_scores: Dict[str, np.ndarray] = {}
    for req in active_requirements:
        req_vec = req_vectors.get(req.unit_id, np.array([]))
        values: List[float] = []
        for pos in range(pool_limit):
            doc_vec = doc_vectors.get(pos)
            if req_vec.size == 0 or doc_vec is None or req_vec.shape != doc_vec.shape:
                values.append(float("-inf"))
            else:
                values.append(float(np.dot(req_vec, doc_vec)))
        raw_match_scores[req.unit_id] = _min_max(np.asarray(values, dtype=float))

    normalized_base_scores = _min_max(np.asarray(pool_doc_scores[:pool_limit], dtype=float))
    normalized_query = normalize_structure_text(query)
    query_anchor_keys = {
        normalize_structure_text(anchor)
        for req in active_requirements
        for anchor in req.anchor_mentions
        if normalize_structure_text(anchor)
    }
    for title in pool_doc_titles[:pool_limit]:
        title_key = normalize_structure_text(title)
        if title_key and _contains_normalized_phrase(normalized_query, title_key):
            query_anchor_keys.add(title_key)

    selected_positions: List[int] = []
    seen_titles: set[str] = set()
    reserve_count = min(max(int(reserve_top_m), 0), target_k, pool_limit)
    for pos in range(reserve_count):
        selected_positions.append(pos)
        title_key = normalize_structure_text(pool_doc_titles[pos]) if pos < len(pool_doc_titles) else ""
        if title_key:
            seen_titles.add(title_key)

    req_by_id = {req.unit_id: req for req in active_requirements}
    coverage_by_req = {req.unit_id: 0.0 for req in active_requirements}
    cover_position_by_req: Dict[str, int] = {}
    dependency_entities_by_req: Dict[str, set[str]] = {}
    binding_candidates_by_req: Dict[str, List[Dict[str, object]]] = {}
    binding_embedding_cache: Dict[str, np.ndarray] = {}
    binding_score_cache: Dict[Tuple[str, str], np.ndarray] = {}
    selection_steps: List[Dict[str, object]] = []

    def doc_entities_for_pos(pos: int) -> set[str]:
        doc_id = pool_doc_ids[pos] if 0 <= pos < len(pool_doc_ids) else None
        if doc_id is None:
            return set()
        return set(doc_idx_to_entities.get(int(doc_id), set()) or set())

    def collect_binding_candidates(req: DTCRequirement, dep_entities: set[str]) -> List[Dict[str, object]]:
        if not enable_dependency_binding or not req.depends_on:
            return []
        rows: List[Dict[str, object]] = []
        seen_keys: set[str] = set()
        for dep in req.depends_on:
            dep_pos = cover_position_by_req.get(dep)
            if dep_pos is None or dep_pos < 0 or dep_pos >= pool_limit:
                continue
            dep_req = req_by_id.get(dep)
            expected_type = dep_req.expected_answer_type if dep_req is not None else "unknown"
            upstream_text = normalize_structure_text(_doc_text_body(pool_docs[dep_pos]))
            upstream_title_key = normalize_structure_text(pool_doc_titles[dep_pos])
            for title_pos, raw_title in enumerate(pool_doc_titles[:pool_limit]):
                title = str(raw_title or "").strip()
                title_key = normalize_structure_text(title)
                if not title_key or title_key in seen_keys:
                    continue
                if title_key == upstream_title_key or title_key in query_anchor_keys:
                    continue
                if not _candidate_type_compatible(title, expected_type):
                    continue
                count, first_pos = _phrase_occurrences(upstream_text, title_key)
                if count <= 0:
                    continue
                seen_keys.add(title_key)
                rows.append({
                    "title": title,
                    "key": title_key,
                    "dep": dep,
                    "dep_position": int(dep_pos),
                    "title_pool_position": int(title_pos),
                    "count": int(count),
                    "first_pos": int(first_pos),
                    "expected_answer_type": str(expected_type or "unknown"),
                })
        rows.sort(key=lambda row: (int(row["first_pos"]), -int(row["count"]), int(row["title_pool_position"])))
        return rows[:max(0, int(binding_max_candidates))]

    def ensure_bound_score(req: DTCRequirement, candidate: Dict[str, object]) -> np.ndarray | None:
        candidate_key = str(candidate.get("key", "") or "")
        if not candidate_key:
            return None
        cache_key = (req.unit_id, candidate_key)
        cached_score = binding_score_cache.get(cache_key)
        if cached_score is not None:
            return cached_score
        candidate_title = str(candidate.get("title", "") or "")
        bound_query = _build_bound_subquery(req.subquery, candidate_title)
        if not bound_query:
            return None
        if bound_query not in binding_embedding_cache:
            if embed_texts_fn is None:
                return None
            embedded = embed_texts_fn([bound_query]) or {}
            vector = embedded.get(bound_query) if isinstance(embedded, dict) else None
            if vector is None:
                return None
            binding_embedding_cache[bound_query] = _normalize_vector(vector)
        req_vec = binding_embedding_cache.get(bound_query, np.array([]))
        values: List[float] = []
        for pos in range(pool_limit):
            doc_vec = doc_vectors.get(pos)
            if req_vec.size == 0 or doc_vec is None or req_vec.shape != doc_vec.shape:
                values.append(float("-inf"))
            else:
                values.append(float(np.dot(req_vec, doc_vec)))
        scores = _min_max(np.asarray(values, dtype=float))
        binding_score_cache[cache_key] = scores
        return scores

    def score_requirement_for_position(req: DTCRequirement, pos: int) -> Tuple[float, Dict[str, object]]:
        dep_entities = dependency_entities_by_req.get(req.unit_id, set())
        req_score = float(raw_match_scores.get(req.unit_id, np.zeros(pool_limit))[pos])
        anchor_score = _anchor_hit_score(
            req.anchor_mentions,
            pool_doc_titles[pos],
            pool_docs[pos],
            doc_entities_for_pos(pos),
        )
        dep_score = _dependency_hit_score(
            dep_entities,
            doc_entities_for_pos(pos),
            pool_doc_titles[pos],
        )
        binding_candidates = binding_candidates_by_req.get(req.unit_id, [])
        binding_rows: List[Dict[str, object]] = []
        best_binding_score = 0.0
        best_binding_hit = 0.0
        best_binding_title = ""
        if binding_candidates:
            for candidate in binding_candidates:
                candidate_key = str(candidate.get("key", "") or "")
                hit_score = _binding_candidate_hit_score(
                    candidate_key,
                    doc_entities_for_pos(pos),
                    pool_doc_titles[pos],
                    pool_docs[pos],
                )
                bound_scores = ensure_bound_score(req, candidate)
                bound_score = float(bound_scores[pos]) if bound_scores is not None else 0.0
                if binding_entity_hit_required and hit_score <= 0.0:
                    effective = 0.0
                else:
                    effective = min(1.0, max(0.0, bound_score + float(dependency_bonus_weight) * hit_score))
                binding_rows.append({
                    "title": str(candidate.get("title", "") or ""),
                    "hit": round(float(hit_score), 4),
                    "score": round(float(bound_score), 4),
                    "effective": round(float(effective), 4),
                })
                if effective > best_binding_score:
                    best_binding_score = float(effective)
                    best_binding_hit = float(hit_score)
                    best_binding_title = str(candidate.get("title", "") or "")
            req_score = best_binding_score
            dep_score = max(dep_score, best_binding_hit)
        elif req.depends_on and dep_entities and dep_score <= 0.0:
            req_score = 0.0
        else:
            req_score = min(
                1.0,
                max(
                    0.0,
                    req_score
                    + float(anchor_bonus_weight) * anchor_score
                    + float(dependency_bonus_weight) * dep_score,
                ),
            )
        return min(1.0, max(0.0, req_score)), {
            "anchor_score": round(float(anchor_score), 4),
            "dependency_score": round(float(dep_score), 4),
            "binding_title": best_binding_title,
            "binding_candidates": binding_rows,
        }

    def mark_coverage_from_position(pos: int, *, mode: str) -> None:
        made_progress = True
        while made_progress:
            made_progress = False
            selected_entity_context = {
                normalize_structure_text(entity)
                for selected_pos in selected_positions
                for entity in doc_entities_for_pos(selected_pos)
            }
            for selected_pos in selected_positions:
                title_key = normalize_structure_text(pool_doc_titles[selected_pos])
                if title_key:
                    selected_entity_context.add(title_key)
            for req in active_requirements:
                if any(dep not in cover_position_by_req for dep in req.depends_on):
                    continue
                dep_entities: set[str] = set()
                for dep in req.depends_on:
                    dep_pos = cover_position_by_req.get(dep)
                    if dep_pos is None:
                        continue
                    dep_entities.update(
                        normalize_structure_text(entity)
                        for entity in doc_entities_for_pos(dep_pos)
                    )
                    dep_title = normalize_structure_text(pool_doc_titles[dep_pos])
                    if dep_title:
                        dep_entities.add(dep_title)
                dependency_entities_by_req[req.unit_id] = dep_entities
                binding_candidates_by_req[req.unit_id] = collect_binding_candidates(req, dep_entities)
                score, score_trace = score_requirement_for_position(req, pos)
                previous_score = float(coverage_by_req.get(req.unit_id, 0.0))
                previous_cover_position = cover_position_by_req.get(req.unit_id)
                if score > previous_score:
                    coverage_by_req[req.unit_id] = score
                    should_set_cover = score >= float(match_threshold) and (
                        req.unit_id not in cover_position_by_req or bool(enable_dependency_binding)
                    )
                    if should_set_cover:
                        cover_position_by_req[req.unit_id] = int(pos)
                        if previous_cover_position != int(pos):
                            made_progress = True
                        selection_steps.append({
                            "step": int(len(selection_steps) + 1),
                            "mode": mode,
                            "covered_requirement_id": req.unit_id,
                            "cover_update": previous_cover_position is not None,
                            "pool_position": int(pos),
                            "title": pool_doc_titles[pos],
                            "coverage_score": round(float(score), 4),
                            "previous_coverage_score": round(float(previous_score), 4),
                            "binding_title": score_trace.get("binding_title", ""),
                        })

    for reserved_pos in list(selected_positions):
        mark_coverage_from_position(reserved_pos, mode="reserve")

    while len(selected_positions) < target_k:
        selected_set = set(selected_positions)
        best_row: Dict[str, object] | None = None
        for pos in range(pool_limit):
            if pos in selected_set:
                continue
            title_key = normalize_structure_text(pool_doc_titles[pos]) if pos < len(pool_doc_titles) else ""
            if non_anchor_title_dedup and title_key and title_key in seen_titles:
                continue
            doc_vec = doc_vectors.get(pos)
            redundancy = 0.0
            if doc_vec is not None and selected_positions:
                selected_sims = [
                    float(np.dot(doc_vec, doc_vectors[selected_pos]))
                    for selected_pos in selected_positions
                    if selected_pos in doc_vectors and doc_vectors[selected_pos].shape == doc_vec.shape
                ]
                redundancy = max(selected_sims) if selected_sims else 0.0
            coverage_gain = 0.0
            newly_crossed: List[str] = []
            req_rows: List[Dict[str, object]] = []
            for req in active_requirements:
                if any(dep not in cover_position_by_req for dep in req.depends_on):
                    continue
                req_score, score_trace = score_requirement_for_position(req, pos)
                previous = float(coverage_by_req.get(req.unit_id, 0.0))
                delta = max(0.0, req_score - previous)
                if req_score >= float(match_threshold) and previous < float(match_threshold):
                    newly_crossed.append(req.unit_id)
                coverage_gain += delta
                if delta > 0 or req_score >= float(match_threshold):
                    req_rows.append({
                        "unit_id": req.unit_id,
                        "score": round(float(req_score), 4),
                        "previous": round(float(previous), 4),
                        "delta": round(float(delta), 4),
                        "binding_title": score_trace.get("binding_title", ""),
                    })
            total_gain = (
                float(coverage_gain)
                + 0.25 * float(len(newly_crossed))
                + float(base_weight) * float(normalized_base_scores[pos])
                - float(redundancy_weight) * max(0.0, float(redundancy))
            )
            row = {
                "pool_position": int(pos),
                "title": pool_doc_titles[pos],
                "coverage_gain": float(coverage_gain),
                "new_requirement_count": int(len(newly_crossed)),
                "new_requirement_ids": list(newly_crossed),
                "base_score": float(normalized_base_scores[pos]),
                "redundancy": float(redundancy),
                "total_gain": float(total_gain),
                "requirement_scores": req_rows,
            }
            if best_row is None or (
                float(row["total_gain"]),
                float(row["coverage_gain"]),
                int(row["new_requirement_count"]),
                -int(pos),
            ) > (
                float(best_row["total_gain"]),
                float(best_row["coverage_gain"]),
                int(best_row["new_requirement_count"]),
                -int(best_row["pool_position"]),
            ):
                best_row = row

        no_acceptable_candidate = best_row is None
        if best_row is not None:
            new_requirement_count = int(best_row["new_requirement_count"])
            if require_new_crossing:
                no_acceptable_candidate = new_requirement_count <= 0
            else:
                no_acceptable_candidate = (
                    float(best_row["coverage_gain"]) <= 1e-9
                    and new_requirement_count <= 0
                )

        if no_acceptable_candidate:
            fill_pos = next((pos for pos in range(pool_limit) if pos not in selected_set), None)
            if fill_pos is None:
                break
            selected_positions.append(int(fill_pos))
            title_key = normalize_structure_text(pool_doc_titles[fill_pos])
            if title_key:
                seen_titles.add(title_key)
            selection_steps.append({
                "step": int(len(selection_steps) + 1),
                "mode": "baseline_fill",
                "pool_position": int(fill_pos),
                "title": pool_doc_titles[fill_pos],
                "reason": (
                    "no_new_requirement_crossing"
                    if bool(require_new_crossing)
                    else "no_positive_requirement_gain"
                ),
            })
            continue

        chosen_pos = int(best_row["pool_position"])
        selected_positions.append(chosen_pos)
        title_key = normalize_structure_text(pool_doc_titles[chosen_pos])
        if title_key:
            seen_titles.add(title_key)
        selection_steps.append({
            "step": int(len(selection_steps) + 1),
            "mode": "dtc_cover",
            "pool_position": chosen_pos,
            "title": pool_doc_titles[chosen_pos],
            "coverage_gain": round(float(best_row["coverage_gain"]), 4),
            "new_requirement_count": int(best_row["new_requirement_count"]),
            "new_requirement_ids": list(best_row["new_requirement_ids"]),
            "base_score": round(float(best_row["base_score"]), 4),
            "redundancy": round(float(best_row["redundancy"]), 4),
            "total_gain": round(float(best_row["total_gain"]), 4),
            "requirement_scores": best_row["requirement_scores"],
        })
        mark_coverage_from_position(chosen_pos, mode="dtc_cover")

    if len(selected_positions) < target_k:
        selected_set = set(selected_positions)
        for pos in range(pool_limit):
            if pos in selected_set:
                continue
            selected_positions.append(pos)
            selected_set.add(pos)
            if len(selected_positions) >= target_k:
                break

    covered_count = sum(1 for value in coverage_by_req.values() if float(value) >= float(match_threshold))
    trace = {
        "selector": "dtc_embed",
        "status": "applied",
        "query": str(query),
        "requirement_count": int(len(active_requirements)),
        "requirements": [req.to_trace() for req in active_requirements],
        "match_threshold": round(float(match_threshold), 4),
        "reserve_top_m": int(reserve_count),
        "dependency_binding_enabled": bool(enable_dependency_binding),
        "dependency_enforced": bool(enforce_dependencies),
        "binding_max_candidates": int(binding_max_candidates),
        "binding_entity_hit_required": bool(binding_entity_hit_required),
        "require_new_crossing": bool(require_new_crossing),
        "binding_candidates_by_requirement": {
            req_id: [
                {
                    "title": str(row.get("title", "") or ""),
                    "dep": str(row.get("dep", "") or ""),
                    "dep_position": int(row.get("dep_position", -1)),
                    "count": int(row.get("count", 0)),
                    "first_pos": int(row.get("first_pos", 10**9)),
                }
                for row in rows
            ]
            for req_id, rows in binding_candidates_by_req.items()
        },
        "covered_requirement_count": int(covered_count),
        "covered_requirement_rate": round(float(covered_count) / float(max(len(active_requirements), 1)), 4),
        "coverage_by_requirement": {
            req_id: round(float(score), 4)
            for req_id, score in coverage_by_req.items()
        },
        "cover_position_by_requirement": dict(cover_position_by_req),
        "selected_positions": list(selected_positions[:target_k]),
        "selected_titles": [pool_doc_titles[pos] for pos in selected_positions[:target_k]],
        "selection_steps": selection_steps,
        "embedding_available_requirement_count": int(sum(req_vectors.get(req.unit_id, np.array([])).size > 0 for req in active_requirements)),
        "embedding_available_doc_count": int(len(doc_vectors)),
    }
    return selected_positions[:target_k], trace
