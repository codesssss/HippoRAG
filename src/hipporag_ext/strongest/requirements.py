from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

from src.hipporag.utils.misc_utils import strip_reasoning_content

from .traces import normalize_entity_text
from .types import RequirementUnit

logger = logging.getLogger(__name__)

UNSUPPORTED_SLOT_FAMILY = "unsupported"


ROLE_SLOT_MAP: Dict[str, List[str]] = {
    "director": ["director", "directed by"],
    "spouse": ["wife", "husband", "spouse", "married to"],
    "parent": ["parent", "father", "mother", "son", "daughter", "child"],
    "award_recipient": ["won", "received", "award", "prize"],
}

ATTR_SLOT_MAP: Dict[str, List[str]] = {
    "birthplace": ["birthplace", "place of birth", "born in", "born at"],
    "death_date": ["death date", "date of death", "died on", "when did"],
    "birth_date": ["birth date", "date of birth", "born on"],
    "nationality": ["nationality", "citizen of"],
    "education": ["educated at", "alma mater", "studied at"],
}

SLOT_EXPECTED_ANSWER_TYPE: Dict[str, str] = {
    "director": "person",
    "spouse": "person",
    "parent": "person",
    "award_recipient": "person",
    "performer": "person",
    "composer": "person",
    "creator": "person",
    "producer": "person",
    "birthplace": "location",
    "death_date": "date",
    "birth_date": "date",
    "nationality": "location",
    "education": "organization",
    "release_date": "date",
    "known_for": "work",
    "location": "location",
    UNSUPPORTED_SLOT_FAMILY: "unknown",
}

SINGLE_VALUED_SLOT_FAMILIES = frozenset(
    {
        "director",
        "spouse",
        "parent",
        "performer",
        "composer",
        "creator",
        "producer",
        "birthplace",
        "death_date",
        "birth_date",
        "nationality",
        "education",
        "release_date",
        "location",
    }
)


SLOT_CANONICALIZATION_MAP: Dict[str, str] = {
    # role slot aliases
    "director": "director", "directed_by": "director", "directed by": "director",
    "filmmaker": "director",
    "spouse": "spouse", "wife": "spouse", "husband": "spouse", "married_to": "spouse",
    "parent": "parent", "father": "parent", "mother": "parent",
    "son": "parent", "daughter": "parent", "child": "parent",
    "award_recipient": "award_recipient", "award": "award_recipient",
    "won": "award_recipient", "prize": "award_recipient",
    "performer": "performer", "singer": "performer", "artist": "performer",
    "star": "performer", "actor": "performer", "actress": "performer",
    "composer": "composer", "written_by": "composer", "songwriter": "composer",
    "creator": "creator", "founder": "creator", "invented_by": "creator",
    "author": "creator", "writer": "creator",
    "producer": "producer", "produced_by": "producer",
    # attr slot aliases
    "birthplace": "birthplace", "place_of_birth": "birthplace",
    "born_in": "birthplace", "birth_place": "birthplace",
    "death_date": "death_date", "date_of_death": "death_date",
    "died_on": "death_date", "death_year": "death_date",
    "birth_date": "birth_date", "date_of_birth": "birth_date",
    "born_on": "birth_date", "birth_year": "birth_date",
    "nationality": "nationality", "citizen_of": "nationality", "country": "nationality",
    "education": "education", "alma_mater": "education", "studied_at": "education",
    "release_date": "release_date", "released": "release_date",
    "release_year": "release_date", "year": "release_date",
    "earlier_date": "release_date",
    "known_for": "known_for", "famous_for": "known_for", "notable_for": "known_for",
    "location": "location", "located_in": "location", "based_in": "location",
    "identity_comparison": "identity_comparison", "same_person": "identity_comparison",
    # minimal alias patch
    "grandfather": "parent", "grandmother": "parent",
    "paternal_grandfather": "parent", "maternal_grandfather": "parent",
    "paternal_grandmother": "parent", "maternal_grandmother": "parent",
    "county": "location",
    # explicit unsupported labels for closed-ontology parsing
    "unsupported": UNSUPPORTED_SLOT_FAMILY,
    "passed_by": UNSUPPORTED_SLOT_FAMILY,
    "prime_minister_when": UNSUPPORTED_SLOT_FAMILY,
    "triggering_event": UNSUPPORTED_SLOT_FAMILY,
    "participants": UNSUPPORTED_SLOT_FAMILY,
}

# Extended role slots for LLM-extracted slots
EXTENDED_ROLE_SLOT_MAP: Dict[str, List[str]] = {
    **ROLE_SLOT_MAP,
    "performer": ["performer", "singer", "artist", "performed by", "sung by", "star", "starring"],
    "composer": ["composer", "composed by", "music by", "score by", "written by"],
    "creator": ["creator", "created by", "founded by", "invented by", "author", "writer", "written by"],
    "producer": ["producer", "produced by"],
}

EXTENDED_ATTR_SLOT_MAP: Dict[str, List[str]] = {
    **ATTR_SLOT_MAP,
    "release_date": ["release date", "released", "release year", "came out", "published"],
    "known_for": ["known for", "famous for", "notable for", "recognized for"],
    "location": ["located in", "based in", "situated in", "found in"],
}

ROLE_SLOT_FAMILIES = frozenset(EXTENDED_ROLE_SLOT_MAP)

ALL_SUPPORTED_SLOTS = frozenset(
    set(ROLE_SLOT_MAP) | set(ATTR_SLOT_MAP)
    | set(EXTENDED_ROLE_SLOT_MAP) | set(EXTENDED_ATTR_SLOT_MAP)
    | {"identity_comparison", UNSUPPORTED_SLOT_FAMILY}
)

CLOSED_ONTOLOGY_SLOT_FAMILIES = (
    "director",
    "spouse",
    "parent",
    "award_recipient",
    "performer",
    "composer",
    "creator",
    "producer",
    "birthplace",
    "death_date",
    "birth_date",
    "nationality",
    "education",
    "release_date",
    "known_for",
    "location",
    "identity_comparison",
    UNSUPPORTED_SLOT_FAMILY,
)


def canonicalize_slot_family(raw_slot: str) -> str | None:
    """Map LLM-output slot to controlled schema. Returns None if unmappable."""
    normalized = raw_slot.strip().lower().replace(" ", "_")
    if normalized in SLOT_CANONICALIZATION_MAP:
        return SLOT_CANONICALIZATION_MAP[normalized]
    # Try without underscores
    no_underscore = normalized.replace("_", " ")
    for canonical, aliases in {**EXTENDED_ROLE_SLOT_MAP, **EXTENDED_ATTR_SLOT_MAP}.items():
        if no_underscore in [a.lower() for a in aliases]:
            return canonical
    # Check if it's directly a supported slot
    if normalized in ALL_SUPPORTED_SLOTS:
        return normalized
    return None


def slot_family_lexical_cues_extended(slot_family: str) -> List[str]:
    """Get lexical cues for both original and extended slot maps."""
    if slot_family in EXTENDED_ROLE_SLOT_MAP:
        return list(EXTENDED_ROLE_SLOT_MAP[slot_family])
    if slot_family in EXTENDED_ATTR_SLOT_MAP:
        return list(EXTENDED_ATTR_SLOT_MAP[slot_family])
    if slot_family in ROLE_SLOT_MAP:
        return list(ROLE_SLOT_MAP[slot_family])
    return list(ATTR_SLOT_MAP.get(slot_family, []))


def normalize_query_text(text: str) -> str:
    return normalize_entity_text(text)


def slot_family_lexical_cues(slot_family: str) -> List[str]:
    if slot_family in ROLE_SLOT_MAP:
        return list(ROLE_SLOT_MAP[slot_family])
    return list(ATTR_SLOT_MAP.get(slot_family, []))


def is_role_slot(slot_family: str) -> bool:
    return slot_family in ROLE_SLOT_FAMILIES


def is_single_valued_slot(slot_family: str) -> bool:
    return slot_family in SINGLE_VALUED_SLOT_FAMILIES


def infer_expected_answer_type(query: str, slot_family: str) -> str:
    normalized_query = normalize_query_text(query)
    if normalized_query.startswith("who "):
        return "person"
    if normalized_query.startswith("where "):
        return "location"
    if normalized_query.startswith("when "):
        return "date"
    if normalized_query.startswith("which film") or normalized_query.startswith("which movie"):
        return "work"
    return SLOT_EXPECTED_ANSWER_TYPE.get(slot_family, "unknown")


def detect_comparator(query: str) -> str | None:
    normalized_query = normalize_query_text(query)
    comparator_cues = {
        "older": ["older", "oldest"],
        "younger": ["younger", "youngest"],
        "later": ["later", "latest", "after"],
        "earlier": ["earlier", "before", "first"],
    }
    for comparator, cues in comparator_cues.items():
        if any(cue in normalized_query for cue in cues):
            return comparator
    return None


def normalize_requirement_filler(value: str) -> str:
    return normalize_entity_text(value)


def fillers_equivalent(left: str, right: str) -> bool:
    normalized_left = normalize_requirement_filler(left)
    normalized_right = normalize_requirement_filler(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    return normalized_left in normalized_right or normalized_right in normalized_left


def _ranked_unique_entities(*groups: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for group in groups:
        for entity in group:
            normalized = normalize_entity_text(entity)
            if not normalized or normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def _select_anchor_entities(
    seed_entities: Sequence[str],
    query_entities: Sequence[str],
    baseline_titles: Sequence[str],
    comparator: str | None,
) -> List[str]:
    anchors = _ranked_unique_entities(seed_entities, query_entities)
    if not anchors:
        anchors = _ranked_unique_entities(baseline_titles)
    if comparator:
        return anchors[:2]
    return anchors[:1]


def _detect_slot_families(query: str) -> Tuple[List[str], List[str]]:
    normalized_query = normalize_query_text(query)
    role_slots: List[Tuple[int, str]] = []
    attr_slots: List[Tuple[int, str]] = []
    for slot_family, cues in ROLE_SLOT_MAP.items():
        positions = [normalized_query.find(cue) for cue in cues if normalized_query.find(cue) >= 0]
        if positions:
            role_slots.append((min(positions), slot_family))
    for slot_family, cues in ATTR_SLOT_MAP.items():
        positions = [normalized_query.find(cue) for cue in cues if normalized_query.find(cue) >= 0]
        if positions:
            attr_slots.append((min(positions), slot_family))
    role_slots.sort()
    attr_slots.sort()
    return [slot for _, slot in role_slots], [slot for _, slot in attr_slots]


def serialize_requirement_units(units: Sequence[RequirementUnit]) -> List[Dict[str, object]]:
    return [asdict(unit) for unit in units]


def build_requirement_probe_text(query: str, unit: RequirementUnit) -> str:
    slot_text = unit.slot_family.replace("_", " ").strip() or "attribute"
    anchor_text = ", ".join(
        normalize_entity_text(anchor)
        for anchor in unit.anchor_entities
        if normalize_entity_text(anchor)
    ).strip()
    if not anchor_text and unit.bridge_targets:
        anchor_text = "the bridge entity resolved by prior support"
    if not anchor_text:
        anchor_text = "the target entity implied by the question"

    cue_text = ", ".join(
        cue.strip()
        for cue in unit.lexical_cues
        if cue and str(cue).strip()
    )
    if not cue_text:
        cue_text = slot_text

    expected_type = str(unit.expected_answer_type or "unknown").strip() or "unknown"
    tier_text = str(unit.tier or "core").strip() or "core"
    return (
        f"Question: {str(query or '').strip()}\n"
        f"Requirement tier: {tier_text}.\n"
        f"Need evidence for the {slot_text} of {anchor_text}.\n"
        f"Expected answer type: {expected_type}.\n"
        f"Lexical cues: {cue_text}."
    )


def extract_requirement_units(
    query: str,
    seed_entities: List[str],
    query_entities: List[str],
    baseline_titles: List[str],
    max_units: int = 4,
) -> tuple[List[RequirementUnit], Dict[str, object]]:
    comparator = detect_comparator(query)
    role_slots, attr_slots = _detect_slot_families(query)
    anchors = _select_anchor_entities(seed_entities, query_entities, baseline_titles, comparator)
    fallback_reason: str | None = None
    units: List[RequirementUnit] = []

    if not anchors:
        fallback_reason = "no_anchor"
    elif not role_slots and not attr_slots:
        fallback_reason = "no_slot_family"
    else:
        anchor_limit = 2 if comparator else 1
        chosen_anchors = anchors[: max(anchor_limit, 1)]
        support_unit_ids: List[str] = []
        if role_slots:
            role_slot = role_slots[0]
            support_tier = "support" if attr_slots else "core"
            for anchor in chosen_anchors:
                unit_id = f"{support_tier}-{role_slot}-{anchor}"
                units.append(
                    RequirementUnit(
                        unit_id=unit_id,
                        tier=support_tier,
                        anchor_entities=[anchor],
                        slot_family=role_slot,
                        expected_answer_type=infer_expected_answer_type(query, role_slot),
                        lexical_cues=slot_family_lexical_cues(role_slot),
                        is_single_valued=is_single_valued_slot(role_slot),
                        comparator=comparator,
                    )
                )
                if support_tier == "support":
                    support_unit_ids.append(unit_id)
        if attr_slots:
            attr_slot = attr_slots[0]
            units.append(
                RequirementUnit(
                    unit_id=f"core-{attr_slot}-{'-'.join(chosen_anchors) if chosen_anchors else 'query'}",
                    tier="core",
                    anchor_entities=list(chosen_anchors),
                    slot_family=attr_slot,
                    expected_answer_type=infer_expected_answer_type(query, attr_slot),
                    bridge_targets=list(support_unit_ids),
                    lexical_cues=slot_family_lexical_cues(attr_slot),
                    is_single_valued=is_single_valued_slot(attr_slot),
                    comparator=comparator,
                )
            )

    truncated = False
    if len(units) > max(int(max_units), 0):
        units = units[: max(int(max_units), 0)]
        truncated = True

    extractor_trace = {
        "extractor_fallback": fallback_reason is not None,
        "extractor_fallback_reason": fallback_reason,
        "extractor_mode": "rule",
        "requirement_unit_count": int(len(units)),
        "core_unit_count": int(sum(1 for unit in units if unit.tier == "core")),
        "support_unit_count": int(sum(1 for unit in units if unit.tier == "support")),
        "empty_requirement_query": bool(len(units) == 0),
        "slot_families": [unit.slot_family for unit in units],
        "anchors": anchors,
        "comparator": comparator,
        "truncated": bool(truncated),
    }
    return units, extractor_trace


# ---------------------------------------------------------------------------
# LLM-based requirement extraction
# ---------------------------------------------------------------------------

RAS_EXTRACTION_PROMPT = """\
You are a query structure parser for multi-hop question answering.

Given a question, decompose it into 1-4 requirement units.
Each unit has these fields:
- tier: "support" (intermediate bridge entity) or "core" (final answer attribute)
- anchor: the entity from the question this unit is about
- slot_family: the relation or attribute being asked (e.g. director, birthplace, death_date)
- expected_answer_type: person / location / date / organization / work / boolean / number / unknown
- is_single_valued: true if only one answer is expected for this slot

Rules:
- "support" = intermediate entity needed to reach the answer (bridge entity)
- "core" = the final information the question asks for
- Only extract structure visible in the question text. Do NOT use world knowledge to resolve entities.
- Simple questions (single entity + single attribute): one "core" unit.
- Nested questions (e.g. "birthplace of the director of X"): support + core.
- Comparison questions ("which is older, A or B"): parallel support + core units.
- For bridge references use angle brackets like <director> to indicate unresolved references.

Output a JSON array only, no explanation.

Examples:

Question: "Where was the director of Inception born?"
[
  {"tier": "support", "anchor": "Inception", "slot_family": "director", "expected_answer_type": "person", "is_single_valued": true},
  {"tier": "core", "anchor": "<director>", "slot_family": "birthplace", "expected_answer_type": "location", "is_single_valued": true}
]

Question: "Which film has the director who died later, A or B?"
[
  {"tier": "support", "anchor": "A", "slot_family": "director", "expected_answer_type": "person", "is_single_valued": true},
  {"tier": "support", "anchor": "B", "slot_family": "director", "expected_answer_type": "person", "is_single_valued": true},
  {"tier": "core", "anchor": "<director_of_A>", "slot_family": "death_date", "expected_answer_type": "date", "is_single_valued": true},
  {"tier": "core", "anchor": "<director_of_B>", "slot_family": "death_date", "expected_answer_type": "date", "is_single_valued": true}
]

Question: "When did Lothair II's mother die?"
[
  {"tier": "support", "anchor": "Lothair II", "slot_family": "mother", "expected_answer_type": "person", "is_single_valued": true},
  {"tier": "core", "anchor": "<mother>", "slot_family": "death_date", "expected_answer_type": "date", "is_single_valued": true}
]

Question: "Are the directors of film A and film B from the same country?"
[
  {"tier": "support", "anchor": "film A", "slot_family": "director", "expected_answer_type": "person", "is_single_valued": true},
  {"tier": "support", "anchor": "film B", "slot_family": "director", "expected_answer_type": "person", "is_single_valued": true},
  {"tier": "core", "anchor": "<directors>", "slot_family": "nationality", "expected_answer_type": "boolean", "is_single_valued": true}
]"""

RAS_CLOSED_ONTOLOGY_EXTRACTION_PROMPT = f"""\
You are a query structure parser for multi-hop question answering.

Decompose the question into 1-4 requirement units. Output a JSON array only.

Each unit must use this schema:
- tier: "support" or "core"
- anchor: entity text from the question, or a bridge reference like <director> or <mother>
- slot_family: one value from this closed ontology only: {", ".join(CLOSED_ONTOLOGY_SLOT_FAMILIES)}
- raw_slot_text: the original fine-grained relation phrase from the question
- expected_answer_type: person / location / date / organization / work / boolean / number / unknown
- is_single_valued: true or false
- bridge_targets: optional list of bridge labels such as ["director"] or ["director_of_a"]

Rules:
- Do not invent a new slot_family outside the closed ontology.
- If the relation is outside the ontology, set slot_family to "{UNSUPPORTED_SLOT_FAMILY}" and preserve the original phrase in raw_slot_text.
- Use "parent" for family-link queries that can be naturally collapsed into the parent family, including mother, father, grandfather, grandmother, paternal_grandfather, and maternal_grandfather.
- Use "location" for location-container queries such as county.
- Only extract structure visible in the question text. Do NOT use world knowledge to resolve entities.
- Preserve support/core structure. Core units that depend on a bridge entity should include bridge_targets.

Examples:

Question: "When did Lothair II's mother die?"
[
  {{"tier": "support", "anchor": "Lothair II", "slot_family": "parent", "raw_slot_text": "mother", "expected_answer_type": "person", "is_single_valued": true}},
  {{"tier": "core", "anchor": "<mother>", "slot_family": "death_date", "raw_slot_text": "die", "expected_answer_type": "date", "is_single_valued": true, "bridge_targets": ["mother"]}}
]

Question: "Who is Raghnall Mac Ruaidhrí's paternal grandfather?"
[
  {{"tier": "core", "anchor": "Raghnall Mac Ruaidhrí", "slot_family": "parent", "raw_slot_text": "paternal_grandfather", "expected_answer_type": "person", "is_single_valued": true}}
]

Question: "Which county are Marufabad and Nasamkhrali in?"
[
  {{"tier": "core", "anchor": "Marufabad", "slot_family": "location", "raw_slot_text": "county", "expected_answer_type": "location", "is_single_valued": true}},
  {{"tier": "core", "anchor": "Nasamkhrali", "slot_family": "location", "raw_slot_text": "county", "expected_answer_type": "location", "is_single_valued": true}}
]

Question: "The Distribution of Industry act was passed by a man who was prime minister when?"
[
  {{"tier": "support", "anchor": "The Distribution of Industry act", "slot_family": "{UNSUPPORTED_SLOT_FAMILY}", "raw_slot_text": "passed_by", "expected_answer_type": "person", "is_single_valued": true}},
  {{"tier": "core", "anchor": "<passed_by>", "slot_family": "{UNSUPPORTED_SLOT_FAMILY}", "raw_slot_text": "prime_minister_when", "expected_answer_type": "date", "is_single_valued": true, "bridge_targets": ["passed_by"]}}
]
"""

_ras_llm_client = None
_ras_llm_model = None


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    try:
        parsed = urlparse(str(base_url))
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _get_ras_llm_client():
    global _ras_llm_client, _ras_llm_model
    if _ras_llm_client is not None:
        return _ras_llm_client, _ras_llm_model
    try:
        import os
        import httpx
        from openai import OpenAI
        base_url = os.environ.get("RAS_LLM_BASE_URL", "http://localhost:8039/v1")
        _ras_llm_model = os.environ.get("RAS_LLM_MODEL", "qwen3-8b")
        http_client = httpx.Client(trust_env=not _is_local_base_url(base_url))
        _ras_llm_client = OpenAI(
            base_url=base_url,
            api_key="sk-",
            http_client=http_client,
        )
    except Exception as e:
        logger.warning("RAS LLM client init failed: %s", e)
        return None, None
    return _ras_llm_client, _ras_llm_model


def _normalize_bridge_ref_label(anchor_raw: str) -> str:
    raw = str(anchor_raw or "").strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    raw = raw.replace("_", " ")
    return normalize_entity_text(raw)


def _expand_bridge_aliases(text: str) -> set[str]:
    normalized = normalize_entity_text(str(text or "").replace("_", " "))
    if not normalized:
        return set()
    aliases = {normalized}
    if len(normalized) > 3:
        if normalized.endswith("s"):
            aliases.add(normalized[:-1])
        else:
            aliases.add(f"{normalized}s")
    return {alias for alias in aliases if alias}


def _ground_anchor_to_entities(
    anchor: str,
    seed_entities: Sequence[str],
    query_entities: Sequence[str],
    baseline_titles: Sequence[str],
) -> str | None:
    """Match LLM-extracted anchor to a known entity.
    Returns None if no confident match — the unit must NOT activate."""
    norm_anchor = normalize_entity_text(anchor)
    if not norm_anchor:
        return None
    if norm_anchor.startswith("<"):
        return None

    all_entities = list(seed_entities) + list(query_entities)

    # Pass 1: exact match against entities
    for entity in all_entities:
        if normalize_entity_text(entity) == norm_anchor:
            return normalize_entity_text(entity)

    # Pass 2: exact match against titles
    for title in baseline_titles:
        norm_title = normalize_entity_text(title)
        if norm_title and norm_anchor == norm_title:
            return norm_title

    # Pass 3: substring match — but require minimum quality
    # Reject if anchor is too short (< 4 chars) to avoid "changed", "blood" etc.
    if len(norm_anchor) < 4:
        return None

    for entity in all_entities:
        norm_entity = normalize_entity_text(entity)
        if not norm_entity:
            continue
        if _is_quality_substring_match(norm_anchor, norm_entity):
            return norm_entity

    for title in baseline_titles:
        norm_title = normalize_entity_text(title)
        if not norm_title:
            continue
        if _is_quality_substring_match(norm_anchor, norm_title):
            return norm_title

    return None


def _coerce_text_list(value: object) -> List[str]:
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple)):
        items: List[str] = []
        for item in value:
            normalized = str(item or "").strip()
            if normalized:
                items.append(normalized)
        return items
    return []


def _build_bridge_target_aliases(unit: RequirementUnit) -> set[str]:
    aliases: set[str] = set()
    slot_labels = [unit.slot_family]
    if unit.raw_slot_family:
        slot_labels.append(unit.raw_slot_family)
    anchor_labels = list(unit.anchor_entities)
    if unit.raw_anchor:
        anchor_labels.append(unit.raw_anchor)

    for slot_label in slot_labels:
        slot_aliases = _expand_bridge_aliases(slot_label)
        aliases.update(slot_aliases)
        for anchor_label in anchor_labels:
            normalized_anchor = normalize_entity_text(str(anchor_label).replace("_", " "))
            if not normalized_anchor:
                continue
            for slot_alias in slot_aliases:
                aliases.add(normalize_entity_text(f"{slot_alias} of {normalized_anchor}"))
    return {alias for alias in aliases if alias}


def _match_bridge_units(
    requested_labels: Sequence[str],
    candidate_units: Sequence[RequirementUnit],
) -> List[RequirementUnit]:
    requested_aliases: set[str] = set()
    for label in requested_labels:
        requested_aliases.update(_expand_bridge_aliases(label))
    if not requested_aliases:
        return []

    matched: List[RequirementUnit] = []
    seen_ids: set[str] = set()
    for unit in candidate_units:
        aliases = _build_bridge_target_aliases(unit)
        if requested_aliases.intersection(aliases) and unit.unit_id not in seen_ids:
            matched.append(unit)
            seen_ids.add(unit.unit_id)
    return matched


def _resolve_bridge_targets(
    *,
    bridge_ref_label: str,
    explicit_target_labels: Sequence[str],
    active_units: List[RequirementUnit],
    grounding_stats: Dict[str, object],
) -> tuple[List[str], str | None]:
    requested_labels = [label for label in explicit_target_labels if str(label or "").strip()]
    if bridge_ref_label:
        requested_labels.append(bridge_ref_label)
    if not requested_labels:
        return [], "missing_bridge_ref"

    support_units = [unit for unit in active_units if unit.tier == "support"]
    support_matches = _match_bridge_units(requested_labels, support_units)
    if support_matches:
        return [unit.unit_id for unit in support_matches], None

    all_matches = _match_bridge_units(requested_labels, active_units)
    if not all_matches:
        return [], "unresolved_bridge_target"

    repaired_ids: List[str] = []
    for unit in all_matches:
        if unit.tier != "support":
            unit.tier = "support"
            if unit.unit_id.startswith("core-"):
                unit.unit_id = unit.unit_id.replace("core-", "support-", 1)
            grounding_stats["tier_repaired_count"] = int(grounding_stats.get("tier_repaired_count", 0)) + 1
        repaired_ids.append(unit.unit_id)
    return repaired_ids, None


def _is_quality_substring_match(anchor: str, target: str) -> bool:
    """Substring match with quality checks to avoid degrading multi-word to short fragment."""
    if anchor == target:
        return True
    # anchor must be contained in target or vice versa
    if anchor not in target and target not in anchor:
        return False
    shorter = anchor if len(anchor) <= len(target) else target
    longer = anchor if len(anchor) > len(target) else target
    # Require the shorter string covers at least 50% of the longer string
    # This prevents "changed" matching "song changed it" (6/18 = 33%)
    ratio = len(shorter) / len(longer)
    if ratio < 0.5:
        return False
    # Reject if shorter is a single common word (< 6 chars) matching a much longer entity
    if len(shorter) < 6 and len(longer) > len(shorter) + 5:
        return False
    return True


def _parsed_to_requirement_units(
    parsed: list,
    seed_entities: List[str],
    query_entities: List[str],
    baseline_titles: List[str],
    max_units: int,
) -> Tuple[List[RequirementUnit], Dict[str, object]]:
    units: List[RequirementUnit] = []
    grounding_stats = {
        "total_parsed_items": len(parsed),
        "parsed_unit_count": 0,
        "active_unit_count": 0,
        "inactive_unit_count": 0,
        "lexical_anchor_candidate_count": 0,
        "resolved_anchor_count": 0,
        "grounded_count": 0,
        "ungrounded_count": 0,
        "bridge_ref_candidate_count": 0,
        "bridge_ref_resolved_count": 0,
        "bridge_ref_count": 0,
        "skipped_non_object": 0,
        "skipped_no_field": 0,
        "skipped_unknown_slot": 0,
        "unsupported_slot_count": 0,
        "tier_repaired_count": 0,
        "inactive_reason_distribution": {},
        "unsupported_raw_slots": [],
    }
    inactive_units: List[Dict[str, object]] = []

    def record_inactive(
        *,
        reason: str,
        tier: str,
        raw_anchor: str,
        raw_slot_family: str,
        canonical_slot_family: str | None,
        bridge_ref_label: str | None = None,
        explicit_bridge_targets: Sequence[str] | None = None,
    ) -> None:
        grounding_stats["inactive_unit_count"] = int(grounding_stats["inactive_unit_count"]) + 1
        distribution = grounding_stats["inactive_reason_distribution"]
        distribution[reason] = int(distribution.get(reason, 0)) + 1
        inactive_units.append(
            {
                "reason": reason,
                "tier": tier,
                "raw_anchor": raw_anchor,
                "raw_slot_family": raw_slot_family,
                "canonical_slot_family": canonical_slot_family,
                "bridge_ref_label": bridge_ref_label,
                "explicit_bridge_targets": list(explicit_bridge_targets or []),
            }
        )

    for item in parsed[:max(max_units * 2, 8)]:
        if not isinstance(item, dict):
            grounding_stats["skipped_non_object"] += 1
            continue
        tier = str(item.get("tier", "core")).strip().lower()
        if tier not in ("core", "support"):
            tier = "core"
        anchor_raw = str(item.get("anchor", "")).strip()
        parser_slot = str(item.get("slot_family", "")).strip().lower()
        raw_slot_text = str(item.get("raw_slot_text", parser_slot or "")).strip().lower()
        answer_type = str(item.get("expected_answer_type", "unknown")).strip().lower()
        single_valued = bool(item.get("is_single_valued", True))
        explicit_bridge_targets = _coerce_text_list(item.get("bridge_targets"))

        if not anchor_raw or not parser_slot:
            grounding_stats["skipped_no_field"] += 1
            continue
        grounding_stats["parsed_unit_count"] = int(grounding_stats["parsed_unit_count"]) + 1

        # Slot canonicalization: map to controlled schema or reject
        canonical_slot = canonicalize_slot_family(parser_slot)
        if canonical_slot is None:
            grounding_stats["skipped_unknown_slot"] += 1
            record_inactive(
                reason="unknown_slot",
                tier=tier,
                raw_anchor=anchor_raw,
                raw_slot_family=raw_slot_text or parser_slot,
                canonical_slot_family=None,
                explicit_bridge_targets=explicit_bridge_targets,
            )
            continue
        if canonical_slot == UNSUPPORTED_SLOT_FAMILY:
            grounding_stats["unsupported_slot_count"] = int(grounding_stats["unsupported_slot_count"]) + 1
            unsupported_raw_slots = grounding_stats["unsupported_raw_slots"]
            normalized_unsupported = normalize_entity_text(raw_slot_text or parser_slot)
            if normalized_unsupported and normalized_unsupported not in unsupported_raw_slots:
                unsupported_raw_slots.append(normalized_unsupported)
            record_inactive(
                reason="unsupported_slot",
                tier=tier,
                raw_anchor=anchor_raw,
                raw_slot_family=raw_slot_text or parser_slot,
                canonical_slot_family=canonical_slot,
                explicit_bridge_targets=explicit_bridge_targets,
            )
            continue
        slot_family = canonical_slot

        is_bridge_ref = anchor_raw.startswith("<") and anchor_raw.endswith(">")
        if is_bridge_ref:
            grounding_stats["bridge_ref_candidate_count"] += 1
            grounding_stats["bridge_ref_count"] += 1
            bridge_ref_label = _normalize_bridge_ref_label(anchor_raw)
            anchor_entities: List[str] = []
            bridge_targets, bridge_error = _resolve_bridge_targets(
                bridge_ref_label=bridge_ref_label,
                explicit_target_labels=explicit_bridge_targets,
                active_units=units,
                grounding_stats=grounding_stats,
            )
            if bridge_error is not None or not bridge_targets:
                record_inactive(
                    reason=str(bridge_error or "unresolved_bridge_target"),
                    tier=tier,
                    raw_anchor=anchor_raw,
                    raw_slot_family=raw_slot_text or parser_slot,
                    canonical_slot_family=slot_family,
                    bridge_ref_label=bridge_ref_label,
                    explicit_bridge_targets=explicit_bridge_targets,
                )
                continue
            grounding_stats["bridge_ref_resolved_count"] = int(grounding_stats["bridge_ref_resolved_count"]) + 1
        else:
            grounding_stats["lexical_anchor_candidate_count"] += 1
            grounded = _ground_anchor_to_entities(
                anchor_raw, seed_entities, query_entities, baseline_titles
            )
            if grounded is None:
                grounding_stats["ungrounded_count"] += 1
                record_inactive(
                    reason="unresolved_anchor",
                    tier=tier,
                    raw_anchor=anchor_raw,
                    raw_slot_family=raw_slot_text or parser_slot,
                    canonical_slot_family=slot_family,
                )
                continue
            grounding_stats["resolved_anchor_count"] += 1
            grounding_stats["grounded_count"] += 1
            anchor_entities = [grounded]
            bridge_ref_label = None
            bridge_targets = []

        cues = slot_family_lexical_cues_extended(slot_family)
        if not cues:
            cues = [slot_family.replace("_", " ")]
        raw_slot_cue = (raw_slot_text or parser_slot).replace("_", " ").strip()
        if raw_slot_cue and raw_slot_cue not in cues:
            cues.append(raw_slot_cue)

        anchor_key = ""
        if anchor_entities:
            anchor_key = normalize_entity_text(anchor_entities[0])
        elif bridge_ref_label:
            anchor_key = bridge_ref_label
        else:
            anchor_key = normalize_entity_text(anchor_raw) or anchor_raw

        units.append(
            RequirementUnit(
                unit_id=f"{tier}-{slot_family}-{anchor_key or 'query'}",
                tier=tier,
                anchor_entities=anchor_entities,
                slot_family=slot_family,
                expected_answer_type=answer_type,
                bridge_targets=bridge_targets,
                lexical_cues=cues,
                is_single_valued=single_valued,
                raw_slot_family=raw_slot_text or parser_slot,
                raw_anchor=anchor_raw,
                anchor_status="bridge_ref" if is_bridge_ref else "grounded",
                bridge_ref_label=bridge_ref_label,
            )
        )
        grounding_stats["active_unit_count"] = int(grounding_stats["active_unit_count"]) + 1
        if len(units) >= max_units:
            break

    lexical_count = int(grounding_stats["lexical_anchor_candidate_count"])
    bridge_count = int(grounding_stats["bridge_ref_candidate_count"])
    resolved_anchor_count = int(grounding_stats["resolved_anchor_count"])
    bridge_ref_resolved_count = int(grounding_stats["bridge_ref_resolved_count"])
    grounding_stats["resolved_anchor_rate"] = (
        float(resolved_anchor_count) / float(lexical_count) if lexical_count else 0.0
    )
    grounding_stats["bridge_ref_resolved_rate"] = (
        float(bridge_ref_resolved_count) / float(bridge_count) if bridge_count else 0.0
    )
    grounding_stats["inactive_units"] = inactive_units
    return units, grounding_stats


def _strip_think_tags(text: str) -> str:
    return strip_reasoning_content(text)


def _extract_json_array(text: str) -> list | None:
    text = _strip_think_tags(text)
    json_start = text.find("[")
    json_end = text.rfind("]")
    if json_start < 0 or json_end < 0 or json_end <= json_start:
        return None
    try:
        return json.loads(text[json_start:json_end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _request_requirement_units_from_llm(
    query: str,
    system_prompt: str = RAS_EXTRACTION_PROMPT,
) -> Tuple[list | None, str, str | None]:
    client, model_name = _get_ras_llm_client()
    if client is None:
        return None, "", "llm_client_unavailable"

    user_msg = f'Question: "{query}"\n\nOutput JSON array only, no explanation. /no_think'
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        raw = str(response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("RAS LLM extraction failed: %s", e)
        return None, "", f"llm_error:{type(e).__name__}"

    parsed = _extract_json_array(raw)
    if parsed is None:
        return None, raw, "json_parse_fail"
    return parsed, raw, None


def _build_llm_extractor_trace(
    *,
    query: str,
    units: Sequence[RequirementUnit],
    raw_output: str,
    grounding_stats: Dict[str, object],
    extractor_mode: str,
    fallback_reason: str | None = None,
) -> Dict[str, object]:
    inactive_units = list(grounding_stats.get("inactive_units", []))
    return {
        "extractor_mode": extractor_mode,
        "extractor_status": "parsed_active" if units else "empty_after_grounding",
        "extractor_fallback": False,
        "extractor_fallback_reason": fallback_reason,
        "requirement_unit_count": len(units),
        "active_requirement_unit_count": len(units),
        "parsed_unit_count": int(grounding_stats.get("parsed_unit_count", 0)),
        "inactive_unit_count": int(grounding_stats.get("inactive_unit_count", 0)),
        "core_unit_count": sum(1 for u in units if u.tier == "core"),
        "support_unit_count": sum(1 for u in units if u.tier == "support"),
        "empty_requirement_query": len(units) == 0,
        "slot_families": [u.slot_family for u in units],
        "anchors": [e for u in units for e in u.anchor_entities],
        "comparator": detect_comparator(query),
        "truncated": False,
        "grounding_stats": dict(grounding_stats),
        "resolved_anchor_count": int(grounding_stats.get("resolved_anchor_count", 0)),
        "lexical_anchor_candidate_count": int(grounding_stats.get("lexical_anchor_candidate_count", 0)),
        "resolved_anchor_rate": float(grounding_stats.get("resolved_anchor_rate", 0.0) or 0.0),
        "bridge_ref_candidate_count": int(grounding_stats.get("bridge_ref_candidate_count", 0)),
        "bridge_ref_resolved_count": int(grounding_stats.get("bridge_ref_resolved_count", 0)),
        "bridge_ref_resolved_rate": float(grounding_stats.get("bridge_ref_resolved_rate", 0.0) or 0.0),
        "unsupported_slot_count": int(grounding_stats.get("unsupported_slot_count", 0)),
        "unsupported_raw_slots": list(grounding_stats.get("unsupported_raw_slots", [])),
        "skipped_unknown_slot": int(grounding_stats.get("skipped_unknown_slot", 0)),
        "inactive_reason_distribution": dict(grounding_stats.get("inactive_reason_distribution", {})),
        "inactive_units": inactive_units[:8],
        "raw_llm_output": raw_output[:500],
    }


def extract_requirement_units_llm(
    query: str,
    seed_entities: List[str],
    query_entities: List[str],
    baseline_titles: List[str],
    max_units: int = 4,
) -> Tuple[List[RequirementUnit], Dict[str, object]]:
    parsed, raw, error_reason = _request_requirement_units_from_llm(query)
    if error_reason is not None:
        return _fallback_to_rule_based(
            query, seed_entities, query_entities, baseline_titles,
            max_units, error_reason, "llm",
        )

    units, grounding_stats = _parsed_to_requirement_units(
        parsed or [], seed_entities, query_entities, baseline_titles, max_units,
    )
    trace = _build_llm_extractor_trace(
        query=query,
        units=units,
        raw_output=raw,
        grounding_stats=grounding_stats,
        extractor_mode="llm",
    )
    if not units:
        return _fallback_to_rule_based(
            query, seed_entities, query_entities, baseline_titles,
            max_units, "empty_after_grounding", "llm",
        )
    return units, trace


def extract_requirement_units_llm_grounded(
    query: str,
    seed_entities: List[str],
    query_entities: List[str],
    baseline_titles: List[str],
    max_units: int = 4,
) -> Tuple[List[RequirementUnit], Dict[str, object]]:
    parsed, raw, error_reason = _request_requirement_units_from_llm(query)
    if error_reason is not None:
        return _fallback_to_rule_based(
            query, seed_entities, query_entities, baseline_titles,
            max_units, error_reason, "llm_grounded",
        )

    units, grounding_stats = _parsed_to_requirement_units(
        parsed or [], seed_entities, query_entities, baseline_titles, max_units,
    )
    trace = _build_llm_extractor_trace(
        query=query,
        units=units,
        raw_output=raw,
        grounding_stats=grounding_stats,
        extractor_mode="llm_grounded",
    )
    return units, trace


def extract_requirement_units_llm_closed_grounded(
    query: str,
    seed_entities: List[str],
    query_entities: List[str],
    baseline_titles: List[str],
    max_units: int = 4,
) -> Tuple[List[RequirementUnit], Dict[str, object]]:
    parsed, raw, error_reason = _request_requirement_units_from_llm(
        query,
        system_prompt=RAS_CLOSED_ONTOLOGY_EXTRACTION_PROMPT,
    )
    if error_reason is not None:
        return _fallback_to_rule_based(
            query, seed_entities, query_entities, baseline_titles,
            max_units, error_reason, "llm_closed_grounded",
        )

    units, grounding_stats = _parsed_to_requirement_units(
        parsed or [], seed_entities, query_entities, baseline_titles, max_units,
    )
    trace = _build_llm_extractor_trace(
        query=query,
        units=units,
        raw_output=raw,
        grounding_stats=grounding_stats,
        extractor_mode="llm_closed_grounded",
    )
    trace["closed_ontology_slot_families"] = list(CLOSED_ONTOLOGY_SLOT_FAMILIES)
    return units, trace


def _fallback_to_rule_based(
    query: str,
    seed_entities: List[str],
    query_entities: List[str],
    baseline_titles: List[str],
    max_units: int,
    llm_fallback_reason: str,
    requested_mode: str,
) -> Tuple[List[RequirementUnit], Dict[str, object]]:
    units, trace = extract_requirement_units(
        query=query,
        seed_entities=seed_entities,
        query_entities=query_entities,
        baseline_titles=baseline_titles,
        max_units=max_units,
    )
    trace["llm_fallback_reason"] = llm_fallback_reason
    trace["requested_extractor_mode"] = requested_mode
    trace["extractor_mode"] = f"rule_fallback_from_{requested_mode}"
    trace["extractor_status"] = "rule_fallback"
    trace["extractor_fallback"] = True
    trace["extractor_fallback_reason"] = llm_fallback_reason
    return units, trace
