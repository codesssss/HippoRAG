from build_query_obligation_units import build_query_obligation_units
from match_query_demands_to_evidence_units import (
    predicate_tokens_for_obligation,
    source_grounded_matches_for_obligation,
)


def _obligation(triple):
    return build_query_obligation_units(query="q", query_triples=[triple])[0]


def test_predicate_tokens_use_raw_relation_without_schema_alias():
    obligation = _obligation(["Ada Lovelace", "author", "?work"])

    assert predicate_tokens_for_obligation(obligation) == {"author"}


def test_source_grounded_match_uses_anchor_predicate_and_explicit_variable_mention():
    obligation = _obligation(["Lionel Messi", "signed for", "?team"])
    unit = {
        "unit_id": "u1",
        "doc_index": 7,
        "title": "Lionel Messi",
        "span_text": "Lionel Messi signed for Barcelona in 2000.",
        "mention_surfaces": ["Lionel Messi", "Barcelona", "2000"],
        "predicate_text": "",
        "argument_texts": [],
        "source": "sentence",
    }

    matches = source_grounded_matches_for_obligation(
        obligation=obligation,
        evidence_units=[unit],
        variable_values={},
    )

    assert len(matches) == 1
    assert matches[0]["doc_index"] == 7
    assert matches[0]["relation_token_hits"] == ["sign"]
    assert matches[0]["subject_anchor_hits"]
    assert matches[0]["variable_bindings"]["team"] == ["barcelona"]


def test_source_grounded_match_rejects_missing_predicate_token():
    obligation = _obligation(["Lionel Messi", "signed for", "?team"])
    unit = {
        "unit_id": "u1",
        "doc_index": 7,
        "title": "Lionel Messi",
        "span_text": "Lionel Messi played in Spain.",
        "mention_surfaces": ["Lionel Messi", "Spain"],
        "predicate_text": "",
        "argument_texts": [],
        "source": "sentence",
    }

    matches = source_grounded_matches_for_obligation(
        obligation=obligation,
        evidence_units=[unit],
        variable_values={},
    )

    assert matches == []


def test_source_grounded_match_requires_visible_bound_anchor():
    obligation = _obligation(["Cristiano Ronaldo", "signed for", "?team"])
    unit = {
        "unit_id": "u1",
        "doc_index": 7,
        "title": "Lionel Messi",
        "span_text": "Lionel Messi signed for Barcelona in 2000.",
        "mention_surfaces": ["Lionel Messi", "Barcelona", "2000"],
        "predicate_text": "",
        "argument_texts": [],
        "source": "sentence",
    }

    matches = source_grounded_matches_for_obligation(
        obligation=obligation,
        evidence_units=[unit],
        variable_values={},
    )

    assert matches == []
