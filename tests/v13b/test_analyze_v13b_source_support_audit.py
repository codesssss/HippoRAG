from analyze_v13b_source_support_audit import (
    classify_source_support_status,
    source_sentences_for_docs,
    source_support_candidates_for_obligation,
)
from build_query_obligation_units import build_query_obligation_units


def _obligation(triple):
    return build_query_obligation_units(query="q", query_triples=[triple])[0]


def test_source_sentences_for_docs_preserves_doc_title_and_sentence():
    rows = source_sentences_for_docs(
        doc_indices=[1],
        openie_docs=[
            {"passage": "Unused\nNothing."},
            {"passage": "Messi\nMessi signed for Barcelona in 2000."},
        ],
    )

    assert rows[0]["doc_index"] == 1
    assert rows[0]["title"] == "Messi"
    assert "signed for Barcelona" in rows[1]["sentence"]


def test_source_support_candidate_uses_anchor_and_predicate_tokens_without_relation_schema():
    obligation = _obligation(["Messi", "signed for", "?team"])
    rows = source_sentences_for_docs(
        doc_indices=[0],
        openie_docs=[{"passage": "Messi\nMessi signed for Barcelona in 2000."}],
    )

    candidates = source_support_candidates_for_obligation(
        obligation=obligation,
        source_sentences=rows,
        variable_values={},
    )

    assert candidates
    assert candidates[0]["relation_token_hits"] == ["sign"]
    assert candidates[0]["subject_sentence_anchor_hits"] == ["messi"]


def test_classify_prefers_exact_openie_over_source_support():
    obligation = _obligation(["Messi", "signed for", "?team"])
    exact = {obligation["obligation_id"]: [{"doc_index": 0}]}

    status = classify_source_support_status(
        obligation=obligation,
        exact_gold_matches=exact,
        source_support_candidates=[{"doc_index": 0}],
        gold_doc_indices=[0],
        variable_values={},
    )

    assert status == "openie_exact_available"


def test_classify_marks_source_support_when_openie_exact_fails():
    obligation = _obligation(["Messi", "signed for", "?team"])

    status = classify_source_support_status(
        obligation=obligation,
        exact_gold_matches={},
        source_support_candidates=[{"doc_index": 0}],
        gold_doc_indices=[0],
        variable_values={},
    )

    assert status == "openie_failed_but_source_supports"


def test_classify_marks_unresolved_variable_when_no_anchor_exists():
    obligation = _obligation(["?x1", "signed for", "?team"])

    status = classify_source_support_status(
        obligation=obligation,
        exact_gold_matches={},
        source_support_candidates=[],
        gold_doc_indices=[0],
        variable_values={},
    )

    assert status == "unresolved_variable_issue"
