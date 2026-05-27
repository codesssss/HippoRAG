from analyze_v13b_grounding_failures import (
    classify_obligation_failure,
    retrieval_critical_obligations_for_analysis,
)
from build_query_obligation_units import build_query_obligation_units
from build_source_title_openie_substrate import build_units_for_doc


def _obligation(triple):
    return build_query_obligation_units(query="q", query_triples=[triple])[0]


def _units(*docs):
    units = []
    for doc_index, triples in enumerate(docs):
        units.extend(
            build_units_for_doc(
                {
                    "idx": f"doc-{doc_index}",
                    "passage": f"Doc {doc_index}\nText.",
                    "extracted_triples": triples,
                    "extracted_entities": [],
                },
                doc_index=doc_index,
                include_source_spans=False,
            )
        )
    return units


def test_classifies_relation_mismatch_when_endpoints_align():
    failure = classify_obligation_failure(
        obligation=_obligation(["Messi", "signed by", "Barcelona"]),
        candidate_units=_units([["Messi", "played for", "Barcelona"]]),
        candidate_doc_indices=[0],
        gold_doc_indices=[0],
    )

    assert failure["failure_type"] == "relation_mismatch"
    assert failure["fact_relation_key_counts"] == {"play_for": 1}


def test_classifies_relation_mismatch_with_title_qualified_endpoint_alias():
    failure = classify_obligation_failure(
        obligation=_obligation(["song B Boy (Song)", "performed by", "?x1"]),
        candidate_units=_units([["B Boy", "is a song by", "Meek Mill"]]),
        candidate_doc_indices=[0],
        gold_doc_indices=[0],
    )

    assert failure["failure_type"] == "relation_mismatch"


def test_classifies_direction_mismatch_when_roles_are_reversed():
    failure = classify_obligation_failure(
        obligation=_obligation(["Messi", "signed by", "Barcelona"]),
        candidate_units=_units([["Barcelona", "signed by", "Messi"]]),
        candidate_doc_indices=[0],
        gold_doc_indices=[0],
    )

    assert failure["failure_type"] == "direction_mismatch"


def test_classifies_candidate_missing_before_openie_details():
    failure = classify_obligation_failure(
        obligation=_obligation(["Messi", "signed by", "Barcelona"]),
        candidate_units=_units([["Messi", "signed by", "Barcelona"]]),
        candidate_doc_indices=[0],
        gold_doc_indices=[0, 1],
    )

    assert failure["failure_type"] == "candidate_missing"
    assert failure["missing_gold_doc_indices"] == [1]


def test_classifies_variable_obligation_with_grounded_variable_anchor():
    failure = classify_obligation_failure(
        obligation=_obligation(["?x1", "died in", "?place"]),
        candidate_units=_units([["Stephen Warbeck", "born on", "3 May 1953"]]),
        candidate_doc_indices=[0],
        gold_doc_indices=[0],
        grounded_variable_values={"x1": {"Stephen Warbeck"}},
    )

    assert failure["failure_type"] == "relation_mismatch"


def test_classifies_all_variable_obligation_without_bindings_as_upstream_missing():
    failure = classify_obligation_failure(
        obligation=_obligation(["?x1", "died in", "?place"]),
        candidate_units=_units([["Stephen Warbeck", "born on", "3 May 1953"]]),
        candidate_doc_indices=[0],
        gold_doc_indices=[0],
        grounded_variable_values={},
    )

    assert failure["failure_type"] == "upstream_binding_missing"
    assert failure["missing_variable_bindings"] == ["x1", "place"]


def test_analyzer_filters_typed_non_retrieval_materialized_obligations():
    obligations = build_query_obligation_units(
        query="q",
        query_triples=[
            ["?x1", "type", "flowering plant"],
            ["Film A", "director", "?x2"],
        ],
    )
    selector = {
        "use_typed_retrieval_critical_obligations": True,
        "typed_query_program": {
            "retrieval_critical_obligation_ids": [obligations[1]["obligation_id"]],
            "materialized_non_retrieval_obligation_ids": [obligations[0]["obligation_id"]],
        },
    }

    filtered = retrieval_critical_obligations_for_analysis(
        obligations=obligations,
        selector=selector,
    )

    assert [row["raw_relation"] for row in filtered] == ["director"]
