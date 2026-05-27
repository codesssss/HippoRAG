from v13b_graph_normalization import bound_endpoint_keys, fact_role_keys, obligation_role_keys


def test_obligation_and_fact_use_same_temporal_relation_key():
    obligation = {
        "obligation_id": "q1",
        "raw_subject": "?x1",
        "raw_relation": "died in",
        "raw_object": "?date",
        "subject": "x1",
        "relation": "died_on",
        "object": "date",
        "subject_is_variable": True,
        "object_is_variable": True,
    }
    fact = {
        "unit_id": "u1",
        "unit_type": "openie_fact",
        "doc_index": 3,
        "subject": "Ermengarde of Tours",
        "relation": "died on",
        "object": "20 March 851",
        "grounded_subject": "Ermengarde of Tours",
        "grounded_object": "20 March 851",
        "fact": ["Ermengarde of Tours", "died on", "20 March 851"],
    }

    assert obligation_role_keys(obligation)["relation_key"] == "died_on"
    assert fact_role_keys(fact)["relation_key"] == "died_on"


def test_bound_endpoint_keys_include_conservative_title_aliases():
    obligation = {
        "raw_subject": "song B Boy (Song)",
        "subject": "song b boy song",
        "subject_is_variable": False,
        "raw_relation": "performed by",
        "relation": "perform_by",
        "raw_object": "?x1",
        "object": "x1",
        "object_is_variable": True,
    }

    keys = bound_endpoint_keys(obligation)

    assert "song b boy song" in keys
    assert "song b boy" in keys
    assert "b boy" in keys
