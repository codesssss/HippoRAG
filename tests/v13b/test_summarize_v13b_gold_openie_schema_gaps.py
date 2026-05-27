from summarize_v13b_gold_openie_schema_gaps import (
    case_endpoint_shape,
    parse_relation_pair,
    relation_family,
    summarize_reports,
)


def test_relation_family_groups_schema_level_frames():
    assert relation_family("released_on") == "temporal_attribute"
    assert relation_family("country_origin") == "place_or_origin_attribute"
    assert relation_family("voice_cast_includ") == "work_metadata_role"
    assert relation_family("<empty>") == "lead_or_untyped_fact"


def test_parse_relation_pair_keeps_missing_right_side_empty():
    assert parse_relation_pair("born_in -> <empty>") == ("born_in", "<empty>")
    assert parse_relation_pair("malformed") == ("malformed", "")


def test_case_endpoint_shape_marks_descriptive_bound_endpoint():
    case = {
        "normalized_obligation": {
            "subject_is_variable": False,
            "subject_surface": "recently abdicated queen",
            "object_is_variable": True,
            "object_surface": "?x2",
        }
    }

    assert case_endpoint_shape(case) == "descriptive_bound_endpoint"


def test_summarize_reports_keeps_candidate_gap_separate_from_openie_gap():
    report = {
        "dataset": "toy",
        "audited_ungrounded_obligation_count": 4,
        "status_counts": {
            "candidate_missing": 1,
            "gold_endpoint_relation_mismatch": 2,
            "gold_endpoint_missing": 1,
        },
        "relation_mismatch_pair_counts": {
            "released_on -> <empty>": 3,
            "country_origin -> voice_cast_includ": 2,
        },
        "representative_cases": {
            "gold_endpoint_missing": [
                {
                    "query_index": 0,
                    "raw_triple": ["recently abdicated queen", "imprisoned by", "?x2"],
                    "normalized_obligation": {
                        "subject_is_variable": False,
                        "subject_surface": "recently abdicated queen",
                        "object_is_variable": True,
                        "object_surface": "?x2",
                    },
                }
            ]
        },
    }

    summary = summarize_reports([report])

    assert summary["status_totals"]["candidate_missing"] == 1
    assert summary["status_totals"]["gold_endpoint_relation_mismatch"] == 2
    assert summary["schema_gap_kind_counts"]["lead_sentence_attribute_not_materialized"] == 3
    assert summary["schema_gap_kind_counts"]["cross_family_relation_mismatch"] == 2
    assert summary["interpretation"]["main_bottleneck"] == "qwen_openie_schema_alignment"
    assert summary["representative_endpoint_shape_counts"]["descriptive_bound_endpoint"] == 1
