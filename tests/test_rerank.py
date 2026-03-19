from src.hipporag.rerank import DSPyFilter


class _DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def infer(self, messages, model, **kwargs):
        self.calls.append({"messages": messages, "model": model, "kwargs": kwargs})
        response = self.responses.pop(0)
        return response, {"finish_reason": "stop"}


class _DummyConfig:
    rerank_dspy_file_path = None
    llm_name = "qwen3-8b"
    rerank_require_non_empty = True


class _DummyHippoRAG:
    def __init__(self, responses):
        self.global_config = _DummyConfig()
        self.llm_model = _DummyLLM(responses)


def test_rerank_retries_with_id_only_json_after_parse_failure():
    malformed = "not valid json"
    recovered = '{"best_ids": [0], "confidence": 0.75}'
    filter_model = DSPyFilter(_DummyHippoRAG([malformed, recovered]))
    filter_model.max_repair_retries = 0

    candidate_items = [
        ("film a", "directed by", "person a"),
        ("film b", "directed by", "person b"),
    ]
    candidate_indices = [7, 8]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=1,
    )

    assert result_indices == [7]
    assert result_items == [("film a", "directed by", "person a")]
    assert rerank_log["attempts"][0]["parse_succeeded"] is False
    assert rerank_log["attempts"][1]["phase"] == "primary"
    assert rerank_log["attempts"][1]["attempt"] == 2
    assert rerank_log["attempts"][1]["allow_think"] is False
    assert rerank_log["used_fallback"] is False
    assert rerank_log["facts_empty_stage"] == "ok"
    assert rerank_log["n_candidates_initial"] == 2
    assert rerank_log["n_facts_after_rerank_raw"] == 1
    assert rerank_log["n_facts_after_mapping"] == 1
    assert rerank_log["n_facts_after_postprocess"] == 1
    assert rerank_log["n_facts_final"] == 1
    assert rerank_log["confidence"] == 0.75
    assert rerank_log["parsed_best_ids"] == [0]
    assert rerank_log["invalid_best_ids"] == []
    assert rerank_log["candidate_id_list"] == [0, 1]
    assert rerank_log["valid_id_range"] == [0, 1]


def test_rerank_falls_back_to_top_candidates_when_all_attempts_fail():
    responses = ["not json at all"]
    filter_model = DSPyFilter(_DummyHippoRAG(responses))
    filter_model.max_parse_retries = 0
    filter_model.max_repair_retries = 0

    candidate_items = [
        ("film a", "directed by", "person a"),
        ("film a", "release year", "1999"),
        ("film c", "directed by", "person c"),
    ]
    candidate_indices = [3, 4, 5]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=2,
    )

    assert result_indices == [3, 4]
    assert result_items == candidate_items[:2]
    assert rerank_log["used_fallback"] is True
    assert rerank_log["fallback_reason"] == "parse_failure"
    assert rerank_log["facts_empty_stage"] == "reranker_parse_failure"
    assert rerank_log["final_non_empty_fallback_applied"] is True
    assert rerank_log["final_non_empty_fallback_reason"] == "parse_failure"
    assert rerank_log["n_facts_after_rerank_raw"] == 0
    assert rerank_log["n_facts_after_mapping"] == 0
    assert rerank_log["n_facts_final"] == 2


def test_rerank_semantic_empty_uses_non_empty_fallback():
    filter_model = DSPyFilter(_DummyHippoRAG(['{"best_ids": []}']))
    filter_model.max_parse_retries = 0
    filter_model.max_repair_retries = 0

    candidate_items = [
        ("film a", "directed by", "person a"),
        ("film b", "directed by", "person b"),
    ]
    candidate_indices = [7, 8]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=1,
    )

    assert result_indices == [7]
    assert result_items == [("film a", "directed by", "person a")]
    assert rerank_log["used_fallback"] is True
    assert rerank_log["fallback_reason"] == "empty_output"
    assert rerank_log["model_semantic_empty_count"] == 1
    assert rerank_log["non_empty_fallback_count"] == 1
    assert rerank_log["final_facts_empty_count"] == 0
    assert rerank_log["facts_empty_stage"] == "reranker_semantic_empty"
    assert rerank_log["final_non_empty_fallback_applied"] is True
    assert rerank_log["final_non_empty_fallback_reason"] == "empty_output"
    assert rerank_log["n_facts_after_rerank_raw"] == 0
    assert rerank_log["n_facts_after_mapping"] == 0
    assert rerank_log["n_facts_final"] == 1


def test_rerank_schema_failure_uses_non_empty_fallback_for_1_based_ids():
    filter_model = DSPyFilter(_DummyHippoRAG(['{"best_ids": [1]}']))
    filter_model.max_parse_retries = 0
    filter_model.max_repair_retries = 0

    candidate_items = [("film a", "directed by", "person a")]
    candidate_indices = [7]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=1,
    )

    assert result_indices == [7]
    assert result_items == [("film a", "directed by", "person a")]
    assert rerank_log["used_fallback"] is True
    assert rerank_log["fallback_reason"] == "schema_failure"
    assert rerank_log["facts_empty_stage"] == "reranker_schema_failure"
    assert rerank_log["schema_failure_count"] >= 1
    assert rerank_log["parsed_best_ids"] == [1]
    assert rerank_log["invalid_best_ids"] == [1]
    assert rerank_log["final_non_empty_fallback_applied"] is True
    assert rerank_log["final_non_empty_fallback_reason"] == "schema_failure"


def test_rerank_schema_failure_uses_non_empty_fallback_for_string_ids():
    filter_model = DSPyFilter(_DummyHippoRAG(['{"best_ids": ["0", "1"]}']))
    filter_model.max_parse_retries = 0
    filter_model.max_repair_retries = 0

    candidate_items = [
        ("film a", "directed by", "person a"),
        ("film b", "directed by", "person b"),
    ]
    candidate_indices = [7, 8]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=1,
    )

    assert result_indices == [7]
    assert result_items == [("film a", "directed by", "person a")]
    assert rerank_log["fallback_reason"] == "schema_failure"
    assert rerank_log["schema_failure_count"] >= 1
    assert rerank_log["invalid_best_ids"] == ["0", "1"]
    assert rerank_log["facts_empty_stage"] == "reranker_schema_failure"


def test_rerank_schema_failure_handles_text_output():
    responses = ["the best candidate is id 0", '{"best_ids": [0]}']
    filter_model = DSPyFilter(_DummyHippoRAG(responses))
    filter_model.max_parse_retries = 0

    candidate_items = [
        ("film a", "directed by", "person a"),
        ("film b", "directed by", "person b"),
    ]
    candidate_indices = [7, 8]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=1,
    )

    assert result_indices == [7]
    assert result_items == [("film a", "directed by", "person a")]
    assert rerank_log["attempts"][0]["parse_succeeded"] is False
    assert "the best candidate is id 0" in rerank_log["attempts"][0]["raw_output_preview"]
    assert rerank_log["parsed_best_ids"] == [0]
    assert rerank_log["invalid_best_ids"] == []


def test_rerank_partial_valid_ids_keep_valid_selection_and_trace_invalid_ids():
    filter_model = DSPyFilter(_DummyHippoRAG(['{"best_ids": [0, 0, 7, 1]}']))
    filter_model.max_parse_retries = 0
    filter_model.max_repair_retries = 0

    candidate_items = [
        ("film a", "directed by", "person a"),
        ("film b", "directed by", "person b"),
    ]
    candidate_indices = [7, 8]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=2,
    )

    assert result_indices == [7, 8]
    assert result_items == candidate_items
    assert rerank_log["used_fallback"] is False
    assert rerank_log["facts_empty_stage"] == "ok"
    assert rerank_log["schema_failure_count"] == 1
    assert rerank_log["parsed_best_ids"] == [0, 0, 7, 1]
    assert rerank_log["invalid_best_ids"] == [7]
    assert rerank_log["mapping_issue_detected"] is True
    assert rerank_log["mapping_issue_reason_counts"]["invalid_best_id"] == 1
    assert rerank_log["mapping_issue_reason_counts"]["duplicate_selected_id"] == 1
    assert rerank_log["mapping_duplicate_drop_count"] == 1
    assert rerank_log["mapping_exact_match_count"] == 2
    assert rerank_log["n_facts_after_mapping"] == 2
    assert rerank_log["n_facts_final"] == 2


def test_rerank_schema_failure_can_keep_legacy_empty_behavior():
    hipporag = _DummyHippoRAG(['{"best_ids": [99]}'])
    hipporag.global_config.rerank_require_non_empty = False
    filter_model = DSPyFilter(hipporag)
    filter_model.max_parse_retries = 0
    filter_model.max_repair_retries = 0

    candidate_items = [
        ("film a", "directed by", "person a"),
        ("film b", "directed by", "person b"),
    ]
    candidate_indices = [7, 8]

    result_indices, result_items, rerank_log = filter_model.rerank(
        query="Who directed film a?",
        candidate_items=candidate_items,
        candidate_indices=candidate_indices,
        len_after_rerank=1,
    )

    assert result_indices == []
    assert result_items == []
    assert rerank_log["used_fallback"] is False
    assert rerank_log["fallback_reason"] == "schema_failure"
    assert rerank_log["final_failure_reason"] == "schema_failure"
    assert rerank_log["final_facts_empty_count"] == 1
    assert rerank_log["facts_empty_stage"] == "reranker_schema_failure"
    assert rerank_log["final_non_empty_fallback_applied"] is False


if __name__ == "__main__":
    test_rerank_retries_with_id_only_json_after_parse_failure()
    test_rerank_falls_back_to_top_candidates_when_all_attempts_fail()
    test_rerank_semantic_empty_uses_non_empty_fallback()
    test_rerank_schema_failure_uses_non_empty_fallback_for_1_based_ids()
    test_rerank_schema_failure_uses_non_empty_fallback_for_string_ids()
    test_rerank_schema_failure_handles_text_output()
    test_rerank_partial_valid_ids_keep_valid_selection_and_trace_invalid_ids()
    test_rerank_schema_failure_can_keep_legacy_empty_behavior()
