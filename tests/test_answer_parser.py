from src.hipporag.utils.misc_utils import extract_answer_from_response


def test_extract_answer_from_standard_marker():
    answer, info = extract_answer_from_response(
        "Thought: The employer is University of Southampton.\nAnswer: 1862."
    )

    assert answer == "1862."
    assert info["used_fallback"] is False
    assert info["error_type"] is None


def test_extract_answer_from_final_answer_marker():
    answer, info = extract_answer_from_response("Final Answer: Paris")

    assert answer == "Paris"
    assert info["used_fallback"] is False
    assert info["error_type"] is None


def test_extract_answer_from_multiline_answer_block():
    answer, info = extract_answer_from_response("Thought: ...\nAnswer:\nThe answer is Marie de Medici\n")

    assert answer == "Marie de Medici"
    assert info["used_fallback"] is False


def test_extract_answer_falls_back_when_marker_missing():
    raw = "The University of Southampton was founded in 1862."
    answer, info = extract_answer_from_response(raw)

    assert answer == raw
    assert info["used_fallback"] is True
    assert info["error_type"] == "missing_answer_marker"


def test_extract_answer_handles_non_string_responses():
    answer, info = extract_answer_from_response(["Answer: 1862"])

    assert answer == "['Answer: 1862']"
    assert info["used_fallback"] is True
    assert info["response_type"] == "list"


def test_extract_answer_from_weak_answer_marker():
    answer, info = extract_answer_from_response(
        "Thought: compare the two dates carefully. So the answer is 20 March 851."
    )

    assert answer == "20 March 851."
    assert info["used_fallback"] is False
    assert info["error_type"] == "weak_answer_marker"


def test_extract_answer_prefers_explicit_answer_marker():
    answer, info = extract_answer_from_response(
        "Thought: The answer is probably wrong.\nAnswer: the correct one"
    )

    assert answer == "the correct one"
    assert info["used_fallback"] is False
    assert info["error_type"] is None


if __name__ == "__main__":
    test_extract_answer_from_standard_marker()
    test_extract_answer_from_final_answer_marker()
    test_extract_answer_from_multiline_answer_block()
    test_extract_answer_falls_back_when_marker_missing()
    test_extract_answer_handles_non_string_responses()
    test_extract_answer_from_weak_answer_marker()
    test_extract_answer_prefers_explicit_answer_marker()
