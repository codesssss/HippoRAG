from src.bsgs.slots import build_oracle_slots_for_sample, sample_to_oracle_slot_record


def test_oracle_slots_parse_musique_refs():
    sample = {
        "id": "q1",
        "question": "When was the director born?",
        "answer": "1970",
        "question_decomposition": [
            {"question": "Who directed Film X?", "answer": "Jane Doe", "paragraph_support_idx": 0},
            {"question": "When was #1 born?", "answer": "1970", "paragraph_support_idx": 1},
        ],
        "paragraphs": [
            {"idx": 0, "title": "Film X", "paragraph_text": "Film X was directed by Jane Doe.", "is_supporting": True},
            {"idx": 1, "title": "Jane Doe", "paragraph_text": "Jane Doe was born in 1970.", "is_supporting": True},
        ],
    }
    slots = build_oracle_slots_for_sample(sample)
    assert len(slots) == 2
    assert slots[1].input_variables == ["x1"]
    assert slots[1].output_variable == "answer"
    record = sample_to_oracle_slot_record(sample)
    assert len(record["supporting_paragraphs"]) == 2
