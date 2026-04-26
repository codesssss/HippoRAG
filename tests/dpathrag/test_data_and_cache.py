from __future__ import annotations

from src.dpathrag.cache import build_cache_record, gold_support_indicators
from src.dpathrag.data import context_documents, evidence_path_entities, normalize_text, support_titles
from src.dpathrag.metrics import recall_at_k, support_complete_at_k
from src.dpathrag.reader import exact_match, format_reader_input, score_prediction, summarize_scores, token_f1
from src.dpathrag.reader_data import build_gold_reader_record, build_pool_reader_record
from src.dpathrag.selector_data import FEATURE_NAMES, featurize_selector_record, selection_overlap, support_metrics_for_indices


def test_support_titles_and_evidence_entities() -> None:
    sample = {
        "supporting_facts": [["Lothair II", 1], ["Ermengarde of Tours", 0], ["Lothair II", 2]],
        "evidences": [
            ["Lothair II", "mother", "Ermengarde of Tours"],
            ["Ermengarde of Tours", "date of death", "20 March 851"],
        ],
    }
    assert support_titles(sample) == ["Ermengarde of Tours", "Lothair II"]
    assert evidence_path_entities(sample) == ["Lothair II", "Ermengarde of Tours", "20 March 851"]


def test_gold_support_indicators_normalize_titles() -> None:
    assert gold_support_indicators([" Lothair II ", "Other"], ["lothair ii"]) == [1, 0]
    assert normalize_text("A,  B!") == "a b"


def test_support_metrics() -> None:
    gold = ["A", "B"]
    pool = ["A", "C", "B"]
    assert recall_at_k(gold, pool, 1) == 0.5
    assert support_complete_at_k(gold, pool, 2) == 0.0
    assert support_complete_at_k(gold, pool, 3) == 1.0


def test_build_cache_record_labels_candidates() -> None:
    sample = {
        "_id": "q1",
        "question": "question?",
        "answer": "answer",
        "type": "comparison",
        "supporting_facts": [["Gold A", 0], ["Gold B", 1]],
        "evidences": [["Gold A", "rel", "Gold B"]],
    }
    pool_record = {
        "query_idx": 0,
        "question": "question?",
        "pool_docs": ["Gold A\ntext", "Distractor\ntext", "Gold B\ntext"],
        "pool_titles": ["Gold A", "Distractor", "Gold B"],
        "pool_doc_scores": [0.9, 0.4, 0.3],
        "pool_doc_ids": [10, 11, 12],
    }
    row = build_cache_record(sample, pool_record, source="dense")
    assert row["qid"] == "q1"
    assert [candidate["gold_support"] for candidate in row["candidates"]] == [1, 0, 1]
    assert row["candidates"][0]["features"]["rank"] == 1.0
    assert "gold_support_indicator" not in row["candidates"][0]["features"]


def test_context_documents_and_gold_reader_record() -> None:
    sample = {
        "_id": "q1",
        "question": "question?",
        "answer": "answer",
        "supporting_facts": [["Gold A", 0], ["Gold B", 1]],
        "context": [["Gold A", ["a1", "a2"]], ["Distractor", ["d"]], ["Gold B", ["b"]]],
    }
    docs = context_documents(sample)
    assert docs[0]["doc"] == "Gold A\na1 a2"
    row = build_gold_reader_record(sample, split="validation")
    assert row["source"] == "gold"
    assert [doc["gold_support"] for doc in row["selected_docs"]] == [1, 1]
    assert row["support_complete"] == 1.0


def test_pool_reader_record() -> None:
    sample = {
        "_id": "q1",
        "question": "question?",
        "answer": "answer",
        "supporting_facts": [["Gold A", 0], ["Gold B", 1]],
    }
    pool_record = {
        "query_idx": 0,
        "question": "question?",
        "pool_docs": ["Gold A\ntext", "Distractor\ntext", "Gold B\ntext"],
        "pool_titles": ["Gold A", "Distractor", "Gold B"],
        "pool_doc_scores": [0.9, 0.4, 0.3],
    }
    row = build_pool_reader_record(sample, pool_record, split="dev1000", source="dense", top_k=2)
    assert row["source"] == "dense"
    assert row["support_recall"] == 0.5
    assert row["support_complete"] == 0.0


def test_reader_format_and_metrics() -> None:
    record = {
        "qid": "q1",
        "source": "gold",
        "question": "Who?",
        "answer": "The Queen",
        "support_recall": 1.0,
        "support_complete": 1.0,
        "selected_docs": [{"title": "Doc", "text": "Doc\nThe queen appears."}],
    }
    prompt = format_reader_input(record)
    assert "Question: Who?" in prompt
    assert "[1] Doc" in prompt
    assert exact_match(["Queen"], "the queen") == 1.0
    assert token_f1(["The Queen"], "queen") == 1.0
    scored = score_prediction(record, "queen")
    summary = summarize_scores([scored])
    assert summary["answer_em"] == 1.0
    assert summary["answer_f1"] == 1.0


def test_selector_featurization_keeps_gold_label_out_of_features() -> None:
    record = {
        "qid": "q1",
        "question": "Who is related to Gold A?",
        "type": "bridge",
        "gold_titles": ["Gold A"],
        "candidates": [
            {
                "rank": 1,
                "title": "Gold A",
                "text": "Gold A\nGold A is a person.",
                "retriever_score": 0.9,
                "gold_support": 1,
            },
            {
                "rank": 2,
                "title": "Other",
                "text": "Other\nDistractor text.",
                "retriever_score": 0.1,
                "gold_support": 0,
            },
        ],
    }
    example = featurize_selector_record(record, max_candidates=4, path_len=3)
    assert len(example.candidate_features[0]) == len(FEATURE_NAMES)
    assert example.target_indices == [0, -100, -100]
    assert example.candidate_mask == [True, True, False, False]
    assert "gold" not in " ".join(FEATURE_NAMES)
    metrics = support_metrics_for_indices(example, [0, 1])
    assert metrics["support_complete"] == 1.0
    assert metrics["duplicate_title"] == 0.0
    assert selection_overlap([0, 1], [1, 2]) == 1 / 3


def test_selector_featurization_accepts_extra_embedding_features() -> None:
    record = {
        "qid": "q1",
        "question": "Who is related to Gold A?",
        "type": "compositional",
        "gold_titles": ["Gold A"],
        "evidences": [["A", "rel", "Bridge"], ["Bridge", "rel", "Answer"]],
        "candidates": [
            {
                "rank": 1,
                "title": "Gold A",
                "text": "Gold A\nBridge appears here.",
                "retriever_score": 0.9,
                "gold_support": 1,
            }
        ],
    }
    example = featurize_selector_record(
        record,
        max_candidates=2,
        path_len=2,
        candidate_extra_features=[[0.2, 0.3]],
        query_extra_features=[0.4, 0.5],
    )
    assert len(example.candidate_features[0]) == len(FEATURE_NAMES) + 2
    assert len(example.query_features) == len(FEATURE_NAMES) + 2
    assert example.bridge_entities == ["Bridge"]
    metrics = support_metrics_for_indices(example, [0])
    assert metrics["bridge_entity_recall"] == 1.0
