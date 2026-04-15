import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_sentence_attribution_probe import extract_probe_cases, split_sentences


def test_split_sentences_breaks_basic_text():
    text = "Alpha is here. Beta is there!\nGamma asks why?"
    assert split_sentences(text) == [
        "Alpha is here.",
        "Beta is there!",
        "Gamma asks why?",
    ]


def test_extract_probe_cases_filters_positive_and_nonpositive():
    corpus = [
        {"title": "Doc A", "text": "alpha. beta."},
        {"title": "Doc B", "text": "bridge. gamma."},
        {"title": "Doc C", "text": "delta. epsilon."},
    ]
    payload = {
        "dataset": "musique",
        "expand_assemble_query_traces": [
            {
                "question": "Q-pos?",
                "gold_titles": ["Doc A"],
                "gold_answers": ["alpha"],
                "baseline_metrics": {"ExactMatch": 0.0, "F1": 0.0},
                "method_metrics": {"ExactMatch": 1.0, "F1": 1.0},
                "expand_assemble_trace": {
                    "query_entities_preview": ["alpha"],
                    "assemble_trace": {
                        "repair_applied": True,
                        "scaffold_titles": ["Doc A", "Doc B"],
                        "ranking_rows": [
                            {"pool_position": 1, "doc_id": 1, "assemble_score": -0.1},
                            {"pool_position": 2, "doc_id": 2, "assemble_score": -0.2},
                        ],
                        "best_repair_swap": {
                            "appended_pool_position": 2,
                            "replace_pool_position": 1,
                            "candidate_title": "Doc C",
                            "replace_title": "Doc B",
                            "candidate_ce_rank": 6,
                            "replaced_incumbent_ce_rank": 5,
                            "connector_gain_count": 1,
                            "anchor_gain_count": 0,
                        },
                    },
                },
                "method_top_titles": ["Doc A", "Doc C"],
            },
            {
                "question": "Q-neg?",
                "gold_titles": ["Doc B"],
                "gold_answers": ["bridge"],
                "baseline_metrics": {"ExactMatch": 1.0, "F1": 1.0},
                "method_metrics": {"ExactMatch": 1.0, "F1": 1.0},
                "expand_assemble_trace": {
                    "query_entities_preview": ["bridge"],
                    "assemble_trace": {
                        "repair_applied": True,
                        "scaffold_titles": ["Doc A", "Doc B"],
                        "ranking_rows": [
                            {"pool_position": 0, "doc_id": 0, "assemble_score": -0.1},
                            {"pool_position": 2, "doc_id": 2, "assemble_score": -0.3},
                        ],
                        "best_repair_swap": {
                            "appended_pool_position": 2,
                            "replace_pool_position": 0,
                            "candidate_title": "Doc C",
                            "replace_title": "Doc A",
                            "candidate_ce_rank": 7,
                            "replaced_incumbent_ce_rank": 5,
                            "connector_gain_count": 1,
                            "anchor_gain_count": 0,
                        },
                    },
                },
                "method_top_titles": ["Doc C", "Doc B"],
            },
        ],
    }

    positive_cases = extract_probe_cases(
        repair_payload=payload,
        corpus=corpus,
        case_mode="applied_positive",
        max_cases=0,
    )
    nonpositive_cases = extract_probe_cases(
        repair_payload=payload,
        corpus=corpus,
        case_mode="applied_nonpositive",
        max_cases=0,
    )

    assert len(positive_cases) == 1
    assert positive_cases[0]["question"] == "Q-pos?"
    assert positive_cases[0]["candidate_title"] == "Doc C"
    assert positive_cases[0]["replace_title"] == "Doc B"

    assert len(nonpositive_cases) == 1
    assert nonpositive_cases[0]["question"] == "Q-neg?"
    assert nonpositive_cases[0]["candidate_title"] == "Doc C"
    assert nonpositive_cases[0]["replace_title"] == "Doc A"
