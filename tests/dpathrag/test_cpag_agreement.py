from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_cpag_agreement.py"
_SPEC = importlib.util.spec_from_file_location("run_cpag_agreement", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _record(source: str, titles: list[str]) -> tuple[str, dict]:
    return (
        source,
        {
            "qid": "q1",
            "question": "Where was the director of Inception born?",
            "gold_titles": ["Inception", "Christopher Nolan"],
            "candidates": [
                {
                    "doc_id": idx,
                    "rank": idx + 1,
                    "title": title,
                    "text": f"{title}\n{text}",
                    "retriever_score": 1.0 / (idx + 1),
                    "gold_support": int(title in {"Inception", "Christopher Nolan"}),
                }
                for idx, (title, text) in enumerate(
                    [
                        ("Inception", "Inception was directed by Christopher Nolan."),
                        ("Christopher Nolan", "Christopher Nolan was born in London."),
                        ("London", "London is a city."),
                    ]
                    if titles == ["Inception", "Christopher Nolan", "London"]
                    else [(title, f"{title} text.") for title in titles]
                )
            ],
        },
    )


def test_dedup_union_docs_merges_pool_sources() -> None:
    docs = _MODULE.dedup_union_docs(
        [
            _record("proprag", ["Inception", "Christopher Nolan", "London"]),
            _record("dense", ["Inception", "London"]),
        ],
        pool_k=3,
        cap=10,
    )
    inception = next(doc for doc in docs if doc["title"] == "Inception")
    assert sorted(inception["pool_sources"]) == ["dense", "proprag"]
    assert inception["pool_count"] if "pool_count" in inception else len(inception["pool_sources"]) == 2


def test_build_doc_signals_uses_openie_triples_for_agreement() -> None:
    docs = _MODULE.dedup_union_docs([_record("proprag", ["Inception", "Christopher Nolan", "London"])], pool_k=3, cap=10)
    openie = {
        "inception": {
            "extracted_entities": ["Inception", "Christopher Nolan"],
            "extracted_triples": [["Inception", "was directed by", "Christopher Nolan"]],
        },
        "christopher nolan": {
            "extracted_entities": ["Christopher Nolan", "London"],
            "extracted_triples": [["Christopher Nolan", "was born in", "London"]],
        },
    }
    rows = _MODULE.build_doc_signals(
        docs,
        question="Where was the director of Inception born?",
        openie_index=openie,
        max_props=4,
    )
    nolan_rows = [row for row in rows if "christopher nolan" in row["entity_norms"]]
    assert len(nolan_rows) == 2
    assert any("christopher nolan" in row["bridge_entities"] for row in rows)


def test_greedy_agreement_select_prefers_bridge_coverage() -> None:
    rows = [
        {
            "title": "Distractor",
            "doc_key": "d0",
            "bridge_entities": [],
            "answer_type_entities": [],
            "anchor_entities": [],
            "pool_count": 1,
            "best_rank": 1,
            "is_singleton_lexical": True,
        },
        {
            "title": "Inception",
            "doc_key": "d1",
            "bridge_entities": ["christopher nolan"],
            "answer_type_entities": ["christopher nolan"],
            "anchor_entities": ["inception"],
            "pool_count": 2,
            "best_rank": 2,
            "is_singleton_lexical": False,
        },
    ]
    assert _MODULE.greedy_agreement_select(rows, top_k=1) == [1]


def test_anchor_first_agreement_select_preserves_query_anchor() -> None:
    rows = [
        {
            "title": "Hub",
            "doc_key": "d0",
            "bridge_entities": ["x", "y", "z"],
            "answer_type_entities": ["x", "y", "z"],
            "anchor_entities": [],
            "pool_count": 2,
            "best_rank": 1,
            "is_singleton_lexical": False,
        },
        {
            "title": "Inception",
            "doc_key": "d1",
            "bridge_entities": ["christopher nolan"],
            "answer_type_entities": ["christopher nolan"],
            "anchor_entities": ["inception"],
            "pool_count": 2,
            "best_rank": 2,
            "is_singleton_lexical": False,
        },
    ]
    assert _MODULE.anchor_first_agreement_select(rows, top_k=1) == [1]


def test_support_metrics_from_titles() -> None:
    metrics = _MODULE.support_metrics(
        ["Inception", "Christopher Nolan"],
        [{"title": "Inception"}, {"title": "London"}],
    )
    assert metrics["support_recall"] == 0.5
    assert metrics["support_complete"] == 0.0
