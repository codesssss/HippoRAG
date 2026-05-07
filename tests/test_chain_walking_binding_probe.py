from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from probe_chain_walking_binding import (  # noqa: E402
    entities_from_raw,
    normalize_match_text,
    rank_pool_by_entities,
    summarize_probe_rows,
    target_missing_gold_rows,
    upstream_positions_for_requirement,
)


def test_entities_from_raw_accepts_json_objects_and_code_fences() -> None:
    raw = """```json
{"entities": [{"text": "New Zealand", "support_span": "born in New Zealand"}, {"text": "Australia"}]}
```"""

    assert entities_from_raw(raw) == [
        {"entity": "New Zealand", "support_span": "born in New Zealand", "why": ""},
        {"entity": "Australia", "support_span": "", "why": ""},
    ]


def test_normalize_match_text_strips_accents_and_parentheticals() -> None:
    assert normalize_match_text("César Gaytan (MSC-94)") == "cesar gaytan"


def test_rank_pool_by_entities_promotes_exact_title_match() -> None:
    titles = [
        "Capital punishment in the United States",
        "Capital punishment in Canada",
        "Capital punishment in New Zealand",
    ]
    docs = [f"{title}\ntext" for title in titles]

    order, best = rank_pool_by_entities(
        pool_titles=titles,
        pool_docs=docs,
        entity_rows=[
            {
                "entity": "Capital punishment in New Zealand",
                "requirement_id": "s2",
                "upstream_title": "Markus Zusak",
            }
        ],
    )

    assert order[0] == 2
    assert best[2]["reason"] == "title_exact"
    assert best[2]["requirement_id"] == "s2"


def test_rank_pool_by_entities_uses_body_when_title_does_not_match() -> None:
    titles = ["Noise", "Paula Santiago", "Other"]
    docs = [
        "Noise\nNo useful city.",
        "Paula Santiago\nPaula Santiago was born in Guadalajara and exhibited in Europe.",
        "Other\nNo useful city.",
    ]

    order, best = rank_pool_by_entities(
        pool_titles=titles,
        pool_docs=docs,
        entity_rows=[{"entity": "Guadalajara", "upstream_title": "César Gaytan"}],
    )

    assert order[0] == 1
    assert best[1]["reason"] == "body_substring"


def test_upstream_positions_prioritizes_binding_dep_positions_then_dedups() -> None:
    requirement = {"unit_id": "s3", "subquery": "Who is the child?", "depends_on": ["s2"]}
    selector_trace = {
        "binding_candidates_by_requirement": {
            "s3": [
                {"dep_position": 9, "title_pool_position": 4},
                {"dep_position": 9, "title_pool_position": 5},
            ],
            "s2": [{"title_pool_position": 2}],
        },
        "selected_binding": {"assignments": {"s2": "John Cabot"}},
        "final_front_pool_positions": [9, 0, 1],
    }
    setr_record = {"setr_selection_trace": {"selected_positions": [1, 9, 12]}}
    pool_titles = [f"Doc {idx}" for idx in range(20)]
    pool_titles[9] = "John Cabot"

    positions = upstream_positions_for_requirement(
        requirement=requirement,
        selector_trace=selector_trace,
        setr_record=setr_record,
        pool_titles=pool_titles,
        source_top_n=2,
        max_upstream_docs=5,
    )

    assert positions[:4] == [9, 2, 0, 1]
    assert len(positions) == len(set(positions))


def test_summarize_probe_rows_sets_decision_from_new_recall_at_5() -> None:
    rows = []
    for idx in range(10):
        recovered = idx < 3
        rows.append(
            {
                "query_index": idx,
                "source_best_gold_rank": 30,
                "best_probe_rank": 4 if recovered else 30,
                "rank_improvement": 26 if recovered else 0,
                "baseline_hit_at_3": 0,
                "baseline_hit_at_5": 0,
                "baseline_hit_at_10": 0,
                "baseline_hit_at_20": 0,
                "probe_hit_at_3": 0,
                "probe_hit_at_5": int(recovered),
                "probe_hit_at_10": int(recovered),
                "probe_hit_at_20": int(recovered),
                "new_hit_at_3": 0,
                "new_hit_at_5": int(recovered),
                "new_hit_at_10": int(recovered),
                "new_hit_at_20": int(recovered),
                "best_probe_score": 100 if recovered else 0,
            }
        )

    summary, metadata = summarize_probe_rows(rows, prompt_rows=[{"parse_ok": 1, "entity_count": 2}])
    overall = next(row for row in summary if row["bucket"] == "overall")

    assert overall["new_recall_at_5"] == 0.3
    assert metadata["decision"] == "strong_go"
    assert metadata["parse_ok_rate"] == 1.0


def test_target_missing_gold_rows_can_filter_to_query_primary_bucket(tmp_path: Path) -> None:
    audit_dir = tmp_path
    (audit_dir / "missing_gold_audit.csv").write_text(
        "dataset,query_index,gold_title,bucket\n"
        "MuSiQue,1,A,gold_source_only_not_rank_or_dbec\n"
        "MuSiQue,2,B,gold_source_only_not_rank_or_dbec\n"
        "MuSiQue,3,C,gold_recovered_by_dbec_final\n",
        encoding="utf-8",
    )
    (audit_dir / "query_audit.csv").write_text(
        "dataset,query_index,primary_bucket\n"
        "MuSiQue,1,gold_source_only_not_rank_or_dbec\n"
        "MuSiQue,2,all_missing_gold_recovered_by_dbec\n",
        encoding="utf-8",
    )

    rows = target_missing_gold_rows(audit_dir=audit_dir, query_primary_only=True)

    assert [row["gold_title"] for row in rows] == ["A"]
