import tempfile
import unittest
from pathlib import Path

from build_query_obligation_cache import (
    build_cache_row,
    build_query_obligation_cache,
    parse_query_indices,
    parse_query_obligation_response,
)
from evaluate_obligation_closed_sto_local_ppr import (
    load_query_obligation_cache,
    query_triples_from_row,
)


class QueryObligationCacheTests(unittest.TestCase):
    def test_parse_query_obligation_response_accepts_fenced_json(self):
        triples = parse_query_obligation_response(
            '```json\n{"triples": [["Lothair II", "mother", "?x"], ["?x", "died in", "?date"]]}\n```'
        )

        self.assertEqual(
            triples,
            [["Lothair II", "mother", "?x"], ["?x", "died in", "?date"]],
        )

    def test_parse_query_indices_accepts_ranges(self):
        self.assertEqual(parse_query_indices("1,3-5,8"), {1, 3, 4, 5, 8})

    def test_build_cache_row_materializes_obligations(self):
        row = build_cache_row(
            query_index=7,
            question="When did Lothair II's mother die?",
            triples=[["Lothair II", "mother", "?x"], ["?x", "died in", "?date"]],
            source="unit_test",
            status="ok",
        )

        self.assertEqual(row["query_index"], 7)
        self.assertEqual(len(row["query_triples"]), 2)
        self.assertEqual(len(row["query_obligations"]), 2)
        self.assertTrue(row["query_obligations"][0]["object_is_variable"])

    def test_build_head_trace_cache_and_loader_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            head_trace_path = tmp / "head_trace.json"
            map_path = tmp / "map.json"
            output_path = tmp / "cache.json"
            report_path.write_text(
                """
{
  "datasets": [
    {
      "dataset": "toy",
      "rows": [
        {"query_index": 0, "question": "Who directed Alpha?"},
        {"query_index": 1, "question": "Where was Beta born?"}
      ]
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            head_trace_path.write_text(
                """
{"head_traces": [
  {"top_k_facts": [["Alpha", "directed by", "?answer"]]},
  {"top_k_facts": []}
]}
""".strip(),
                encoding="utf-8",
            )
            map_path.write_text('{"toy": "' + str(head_trace_path) + '"}', encoding="utf-8")

            build_query_obligation_cache(
                report_path=report_path,
                source="head_trace_top_k_facts",
                output_json_path=output_path,
                datasets_filter=set(),
                query_indices=set(),
                max_queries=0,
                head_trace_topk_map={"toy": {0: [["Alpha", "directed by", "?answer"]]}},
            )

            cache = load_query_obligation_cache(output_path, "toy")
            self.assertEqual(cache[0], [["Alpha", "directed by", "?answer"]])
            self.assertNotIn(1, cache)

    def test_query_triples_from_row_prefers_explicit_row_then_cache_then_trace(self):
        self.assertEqual(
            query_triples_from_row(
                {"query_index": 3, "query_triples": [["Row", "rel", "Obj"]]},
                head_traces=[{"top_k_facts": [["Trace", "rel", "Obj"]]}],
                query_obligation_cache={3: [["Cache", "rel", "Obj"]]},
            ),
            [["Row", "rel", "Obj"]],
        )
        self.assertEqual(
            query_triples_from_row(
                {"query_index": 3},
                head_traces=[{}, {}, {}, {"top_k_facts": [["Trace", "rel", "Obj"]]}],
                query_obligation_cache={3: [["Cache", "rel", "Obj"]]},
            ),
            [["Cache", "rel", "Obj"]],
        )
        self.assertEqual(
            query_triples_from_row(
                {"query_index": 3},
                query_obligation_cache={3: [["Cache", "rel", "Obj"]]},
            ),
            [["Cache", "rel", "Obj"]],
        )


if __name__ == "__main__":
    unittest.main()
