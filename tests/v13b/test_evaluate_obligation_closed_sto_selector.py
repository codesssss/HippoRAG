import unittest

from evaluate_obligation_closed_sto_selector import (
    candidate_doc_indices_from_row,
    build_candidate_sto_units,
    frontier_query_tokens,
    seed_unit_ids_for_query,
    select_obligation_closed_evidence,
)


class ObligationClosedSTOSelectorTests(unittest.TestCase):
    def test_candidate_doc_indices_are_deduplicated_in_source_order(self):
        row = {
            "candidate_doc_indices": [3, 2, 3],
            "retrieved_doc_indices_top10": [2, 4],
            "bm25_doc_indices_top10": [5, 3],
        }

        self.assertEqual(candidate_doc_indices_from_row(row), [3, 2, 4, 5])

    def test_selector_picks_closed_unit_that_covers_query_chain(self):
        openie_docs = [
            {
                "idx": "chunk-race",
                "passage": "Race\nRace was held in Phoenix.",
                "extracted_entities": ["Race", "Phoenix"],
                "extracted_triples": [["Race", "held in", "Phoenix"]],
            },
            {
                "idx": "chunk-city",
                "passage": "City\nPhoenix is in Arizona.",
                "extracted_entities": ["Phoenix", "Arizona"],
                "extracted_triples": [["Phoenix", "in", "Arizona"]],
            },
            {
                "idx": "chunk-person",
                "passage": "Person\nPerson was born in Arizona.",
                "extracted_entities": ["Person", "Arizona"],
                "extracted_triples": [["Person", "born in", "Arizona"]],
            },
            {
                "idx": "chunk-noise",
                "passage": "Noise\nNoise mentions nothing useful.",
                "extracted_entities": ["Noise"],
                "extracted_triples": [["Noise", "mentions", "nothing useful"]],
            },
        ]

        result = select_obligation_closed_evidence(
            query="Who was born in Arizona after the race held in Phoenix?",
            candidate_doc_indices=[3, 0, 1, 2],
            openie_docs=openie_docs,
            evidence_set_size=5,
            max_path_edges=2,
            max_endpoint_degree=8,
        )

        self.assertEqual(result["retrieved_doc_indices"][:3], [0, 1, 2])
        self.assertTrue(result["selected_closed_units"])
        self.assertEqual(result["selected_closed_units"][0]["path_endpoint_keys"], ["phoenix", "arizona"])
        self.assertGreaterEqual(result["seed_unit_count"], 1)

    def test_query_seed_uses_structural_endpoint_tokens(self):
        openie_docs = [
            {
                "idx": "chunk-race",
                "passage": "Race\nRace was held in Phoenix.",
                "extracted_entities": ["Race", "Phoenix"],
                "extracted_triples": [["Race", "held in", "Phoenix"]],
            },
            {
                "idx": "chunk-city",
                "passage": "City\nPhoenix is in Arizona.",
                "extracted_entities": ["Phoenix", "Arizona"],
                "extracted_triples": [["Phoenix", "in", "Arizona"]],
            },
            {
                "idx": "chunk-person",
                "passage": "Person\nPerson was born in Arizona.",
                "extracted_entities": ["Person", "Arizona"],
                "extracted_triples": [["Person", "born in", "Arizona"]],
            },
        ]
        units = build_candidate_sto_units(candidate_doc_indices=[0, 1, 2], openie_docs=openie_docs)

        seed_ids = seed_unit_ids_for_query(
            query_tokens={"race"},
            candidate_units=units,
        )

        self.assertEqual(len(seed_ids), 1)

    def test_frontier_query_tokens_keep_rare_structural_tokens(self):
        openie_docs = [
            {
                "idx": "chunk-alpha",
                "passage": "Alpha\nAlpha was born in Paris.",
                "extracted_entities": ["Alpha", "Paris"],
                "extracted_triples": [["Alpha", "born in", "Paris"]],
            },
            {
                "idx": "chunk-beta",
                "passage": "Beta\nBeta was born in Paris.",
                "extracted_entities": ["Beta", "Paris"],
                "extracted_triples": [["Beta", "born in", "Paris"]],
            },
            {
                "idx": "chunk-gamma",
                "passage": "Gamma\nGamma was born in Lyon.",
                "extracted_entities": ["Gamma", "Lyon"],
                "extracted_triples": [["Gamma", "born in", "Lyon"]],
            },
        ]
        units = build_candidate_sto_units(candidate_doc_indices=[0, 1, 2], openie_docs=openie_docs)

        frontier = frontier_query_tokens(
            query_tokens={"born", "paris", "gamma"},
            candidate_units=units,
        )

        self.assertIn("gamma", frontier)
        self.assertNotIn("paris", frontier)

    def test_selector_respects_reader_budget_for_closed_units(self):
        openie_docs = [
            {
                "idx": "chunk-a",
                "passage": "A\nA links to B.",
                "extracted_entities": ["A", "B"],
                "extracted_triples": [["A", "links to", "B"]],
            },
            {
                "idx": "chunk-b",
                "passage": "B\nB links to C.",
                "extracted_entities": ["B", "C"],
                "extracted_triples": [["B", "links to", "C"]],
            },
            {
                "idx": "chunk-c",
                "passage": "C\nC links to D.",
                "extracted_entities": ["C", "D"],
                "extracted_triples": [["C", "links to", "D"]],
            },
            {
                "idx": "chunk-d",
                "passage": "D\nD links to E.",
                "extracted_entities": ["D", "E"],
                "extracted_triples": [["D", "links to", "E"]],
            },
        ]

        result = select_obligation_closed_evidence(
            query="A B C D E",
            candidate_doc_indices=[0, 1, 2, 3],
            openie_docs=openie_docs,
            evidence_set_size=2,
            max_path_edges=3,
            max_endpoint_degree=8,
        )

        self.assertLessEqual(len(result["retrieved_doc_indices"]), 2)
        for unit in result["selected_closed_units"]:
            self.assertLessEqual(len(unit["emitted_doc_indices"]), 2)


if __name__ == "__main__":
    unittest.main()
