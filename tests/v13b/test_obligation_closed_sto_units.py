import json
import tempfile
import unittest
from pathlib import Path

from build_obligation_closed_sto_units import (
    build_obligation_closed_sto_unit_store,
    main,
    usable_endpoint,
)
from build_source_title_openie_substrate import build_units_for_doc


def sto_units_from_docs(docs):
    units = []
    for doc_index, doc in enumerate(docs):
        units.extend(build_units_for_doc(doc, doc_index=doc_index, include_source_spans=False))
    return units


class ObligationClosedSTOUnitTests(unittest.TestCase):
    def test_materializes_event_location_performer_chain_as_closed_unit(self):
        units = sto_units_from_docs(
            [
                {
                    "idx": "chunk-959",
                    "passage": "Desert Diamond West Valley Phoenix Grand Prix\nThe Indy car race was held in Phoenix.",
                    "extracted_entities": ["Desert Diamond West Valley Phoenix Grand Prix", "Phoenix"],
                    "extracted_triples": [
                        ["Desert Diamond West Valley Phoenix Grand Prix", "held in", "Phoenix"]
                    ],
                },
                {
                    "idx": "chunk-951",
                    "passage": "Tucson, Arizona\nPhoenix is the largest city in Arizona.",
                    "extracted_entities": ["Phoenix", "Arizona"],
                    "extracted_triples": [["Phoenix", "largest city in", "Arizona"]],
                },
                {
                    "idx": "chunk-962",
                    "passage": "Charles Mingus\nCharles Mingus was from Arizona.",
                    "extracted_entities": ["Charles Mingus", "Arizona"],
                    "extracted_triples": [["Charles Mingus", "from", "Arizona"]],
                },
                {
                    "idx": "chunk-952",
                    "passage": "A Modern Jazz Symposium of Music and Poetry\nA Modern Jazz Symposium of Music and Poetry was performed by Charles Mingus.",
                    "extracted_entities": ["A Modern Jazz Symposium of Music and Poetry", "Charles Mingus"],
                    "extracted_triples": [
                        ["A Modern Jazz Symposium of Music and Poetry", "performed by", "Charles Mingus"]
                    ],
                },
            ]
        )

        store = build_obligation_closed_sto_unit_store(
            units,
            dataset_name="synthetic",
            max_path_edges=3,
            max_endpoint_degree=8,
        )

        matching = [
            unit
            for unit in store["closed_units"]
            if unit["terminal_doc_index"] == 0 and unit["anchor_doc_index"] == 3
        ]
        self.assertEqual(len(matching), 1)
        unit = matching[0]
        self.assertEqual(unit["closed_obligation_type"], "variable_transfer_path")
        self.assertEqual(unit["path_doc_indices"], [0, 1, 2, 3])
        self.assertEqual(unit["emitted_doc_indices"], [0, 1, 2, 3])
        self.assertEqual(unit["path_endpoint_keys"], ["phoenix", "arizona", "charles mingus"])
        self.assertEqual(unit["path_edge_count"], 3)
        self.assertEqual(len(unit["relation_roles"]), 4)

    def test_weak_endpoint_is_not_a_variable_transfer_edge(self):
        self.assertFalse(usable_endpoint("the city"))
        self.assertFalse(usable_endpoint("song"))
        units = sto_units_from_docs(
            [
                {
                    "idx": "chunk-alpha",
                    "passage": "Alpha\nThe city borders Beta.",
                    "extracted_entities": ["Beta"],
                    "extracted_triples": [["the city", "borders", "Beta"]],
                },
                {
                    "idx": "chunk-gamma",
                    "passage": "Gamma\nThe city was founded by Delta.",
                    "extracted_entities": ["Delta"],
                    "extracted_triples": [["the city", "founded by", "Delta"]],
                },
            ]
        )

        store = build_obligation_closed_sto_unit_store(units, max_path_edges=1, max_endpoint_degree=8)

        self.assertEqual(store["summary"]["closed_unit_count"], 0)
        self.assertNotIn("the city", {endpoint for unit in store["closed_units"] for endpoint in unit["path_endpoint_keys"]})

    def test_endpoint_degree_cap_removes_hub_transfer(self):
        units = sto_units_from_docs(
            [
                {
                    "idx": "chunk-a",
                    "passage": "A\nA mentions Hub.",
                    "extracted_entities": ["A", "Hub"],
                    "extracted_triples": [["A", "mentions", "Hub"]],
                },
                {
                    "idx": "chunk-b",
                    "passage": "B\nB mentions Hub.",
                    "extracted_entities": ["B", "Hub"],
                    "extracted_triples": [["B", "mentions", "Hub"]],
                },
                {
                    "idx": "chunk-c",
                    "passage": "C\nC mentions Hub.",
                    "extracted_entities": ["C", "Hub"],
                    "extracted_triples": [["C", "mentions", "Hub"]],
                },
            ]
        )

        store = build_obligation_closed_sto_unit_store(units, max_path_edges=1, max_endpoint_degree=2)

        self.assertEqual(store["summary"]["hub_endpoint_skipped_count"], 1)
        self.assertEqual(store["summary"]["closed_unit_count"], 0)

    def test_direct_shared_endpoint_creates_direct_closed_unit(self):
        units = sto_units_from_docs(
            [
                {
                    "idx": "chunk-a",
                    "passage": "A\nA was born in Delta.",
                    "extracted_entities": ["A", "Delta"],
                    "extracted_triples": [["A", "born in", "Delta"]],
                },
                {
                    "idx": "chunk-b",
                    "passage": "B\nB is located in Delta.",
                    "extracted_entities": ["B", "Delta"],
                    "extracted_triples": [["B", "located in", "Delta"]],
                },
            ]
        )

        store = build_obligation_closed_sto_unit_store(units, max_path_edges=1, max_endpoint_degree=8)

        self.assertEqual(store["summary"]["closed_unit_type_counts"], {"direct_variable_transfer": 2})
        self.assertEqual({unit["path_endpoint_keys"][0] for unit in store["closed_units"]}, {"delta"})

    def test_closed_unit_carries_all_facts_from_emitted_docs(self):
        units = sto_units_from_docs(
            [
                {
                    "idx": "chunk-race",
                    "passage": "Race Page\nMario won the race. The race was held in Phoenix.",
                    "extracted_entities": ["Mario", "race", "Phoenix"],
                    "extracted_triples": [
                        ["Mario", "won", "race"],
                        ["race", "held in", "Phoenix"],
                    ],
                },
                {
                    "idx": "chunk-city",
                    "passage": "City Page\nPhoenix is in Arizona.",
                    "extracted_entities": ["Phoenix", "Arizona"],
                    "extracted_triples": [["Phoenix", "in", "Arizona"]],
                },
            ]
        )

        store = build_obligation_closed_sto_unit_store(units, max_path_edges=1, max_endpoint_degree=8)

        matching = [
            unit
            for unit in store["closed_units"]
            if unit["terminal_doc_index"] == 0 and unit["anchor_doc_index"] == 1
        ]
        self.assertEqual(len(matching), 1)
        doc0_fact_ids = matching[0]["emitted_doc_fact_ids"]["0"]
        self.assertEqual(len(doc0_fact_ids), 2)

    def test_non_transfer_facts_are_payload_not_source_path_nodes(self):
        units = sto_units_from_docs(
            [
                {
                    "idx": "chunk-race",
                    "passage": "Race Page\nMario won the race. The race was held in Phoenix.",
                    "extracted_entities": ["Mario", "race", "Phoenix"],
                    "extracted_triples": [
                        ["Mario", "won", "race"],
                        ["race", "held in", "Phoenix"],
                    ],
                },
                {
                    "idx": "chunk-city",
                    "passage": "City Page\nPhoenix is in Arizona.",
                    "extracted_entities": ["Phoenix", "Arizona"],
                    "extracted_triples": [["Phoenix", "in", "Arizona"]],
                },
            ]
        )

        store = build_obligation_closed_sto_unit_store(units, max_path_edges=1, max_endpoint_degree=8)

        self.assertEqual(store["summary"]["non_transfer_fact_excluded_from_source_edge_count"], 1)
        matching = [
            unit
            for unit in store["closed_units"]
            if unit["terminal_doc_index"] == 0 and unit["anchor_doc_index"] == 1
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["source_edge_count"], 0)
        self.assertEqual(len(matching[0]["emitted_doc_fact_ids"]["0"]), 2)

    def test_cli_writes_store_and_markdown(self):
        units = sto_units_from_docs(
            [
                {
                    "idx": "chunk-a",
                    "passage": "A\nA was born in Delta.",
                    "extracted_entities": ["A", "Delta"],
                    "extracted_triples": [["A", "born in", "Delta"]],
                },
                {
                    "idx": "chunk-b",
                    "passage": "B\nB is located in Delta.",
                    "extracted_entities": ["B", "Delta"],
                    "extracted_triples": [["B", "located in", "Delta"]],
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "sto.jsonl"
            output_json = root / "closed.json"
            output_md = root / "closed.md"
            input_path.write_text(
                "\n".join(json.dumps(unit, sort_keys=True) for unit in units) + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--dataset-name",
                    "synthetic",
                    "--sto-units-jsonl",
                    str(input_path),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                    "--max-path-edges",
                    "1",
                    "--max-endpoint-degree",
                    "8",
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["closed_unit_count"], 2)
            self.assertTrue(output_md.exists())


if __name__ == "__main__":
    unittest.main()
