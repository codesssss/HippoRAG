import unittest

from evaluate_obligation_closed_sto_local_ppr import (
    build_fact_out_neighbors,
    query_triples_from_row,
    reverse_target_set_local_ppr,
    score_closed_units_by_source_ppr,
    select_obligation_closed_local_ppr_evidence,
    source_set_local_ppr,
)
from evaluate_obligation_closed_sto_selector import build_candidate_sto_units


class ObligationClosedSTOLocalPPRTests(unittest.TestCase):
    def test_reverse_target_local_ppr_matches_chain_contribution(self):
        result = reverse_target_set_local_ppr(
            out_neighbors={
                "source": ["middle"],
                "middle": ["target"],
                "target": [],
            },
            target_node_ids=["target"],
            alpha=0.2,
            residual_epsilon=1e-12,
        )

        estimate = result["estimate"]
        self.assertAlmostEqual(estimate["target"], 0.2)
        self.assertAlmostEqual(estimate["middle"], 0.16)
        self.assertAlmostEqual(estimate["source"], 0.128)
        self.assertFalse(result["truncated"])

    def test_source_local_ppr_matches_chain_distribution(self):
        result = source_set_local_ppr(
            out_neighbors={
                "source": ["middle"],
                "middle": ["target"],
                "target": [],
            },
            source_node_ids=["source"],
            alpha=0.2,
            residual_epsilon=1e-12,
        )

        estimate = result["estimate"]
        self.assertAlmostEqual(estimate["source"], 0.2)
        self.assertAlmostEqual(estimate["middle"], 0.16)
        self.assertAlmostEqual(estimate["target"], 0.128)
        self.assertFalse(result["truncated"])

    def test_source_ppr_scores_reachable_unit_above_lexical_noise(self):
        scored = score_closed_units_by_source_ppr(
            closed_units=[
                {
                    "closed_evidence_unit_id": "correct",
                    "path_unit_ids": ["answer"],
                    "emitted_doc_indices": [0, 1, 2],
                    "path_edge_count": 2,
                    "source_edge_count": 0,
                },
                {
                    "closed_evidence_unit_id": "noise",
                    "path_unit_ids": ["unreachable_noise"],
                    "emitted_doc_indices": [9],
                    "path_edge_count": 1,
                    "source_edge_count": 0,
                },
            ],
            seed_unit_ids=["query_seed"],
            out_neighbors={
                "query_seed": ["bridge"],
                "bridge": ["answer"],
                "answer": [],
                "unreachable_noise": [],
            },
            alpha=0.2,
            residual_epsilon=1e-12,
        )

        by_id = {unit["closed_evidence_unit_id"]: unit for unit in scored}
        self.assertGreater(by_id["correct"]["closed_unit_local_ppr_score"], 0.0)
        self.assertEqual(by_id["noise"]["closed_unit_local_ppr_score"], 0.0)

    def test_fact_graph_uses_endpoint_and_source_edges(self):
        openie_docs = [
            {
                "idx": "chunk-a",
                "passage": "A\nA links to Shared.",
                "extracted_entities": ["A", "Shared"],
                "extracted_triples": [["A", "links to", "Shared"]],
            },
            {
                "idx": "chunk-b",
                "passage": "B\nB mentions Shared and Local.",
                "extracted_entities": ["B", "Shared", "Local"],
                "extracted_triples": [
                    ["B", "mentions", "Shared"],
                    ["B", "also mentions", "Local"],
                ],
            },
        ]
        candidate_units = build_candidate_sto_units(candidate_doc_indices=[0, 1], openie_docs=openie_docs)

        out_neighbors, stats = build_fact_out_neighbors(candidate_units=candidate_units, max_endpoint_degree=8)

        self.assertGreaterEqual(stats["fact_node_count"], 3)
        self.assertGreaterEqual(stats["directed_fact_edge_count"], 2)
        self.assertTrue(any(neighbors for neighbors in out_neighbors.values()))

    def test_selector_uses_ppr_closed_unit_not_token_coverage(self):
        openie_docs = [
            {
                "idx": "chunk-race",
                "passage": "Indy race\nThe Indy race was held in Phoenix.",
                "extracted_entities": ["Indy race", "Phoenix"],
                "extracted_triples": [["Indy race", "held in", "Phoenix"]],
            },
            {
                "idx": "chunk-city",
                "passage": "Phoenix\nPhoenix is in Arizona.",
                "extracted_entities": ["Phoenix", "Arizona"],
                "extracted_triples": [["Phoenix", "in", "Arizona"]],
            },
            {
                "idx": "chunk-performer",
                "passage": "Charles Mingus\nCharles Mingus was from Arizona.",
                "extracted_entities": ["Charles Mingus", "Arizona"],
                "extracted_triples": [["Charles Mingus", "from", "Arizona"]],
            },
            {
                "idx": "chunk-album",
                "passage": "Album\nA Modern Jazz Symposium of Music and Poetry was performed by Charles Mingus.",
                "extracted_entities": [
                    "A Modern Jazz Symposium of Music and Poetry",
                    "Charles Mingus",
                ],
                "extracted_triples": [
                    [
                        "A Modern Jazz Symposium of Music and Poetry",
                        "performed by",
                        "Charles Mingus",
                    ]
                ],
            },
            {
                "idx": "chunk-noise",
                "passage": "Noise\nThis passage repeats Indy race city state performer modern jazz symposium music poetry but has no transfer endpoint.",
                "extracted_entities": ["Noise"],
                "extracted_triples": [["Noise", "repeats", "unconnected words"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query=(
                "Who won the Indy race in the city in the state where the performer "
                "of A Modern Jazz Symposium of Music and Poetry is from?"
            ),
            query_triples=[
                ["Indy race", "held in", "Phoenix"],
                ["Phoenix", "in", "Arizona"],
                ["Charles Mingus", "from", "Arizona"],
                ["A Modern Jazz Symposium of Music and Poetry", "performed by", "Charles Mingus"],
            ],
            candidate_doc_indices=[4, 0, 1, 2, 3],
            openie_docs=openie_docs,
            evidence_set_size=5,
            max_path_edges=3,
            max_endpoint_degree=8,
            alpha=0.2,
            residual_epsilon=1e-12,
        )

        self.assertTrue(result["selected_closed_units"])
        self.assertEqual(
            set(result["selected_closed_units"][0]["path_endpoint_keys"]),
            {"phoenix", "arizona", "charles mingus"},
        )
        self.assertEqual(set(result["retrieved_doc_indices"][:4]), {0, 1, 2, 3})
        self.assertNotEqual(result["retrieved_doc_indices"][0], 4)

    def test_selector_abstains_to_prior_without_query_obligations(self):
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
                "idx": "chunk-noise",
                "passage": "Noise\nNoise repeats A B C.",
                "extracted_entities": ["Noise"],
                "extracted_triples": [["Noise", "repeats", "A B C"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="What links to C?",
            candidate_doc_indices=[2, 0, 1],
            fallback_doc_indices=[0, 1, 2],
            openie_docs=openie_docs,
            evidence_set_size=2,
        )

        self.assertEqual(result["seed_source"], "no_query_obligation_abstain_to_prior")
        self.assertEqual(result["retrieved_doc_indices"], [0, 1])
        self.assertEqual(result["selected_closed_units"], [])

    def test_selector_abstains_when_query_obligation_program_is_not_fully_grounded(self):
        openie_docs = [
            {
                "idx": "chunk-a",
                "passage": "A\nA links to B.",
                "extracted_entities": ["A", "B"],
                "extracted_triples": [["A", "links to", "B"]],
            },
            {
                "idx": "chunk-b",
                "passage": "B\nB has no death date.",
                "extracted_entities": ["B"],
                "extracted_triples": [["B", "mentions", "unknown"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="When did B linked from A die?",
            query_triples=[
                ["A", "links to", "?x"],
                ["?x", "died in", "?date"],
            ],
            candidate_doc_indices=[1, 0],
            fallback_doc_indices=[0, 1],
            openie_docs=openie_docs,
            evidence_set_size=2,
        )

        self.assertEqual(result["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(result["query_obligation_grounded_count"], 1)
        self.assertEqual(result["retrieved_doc_indices"], [0, 1])
        self.assertEqual(result["selected_closed_units"], [])

    def test_typed_retrieval_critical_ablation_ignores_temporal_answer_extraction_blocker(self):
        openie_docs = [
            {
                "idx": "chunk-album",
                "passage": "Album Y\nAlbum Y was published by Alice.",
                "extracted_entities": ["Album Y", "Alice"],
                "extracted_triples": [["Album Y", "published by", "Alice"]],
            },
            {
                "idx": "chunk-prior",
                "passage": "Prior\nA prior document.",
                "extracted_entities": ["Prior"],
                "extracted_triples": [["Prior", "mentions", "something"]],
            },
        ]
        query_triples = [
            ["Album Y", "published by", "?publisher"],
            ["?publisher", "ended in", "?year"],
        ]

        untyped = select_obligation_closed_local_ppr_evidence(
            query="What year did the publisher of Album Y end?",
            query_triples=query_triples,
            candidate_doc_indices=[1, 0],
            fallback_doc_indices=[1, 0],
            openie_docs=openie_docs,
            evidence_set_size=1,
        )
        self.assertEqual(untyped["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(untyped["retrieved_doc_indices"], [1])

        typed = select_obligation_closed_local_ppr_evidence(
            query="What year did the publisher of Album Y end?",
            query_triples=query_triples,
            candidate_doc_indices=[1, 0],
            fallback_doc_indices=[1, 0],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=1,
            max_path_edges=1,
            max_endpoint_degree=8,
            alpha=0.2,
            residual_epsilon=1e-12,
        )

        self.assertEqual(typed["query_obligation_count"], 1)
        self.assertEqual(typed["materialized_query_obligation_count"], 2)
        self.assertEqual(typed["typed_materialized_non_retrieval_obligation_count"], 1)
        self.assertEqual(typed["typed_retrieval_critical_count"], 1)
        self.assertEqual(typed["typed_retrieval_critical_grounded_count"], 1)
        self.assertEqual(typed["seed_source"], "query_obligation_program_evidence_assembler")
        self.assertEqual(typed["retrieved_doc_indices"], [0])

    def test_typed_retrieval_critical_ablation_abstains_on_builder_filtered_critical_triple(self):
        openie_docs = [
            {
                "idx": "chunk-castle",
                "passage": "Bytham Castle\nBytham Castle is in Castle Bytham.",
                "extracted_entities": ["Bytham Castle", "Castle Bytham"],
                "extracted_triples": [["Bytham Castle", "located in", "Castle Bytham"]],
            },
            {
                "idx": "chunk-prior",
                "passage": "Prior\nA prior document.",
                "extracted_entities": ["Prior"],
                "extracted_triples": [["Prior", "mentions", "something"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="Bytham Castle is a castle in the civil parish of how many houses?",
            query_triples=[
                ["Bytham Castle", "located in", "?x1"],
                ["?x1", "has", "?x2"],
            ],
            candidate_doc_indices=[1, 0],
            fallback_doc_indices=[1, 0],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            evidence_set_size=1,
        )

        self.assertEqual(result["seed_source"], "typed_retrieval_critical_unmaterialized_abstain_to_prior")
        self.assertEqual(result["abstention_reason"], "retrieval_critical_unmaterialized")
        self.assertEqual(result["typed_retrieval_critical_count"], 2)
        self.assertEqual(result["typed_retrieval_critical_unmaterialized_count"], 1)
        self.assertEqual(result["retrieved_doc_indices"], [1])

    def test_source_span_grounding_can_complete_role_anchored_obligation_when_enabled(self):
        openie_docs = [
            {
                "idx": "chunk-book",
                "passage": "Book A\nBook A was published in 1946.",
                "extracted_entities": ["Book A", "1946"],
                "extracted_triples": [["Book A", "published on", "1946"]],
            },
            {
                "idx": "chunk-prior",
                "passage": "Prior\nA prior document.",
                "extracted_entities": ["Prior"],
                "extracted_triples": [["Prior", "mentions", "something"]],
            },
        ]
        query_triples = [["Book A", "was published in", "?date"]]

        strict = select_obligation_closed_local_ppr_evidence(
            query="Which book was published first?",
            query_triples=query_triples,
            candidate_doc_indices=[1, 0],
            fallback_doc_indices=[1, 0],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_support_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=1,
            max_path_edges=1,
            max_endpoint_degree=8,
        )
        self.assertEqual(strict["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(strict["retrieved_doc_indices"], [1])

        span = select_obligation_closed_local_ppr_evidence(
            query="Which book was published first?",
            query_triples=query_triples,
            candidate_doc_indices=[1, 0],
            fallback_doc_indices=[1, 0],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_support_obligation_grounding=True,
            allow_source_span_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=1,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        self.assertEqual(span["seed_source"], "query_obligation_program_evidence_assembler")
        self.assertEqual(span["query_obligation_source_span_grounded_count"], 1)
        self.assertEqual(span["retrieved_doc_indices"], [0])
        span_matches = [
            match
            for matches in span["source_span_grounded_obligation_matches"].values()
            for match in matches
        ]
        self.assertEqual(len(span_matches), 1)
        self.assertIn("source_span_grounding", span_matches[0]["source_span_grounding_reasons"])
        self.assertIn("publish", span_matches[0]["relation_token_hits"])
        self.assertEqual(span_matches[0]["source_span_transport_endpoint_keys"], ["book a"])
        self.assertTrue(span_matches[0]["source_span_tuple_relation_aligned"])

    def test_typed_comparison_program_assembles_components_without_fake_bridge(self):
        openie_docs = [
            {
                "idx": "film-a",
                "passage": "Film A\nFilm A was directed by Alice.",
                "extracted_entities": ["Film A", "Alice"],
                "extracted_triples": [["Film A", "directed by", "Alice"]],
            },
            {
                "idx": "alice",
                "passage": "Alice\nAlice was born in 1950.",
                "extracted_entities": ["Alice", "1950"],
                "extracted_triples": [["Alice", "was born in", "1950"]],
            },
            {
                "idx": "film-b",
                "passage": "Film B\nFilm B was directed by Bob.",
                "extracted_entities": ["Film B", "Bob"],
                "extracted_triples": [["Film B", "directed by", "Bob"]],
            },
            {
                "idx": "bob",
                "passage": "Bob\nBob was born in 1960.",
                "extracted_entities": ["Bob", "1960"],
                "extracted_triples": [["Bob", "was born in", "1960"]],
            },
        ]
        result = select_obligation_closed_local_ppr_evidence(
            query="Which film has the director who was born first, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x2"],
                ["?x1", "was born in", "?date1"],
                ["?x2", "was born in", "?date2"],
                ["?date1", "earlier than", "?date2"],
            ],
            candidate_doc_indices=[0, 1, 2, 3],
            fallback_doc_indices=[0, 1, 2, 3],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=4,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        program = result["program_evidence_assembly"]
        self.assertTrue(program["feasible"])
        self.assertEqual(program["assembly_mode"], "typed_operator_component_cover")
        self.assertEqual(program["typed_component_reason"], "comparison")
        self.assertEqual(program["path_edges"], [])
        self.assertEqual(set(result["retrieved_doc_indices"]), {0, 1, 2, 3})

    def test_older_than_program_lowers_to_birth_date_operand_evidence(self):
        openie_docs = [
            {
                "idx": "airheads",
                "passage": "Airheads\nAirheads was directed by Michael Lehmann.",
                "extracted_entities": ["Airheads", "Michael Lehmann"],
                "extracted_triples": [["Airheads", "directed by", "Michael Lehmann"]],
            },
            {
                "idx": "michael",
                "passage": "Michael Lehmann\nMichael Lehmann was born on March 30, 1957.",
                "extracted_entities": ["Michael Lehmann", "March 30, 1957"],
                "extracted_triples": [["Michael Lehmann", "born on", "March 30, 1957"]],
            },
            {
                "idx": "cabin",
                "passage": "Return To Cabin By The Lake\nReturn To Cabin By The Lake was directed by Po-Chih Leong.",
                "extracted_entities": ["Return To Cabin By The Lake", "Po-Chih Leong"],
                "extracted_triples": [["Return To Cabin By The Lake", "directed by", "Po-Chih Leong"]],
            },
            {
                "idx": "leong",
                "passage": "Po-Chih Leong\nPo-Chih Leong was born on 31 December 1939.",
                "extracted_entities": ["Po-Chih Leong", "31 December 1939"],
                "extracted_triples": [["Po-Chih Leong", "born on", "31 December 1939"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="Which film has the director who is older than the other, Airheads or Return To Cabin By The Lake?",
            query_triples=[
                ["Airheads", "directed by", "?x1"],
                ["Return To Cabin By The Lake", "directed by", "?x2"],
                ["?x1", "is older than", "?x2"],
            ],
            candidate_doc_indices=[0, 1, 2, 3],
            fallback_doc_indices=[0, 2],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=4,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        program = result["program_evidence_assembly"]
        self.assertTrue(program["feasible"])
        self.assertEqual(program["assembly_mode"], "typed_operator_component_cover")
        self.assertIn(["?x1", "born on", "?date1"], result["program_query_triples"])
        self.assertIn(["?x2", "born on", "?date2"], result["program_query_triples"])
        self.assertEqual(set(result["retrieved_doc_indices"]), {0, 1, 2, 3})
        self.assertEqual(
            {tuple(row["fact"]) for row in program["path_facts"]},
            {
                ("Airheads", "directed by", "Michael Lehmann"),
                ("Michael Lehmann", "born on", "March 30, 1957"),
                ("Return To Cabin By The Lake", "directed by", "Po-Chih Leong"),
                ("Po-Chih Leong", "born on", "31 December 1939"),
            },
        )

    def test_born_later_program_lowers_to_birth_date_operand_evidence(self):
        openie_docs = [
            {
                "idx": "film-a",
                "passage": "Film A\nFilm A was directed by Alice.",
                "extracted_entities": ["Film A", "Alice"],
                "extracted_triples": [["Film A", "directed by", "Alice"]],
            },
            {
                "idx": "alice",
                "passage": "Alice\nAlice was born on 1 January 1990.",
                "extracted_entities": ["Alice", "1 January 1990"],
                "extracted_triples": [["Alice", "born on", "1 January 1990"]],
            },
            {
                "idx": "film-b",
                "passage": "Film B\nFilm B was directed by Bob.",
                "extracted_entities": ["Film B", "Bob"],
                "extracted_triples": [["Film B", "directed by", "Bob"]],
            },
            {
                "idx": "bob",
                "passage": "Bob\nBob was born on 2 February 2001.",
                "extracted_entities": ["Bob", "2 February 2001"],
                "extracted_triples": [["Bob", "born on", "2 February 2001"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="Which film has the director who was born later, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x2"],
                ["?x1", "was born later than", "?x2"],
            ],
            candidate_doc_indices=[0, 1, 2, 3],
            fallback_doc_indices=[],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=4,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        program = result["program_evidence_assembly"]
        self.assertTrue(program["feasible"])
        self.assertEqual(program["assembly_mode"], "typed_operator_component_cover")
        self.assertIn(["?x1", "born on", "?date1"], result["program_query_triples"])
        self.assertIn(["?x2", "born on", "?date2"], result["program_query_triples"])
        self.assertEqual(set(result["retrieved_doc_indices"]), {0, 1, 2, 3})

    def test_born_later_program_repairs_duplicate_branch_variable(self):
        openie_docs = [
            {
                "idx": "film-a",
                "passage": "Film A\nFilm A was directed by Alice.",
                "extracted_entities": ["Film A", "Alice"],
                "extracted_triples": [["Film A", "directed by", "Alice"]],
            },
            {
                "idx": "alice",
                "passage": "Alice\nAlice was born in 1990.",
                "extracted_entities": ["Alice", "1990"],
                "extracted_triples": [["Alice", "was born in", "1990"]],
            },
            {
                "idx": "film-b",
                "passage": "Canonical Film B\nCanonical Film B, released as Film B, was directed by Bob.",
                "extracted_entities": ["Canonical Film B", "Film B", "Bob"],
                "extracted_triples": [
                    ["Canonical Film B", "released as", "Film B"],
                    ["Canonical Film B", "directed by", "Bob"],
                    ["Bob", "directed", "Canonical Film B"],
                ],
            },
            {
                "idx": "bob",
                "passage": "Bob\nBob was born on 2 February 2001.",
                "extracted_entities": ["Bob", "2 February 2001"],
                "extracted_triples": [["Bob", "born on", "2 February 2001"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="Which film has the director who was born later, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x1"],
                ["?x1", "was born in", "?date"],
                ["?x1", "was born later than", "?x2"],
            ],
            candidate_doc_indices=[0, 1, 2, 3],
            fallback_doc_indices=[],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_support_obligation_grounding=True,
            allow_source_span_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=4,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        program = result["program_evidence_assembly"]
        self.assertTrue(program["feasible"])
        self.assertEqual(program["assembly_mode"], "typed_operator_component_cover")
        self.assertIn(["Film B", "directed by", "?x2"], result["program_query_triples"])
        self.assertNotIn(["Film B", "directed by", "?x1"], result["program_query_triples"])
        self.assertIn(["?date", "earlier than", "?date2"], result["program_query_triples"])
        self.assertEqual(result["typed_retrieval_critical_grounded_count"], 4)
        self.assertEqual(result["typed_retrieval_critical_count"], 4)
        self.assertEqual(result["query_obligation_support_grounded_count"], 1)
        self.assertEqual(result["query_obligation_source_span_grounded_count"], 0)
        support_bindings = [
            match.get("variable_bindings", {})
            for matches in result["support_grounded_obligation_matches"].values()
            for match in matches
        ]
        self.assertEqual(support_bindings, [{"x2": ["bob"]}])
        self.assertIn(
            ("Bob", "born on", "2 February 2001"),
            {tuple(row["fact"]) for row in program["path_facts"]},
        )
        self.assertEqual(set(result["retrieved_doc_indices"]), {0, 1, 2, 3})

    def test_released_first_program_lowers_to_release_date_operand_evidence(self):
        openie_docs = [
            {
                "idx": "film-a",
                "passage": "Film A\nFilm A was released in 1946.",
                "extracted_entities": ["Film A", "1946"],
                "extracted_triples": [["Film A", "released in", "1946"]],
            },
            {
                "idx": "film-b",
                "passage": "Film B\nFilm B was released in 1952.",
                "extracted_entities": ["Film B", "1952"],
                "extracted_triples": [["Film B", "released in", "1952"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="Which film was released first, Film A or Film B?",
            query_triples=[["Film A", "earlier than", "Film B"]],
            candidate_doc_indices=[0, 1],
            fallback_doc_indices=[],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=2,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        program = result["program_evidence_assembly"]
        self.assertTrue(program["feasible"])
        self.assertEqual(program["assembly_mode"], "typed_operator_component_cover")
        self.assertIn(["Film A", "released in", "?date1"], result["program_query_triples"])
        self.assertIn(["Film B", "released in", "?date2"], result["program_query_triples"])
        self.assertEqual(set(result["retrieved_doc_indices"]), {0, 1})

    def test_died_later_program_lowers_to_death_date_operand_evidence(self):
        openie_docs = [
            {
                "idx": "film-a",
                "passage": "Film A\nFilm A was directed by Alice.",
                "extracted_entities": ["Film A", "Alice"],
                "extracted_triples": [["Film A", "directed by", "Alice"]],
            },
            {
                "idx": "alice",
                "passage": "Alice\nAlice died on 1 January 1990.",
                "extracted_entities": ["Alice", "1 January 1990"],
                "extracted_triples": [["Alice", "died on", "1 January 1990"]],
            },
            {
                "idx": "film-b",
                "passage": "Film B\nFilm B was directed by Bob.",
                "extracted_entities": ["Film B", "Bob"],
                "extracted_triples": [["Film B", "directed by", "Bob"]],
            },
            {
                "idx": "bob",
                "passage": "Bob\nBob died on 2 February 2001.",
                "extracted_entities": ["Bob", "2 February 2001"],
                "extracted_triples": [["Bob", "died on", "2 February 2001"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="Which film has the director who died later, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x2"],
                ["?x1", "died later than", "?x2"],
            ],
            candidate_doc_indices=[0, 1, 2, 3],
            fallback_doc_indices=[],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=4,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        program = result["program_evidence_assembly"]
        self.assertTrue(program["feasible"])
        self.assertEqual(program["assembly_mode"], "typed_operator_component_cover")
        self.assertIn(["?x1", "died on", "?date1"], result["program_query_triples"])
        self.assertIn(["?x2", "died on", "?date2"], result["program_query_triples"])
        self.assertEqual(set(result["retrieved_doc_indices"]), {0, 1, 2, 3})

    def test_temporal_event_component_uses_exact_relation_after_canonicalization(self):
        openie_docs = [
            {
                "idx": "film-a",
                "passage": "Film A\nFilm A was directed by Alice.",
                "extracted_entities": ["Film A", "Alice"],
                "extracted_triples": [["Film A", "directed by", "Alice"]],
            },
            {
                "idx": "alice",
                "passage": "Alice\nAlice born 1950 is a screenwriter.",
                "extracted_entities": ["Alice", "1950", "London", "screenwriter"],
                "extracted_triples": [
                    ["Alice", "was born in", "London"],
                    ["Alice", "born on", "1950"],
                    ["Alice", "is", "screenwriter"],
                ],
            },
            {
                "idx": "film-b",
                "passage": "Film B\nFilm B was directed by Bob.",
                "extracted_entities": ["Film B", "Bob"],
                "extracted_triples": [["Film B", "directed by", "Bob"]],
            },
            {
                "idx": "bob",
                "passage": "Bob\nBob born 1960 is a screenwriter.",
                "extracted_entities": ["Bob", "1960", "Paris", "screenwriter"],
                "extracted_triples": [
                    ["Bob", "was born in", "Paris"],
                    ["Bob", "born on", "1960"],
                    ["Bob", "is", "screenwriter"],
                ],
            },
        ]
        result = select_obligation_closed_local_ppr_evidence(
            query="Which film has the director who was born first, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x2"],
                ["?x1", "was born in", "?date1"],
                ["?x2", "was born in", "?date2"],
                ["?date1", "earlier than", "?date2"],
            ],
            candidate_doc_indices=[0, 1, 2, 3],
            fallback_doc_indices=[0, 1, 2, 3],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_source_span_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=4,
            max_path_edges=1,
            max_endpoint_degree=8,
        )

        program = result["program_evidence_assembly"]
        self.assertEqual(program["assembly_mode"], "typed_operator_component_cover")
        self.assertEqual(
            result["typed_variable_type_constraints"],
            {"date1": "temporal", "date2": "temporal"},
        )
        source_span_facts = [
            row for row in program["path_facts"] if row.get("evidence_grounding") == "source_span"
        ]
        self.assertEqual(source_span_facts, [])
        self.assertEqual(result["query_obligation_source_span_grounded_count"], 0)
        self.assertEqual(result["typed_retrieval_critical_grounded_count"], 4)

    def test_source_span_grounding_rejects_weak_relation_only_or_missing_role_anchor(self):
        openie_docs = [
            {
                "idx": "chunk-film",
                "passage": "Film A\nFilm A is a 1946 Indian film.",
                "extracted_entities": ["Film A", "1946"],
                "extracted_triples": [["Actor", "co-starred with", "Other"]],
            },
            {
                "idx": "chunk-wrong-death",
                "passage": "Michel de Klerk\nMichel de Klerk died in Amsterdam.",
                "extracted_entities": ["Michel de Klerk", "Amsterdam"],
                "extracted_triples": [["Michel de Klerk", "died in", "Amsterdam"]],
            },
        ]

        weak = select_obligation_closed_local_ppr_evidence(
            query="Which film was released first?",
            query_triples=[["Film A", "was released in", "?date"]],
            candidate_doc_indices=[0],
            fallback_doc_indices=[0],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_source_span_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=1,
        )
        self.assertEqual(weak["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(weak["query_obligation_source_span_grounded_count"], 0)

        missing_anchor = select_obligation_closed_local_ppr_evidence(
            query="Where did Titian die?",
            query_triples=[["Titian", "died in", "?place"]],
            candidate_doc_indices=[1],
            fallback_doc_indices=[1],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_source_span_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=1,
        )
        self.assertEqual(missing_anchor["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(missing_anchor["query_obligation_source_span_grounded_count"], 0)

    def test_source_span_grounding_can_materialize_concrete_year_attribute(self):
        openie_docs = [
            {
                "idx": "chunk-film",
                "passage": "Summer Wars\nSummer Wars is a 2009 Japanese animated science fiction film.",
                "extracted_entities": ["Summer Wars", "2009"],
                "extracted_triples": [["Summer Wars", "is a", "2009 Japanese animated science fiction film"]],
            }
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="What year was Summer Wars?",
            query_triples=[["Summer Wars", "year", "2009"]],
            candidate_doc_indices=[0],
            fallback_doc_indices=[0],
            openie_docs=openie_docs,
            use_typed_retrieval_critical_obligations=True,
            allow_source_span_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=1,
        )

        self.assertEqual(result["query_obligation_source_span_grounded_count"], 1)
        self.assertEqual(result["seed_source"], "query_obligation_program_evidence_assembler")
        self.assertEqual(result["retrieved_doc_indices"], [0])

    def test_support_grounding_can_complete_program_only_when_enabled(self):
        openie_docs = [
            {
                "idx": "chunk-album",
                "passage": "Album Y\nAlbum Y is an album by Alice.",
                "extracted_entities": ["Album Y", "Alice"],
                "extracted_triples": [["Album Y", "is an album by", "Alice"]],
            },
            {
                "idx": "chunk-person",
                "passage": "Alice\nAlice was born in Arizona.",
                "extracted_entities": ["Alice", "Arizona"],
                "extracted_triples": [["Alice", "was born in", "Arizona"]],
            },
            {
                "idx": "chunk-city",
                "passage": "Phoenix\nPhoenix is located in Arizona.",
                "extracted_entities": ["Phoenix", "Arizona"],
                "extracted_triples": [["Phoenix", "is located in", "Arizona"]],
            },
            {
                "idx": "chunk-race",
                "passage": (
                    "Phoenix Grand Prix\n"
                    "The Indy car race was held at Phoenix in 2016. "
                    "Phoenix has a history of open wheel races."
                ),
                "extracted_entities": ["Phoenix Grand Prix", "Phoenix", "2016", "open wheel races"],
                "extracted_triples": [
                    ["Phoenix Grand Prix", "was held in", "2016"],
                    ["Phoenix", "has a history of", "open wheel races"],
                ],
            },
            {
                "idx": "chunk-noise",
                "passage": "Noise\nNoise repeats album race city state words without facts.",
                "extracted_entities": ["Noise"],
                "extracted_triples": [["Noise", "repeats", "words"]],
            },
        ]
        query_triples = [
            ["Album Y", "is an album by", "?x1"],
            ["?x1", "was born in", "?state"],
            ["?city", "located in", "?state"],
            ["?race", "held in", "?city"],
        ]

        strict = select_obligation_closed_local_ppr_evidence(
            query="Who won the race in the city in the state where Album Y's performer was born?",
            query_triples=query_triples,
            candidate_doc_indices=[4, 0, 1, 2, 3],
            fallback_doc_indices=[4, 0, 1, 2, 3],
            openie_docs=openie_docs,
            evidence_set_size=5,
        )
        self.assertEqual(strict["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(strict["query_obligation_grounded_count"], 3)
        self.assertEqual(strict["query_obligation_support_grounded_count"], 0)

        support = select_obligation_closed_local_ppr_evidence(
            query="Who won the race in the city in the state where Album Y's performer was born?",
            query_triples=query_triples,
            candidate_doc_indices=[4, 0, 1, 2, 3],
            fallback_doc_indices=[4, 0, 1, 2, 3],
            openie_docs=openie_docs,
            allow_support_obligation_grounding=True,
            evidence_set_size=5,
            max_path_edges=3,
            max_endpoint_degree=8,
            alpha=0.2,
            residual_epsilon=1e-12,
        )

        self.assertNotEqual(support["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(support["query_obligation_grounded_count"], 4)
        self.assertEqual(support["query_obligation_exact_grounded_count"], 3)
        self.assertEqual(support["query_obligation_support_grounded_count"], 1)
        support_matches = [
            match
            for matches in support["support_grounded_obligation_matches"].values()
            for match in matches
        ]
        self.assertEqual(len(support_matches), 1)
        self.assertEqual(support_matches[0]["doc_index"], 3)
        self.assertIn("relation_exact", support_matches[0]["support_grounding_reasons"])
        self.assertIn("source_text_anchor_hit", support_matches[0]["support_grounding_reasons"])
        self.assertIn("race", support_matches[0]["obligation_cue_token_hits"])

        assembled = select_obligation_closed_local_ppr_evidence(
            query="Who won the race in the city in the state where Album Y's performer was born?",
            query_triples=query_triples,
            candidate_doc_indices=[4, 0, 1, 2, 3],
            fallback_doc_indices=[4, 0, 1, 2, 3],
            openie_docs=openie_docs,
            allow_support_obligation_grounding=True,
            allow_program_evidence_assembler=True,
            evidence_set_size=5,
            max_path_edges=3,
            max_endpoint_degree=8,
        )

        self.assertEqual(assembled["seed_source"], "query_obligation_program_evidence_assembler")
        self.assertTrue(assembled["program_evidence_assembly"]["feasible"])
        self.assertEqual(
            set(assembled["program_evidence_assembly"]["covered_query_obligation_ids"]),
            {item["obligation_id"] for item in assembled["query_obligations"]},
        )
        self.assertEqual(set(assembled["retrieved_doc_indices"][:4]), {0, 1, 2, 3})

    def test_support_grounding_requires_all_grounded_role_anchors(self):
        openie_docs = [
            {
                "idx": "chunk-crucifixion",
                "passage": "Crucifixion\nCrucifixion was created by Titian.",
                "extracted_entities": ["Crucifixion", "Titian"],
                "extracted_triples": [["Crucifixion", "created by", "Titian"]],
            },
            {
                "idx": "chunk-plague",
                "passage": "Black Death\nPlague occurred in Amsterdam.",
                "extracted_entities": ["Plague", "Amsterdam"],
                "extracted_triples": [["Plague", "occurred in", "Amsterdam"]],
            },
            {
                "idx": "chunk-wrong-death",
                "passage": "Michel de Klerk\nMichel de Klerk died in Amsterdam.",
                "extracted_entities": ["Michel de Klerk", "Amsterdam"],
                "extracted_triples": [["Michel de Klerk", "died in", "Amsterdam"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="How many times did plague occur in the place where Crucifixion's creator died?",
            query_triples=[
                ["Crucifixion", "created by", "?x1"],
                ["?x1", "died in", "?place"],
                ["plague", "occurred in", "?place"],
            ],
            candidate_doc_indices=[0, 1, 2],
            fallback_doc_indices=[0, 1, 2],
            openie_docs=openie_docs,
            allow_support_obligation_grounding=True,
            evidence_set_size=3,
        )

        self.assertEqual(result["seed_source"], "query_obligation_incomplete_grounding_abstain_to_prior")
        self.assertEqual(result["query_obligation_exact_grounded_count"], 2)
        self.assertEqual(result["query_obligation_support_grounded_count"], 0)
        self.assertEqual(result["query_obligation_grounded_count"], 2)

    def test_support_grounding_treats_indexed_temporal_variables_as_generic_cues(self):
        openie_docs = [
            {
                "idx": "chunk-film",
                "passage": "Film B\nFilm B was directed by R. G. Springsteen.",
                "extracted_entities": ["Film B", "R. G. Springsteen"],
                "extracted_triples": [["Film B", "directed by", "R. G. Springsteen"]],
            },
            {
                "idx": "chunk-director",
                "passage": "R. G. Springsteen\nRobert G. Springsteen died on December 9, 1989.",
                "extracted_entities": ["Robert G. Springsteen", "December 9, 1989"],
                "extracted_triples": [["Robert G. Springsteen", "died on", "December 9, 1989"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="Which film has the director who died later, Film A or Film B?",
            query_triples=[
                ["Film B", "directed by", "?x2"],
                ["?x2", "died on", "?date2"],
            ],
            candidate_doc_indices=[0, 1],
            fallback_doc_indices=[0, 1],
            openie_docs=openie_docs,
            allow_support_obligation_grounding=True,
            evidence_set_size=2,
        )

        self.assertEqual(result["query_obligation_exact_grounded_count"], 1)
        self.assertEqual(result["query_obligation_support_grounded_count"], 1)
        support_matches = [
            match
            for matches in result["support_grounded_obligation_matches"].values()
            for match in matches
        ]
        self.assertEqual(len(support_matches), 1)
        self.assertEqual(support_matches[0]["doc_index"], 1)
        self.assertIn("relation_exact", support_matches[0]["support_grounding_reasons"])
        self.assertIn("subject_text_anchor_hit", support_matches[0]["support_grounding_reasons"])
        self.assertEqual(support_matches[0]["obligation_cue_token_hits"], [])

    def test_obligation_bound_seed_prevents_role_reversed_branch(self):
        openie_docs = [
            {
                "idx": "chunk-mother",
                "passage": "Lothair II\nLothair II's mother was Ermengarde of Tours.",
                "extracted_entities": ["Lothair II", "Ermengarde of Tours"],
                "extracted_triples": [["Lothair II", "mother", "Ermengarde of Tours"]],
            },
            {
                "idx": "chunk-death",
                "passage": "Ermengarde of Tours\nErmengarde of Tours died in 851.",
                "extracted_entities": ["Ermengarde of Tours", "851"],
                "extracted_triples": [["Ermengarde of Tours", "died in", "851"]],
            },
            {
                "idx": "chunk-daughter",
                "passage": "Bertha daughter of Lothair II\nBertha was a daughter of Lothair II.",
                "extracted_entities": ["Bertha", "Lothair II"],
                "extracted_triples": [["Bertha", "daughter of", "Lothair II"]],
            },
            {
                "idx": "chunk-noise-death",
                "passage": "Bertha\nBertha died in 925.",
                "extracted_entities": ["Bertha", "925"],
                "extracted_triples": [["Bertha", "died in", "925"]],
            },
        ]

        result = select_obligation_closed_local_ppr_evidence(
            query="When did Lothair II's mother die?",
            query_triples=[
                ["Lothair II", "mother", "?x"],
                ["?x", "died in", "?date"],
            ],
            candidate_doc_indices=[2, 3, 0, 1],
            openie_docs=openie_docs,
            evidence_set_size=5,
            max_path_edges=2,
            max_endpoint_degree=8,
            alpha=0.2,
            residual_epsilon=1e-12,
        )

        self.assertEqual(result["seed_source"], "query_obligation_bound_facts")
        self.assertEqual(result["query_obligation_count"], 2)
        self.assertEqual(result["query_obligation_match_count"], 2)
        self.assertEqual(result["retrieved_doc_indices"][:2], [0, 1])
        self.assertEqual(result["selected_closed_units"][0]["path_endpoint_keys"], ["ermengarde of tours"])

    def test_query_triples_from_row_prefers_row_then_head_trace(self):
        self.assertEqual(
            query_triples_from_row({"query_triples": [["Alpha", "born in", "?x"]]}, head_traces=[]),
            [["Alpha", "born in", "?x"]],
        )
        self.assertEqual(
            query_triples_from_row(
                {"query_index": 1},
                head_traces=[
                    {"top_k_facts": [["Wrong", "rel", "Node"]]},
                    {"top_k_facts": [["Beta", "located in", "Paris"]]},
                ],
            ),
            [["Beta", "located in", "Paris"]],
        )


if __name__ == "__main__":
    unittest.main()
