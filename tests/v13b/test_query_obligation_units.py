import unittest

from build_query_obligation_units import (
    build_query_obligation_units,
    compatible_binding_intersection,
    is_open_answer_assertion,
    match_query_obligations_to_sto_facts,
    obligation_matches_fact,
    obligation_seed_unit_ids,
    relation_signature,
    temporal_event_relation_signature,
)


class QueryObligationUnitTests(unittest.TestCase):
    def test_relation_signature_softens_surface_forms(self):
        self.assertEqual(relation_signature("was signed by"), "sign")
        self.assertEqual(relation_signature("died in"), "died_in")
        self.assertEqual(relation_signature("director"), "direct")
        self.assertEqual(relation_signature("designer"), "design")
        self.assertEqual(relation_signature("owner"), "own")
        self.assertEqual(relation_signature("winner"), "won")

    def test_temporal_event_relation_canonicalizes_date_objects_only(self):
        self.assertEqual(temporal_event_relation_signature("was born in", "?date1"), "born_on")
        self.assertEqual(temporal_event_relation_signature("was born in", "1950"), "born_on")
        self.assertEqual(temporal_event_relation_signature("died in", "August 17, 1987"), "died_on")
        self.assertEqual(temporal_event_relation_signature("was born in", "?place"), "born_in")
        self.assertEqual(temporal_event_relation_signature("was born in", "London"), "born_in")
        self.assertEqual(temporal_event_relation_signature("release date", "?date1"), "released_on")
        self.assertEqual(temporal_event_relation_signature("was released in", "1946"), "released_on")
        self.assertEqual(temporal_event_relation_signature("released by", "Universal Pictures"), "releas")
        self.assertEqual(temporal_event_relation_signature("birthplace", "?place"), "born_in")

    def test_temporal_event_fact_relation_matches_query_date_obligation(self):
        obligations = build_query_obligation_units(
            query="Who was born first, Alice or Bob?",
            query_triples=[["Alice", "born on", "?date1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-alice-born",
            "doc_index": 1,
            "subject": "Alice",
            "relation": "was born in",
            "object": "1950",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_release_date_obligation_matches_released_temporal_fact(self):
        obligations = build_query_obligation_units(
            query="Which film was released first?",
            query_triples=[["Phoolwari", "release date", "?date1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-release",
            "doc_index": 1,
            "subject": "Phoolwari",
            "relation": "was released in",
            "object": "1946",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_director_obligation_matches_directed_by_fact_same_roles(self):
        obligations = build_query_obligation_units(
            query="Who directed El Tonto?",
            query_triples=[["El Tonto", "director", "?x1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-director",
            "doc_index": 1,
            "subject": "El Tonto",
            "relation": "directed by",
            "object": "Charlie Day",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_written_by_obligation_matches_bare_by_fact_same_roles(self):
        obligations = build_query_obligation_units(
            query="Who wrote The Adventure of the Seven Clocks?",
            query_triples=[["The Adventure of the Seven Clocks", "written by", "?x1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-author",
            "doc_index": 1,
            "subject": "The Adventure of the Seven Clocks",
            "relation": "by",
            "object": "Adrian Conan Doyle",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_wrote_obligation_matches_written_by_fact_inverse_roles(self):
        obligations = build_query_obligation_units(
            query="Did Peter Laufer write No Animals Were Harmed?",
            query_triples=[["Peter Laufer", "wrote", "No Animals Were Harmed"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-written",
            "doc_index": 1,
            "subject": "No Animals Were Harmed",
            "relation": "is written by",
            "object": "Peter Laufer",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_inverse_written_binding_maps_query_variable_to_fact_object_role(self):
        obligations = build_query_obligation_units(
            query="Who wrote No Animals Were Harmed?",
            query_triples=[["?x1", "wrote", "No Animals Were Harmed"]],
        )
        candidate_units = [
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-written",
                "doc_index": 1,
                "subject": "No Animals Were Harmed",
                "relation": "is written by",
                "object": "Peter Laufer",
            }
        ]

        matches = match_query_obligations_to_sto_facts(
            obligations=obligations,
            candidate_units=candidate_units,
        )

        obligation_id = obligations[0]["obligation_id"]
        self.assertEqual(matches[obligation_id][0]["role_alignment"], "inverse")
        self.assertEqual(matches[obligation_id][0]["variable_bindings"], {"x1": ["peter laufer"]})

    def test_birthplace_obligation_matches_born_in_place_fact(self):
        obligations = build_query_obligation_units(
            query="Where was Erik Hort born?",
            query_triples=[["Erik Hort", "birthplace", "?place"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-birthplace",
            "doc_index": 1,
            "subject": "Erik Hort",
            "relation": "born in",
            "object": "Montebello, New York",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_temporal_event_relation_does_not_match_birthplace_fact(self):
        obligations = build_query_obligation_units(
            query="Who was born first, Alice or Bob?",
            query_triples=[["Alice", "born on", "?date1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-alice-birthplace",
            "doc_index": 1,
            "subject": "Alice",
            "relation": "was born in",
            "object": "London",
        }

        self.assertFalse(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_compatible_binding_intersection_canonicalizes_embedded_endpoint(self):
        self.assertEqual(
            compatible_binding_intersection({"108 miles southeast of phoenix"}, {"phoenix"}),
            {"phoenix"},
        )

    def test_open_answer_assertion_is_not_a_retrieval_obligation(self):
        self.assertTrue(
            is_open_answer_assertion(
                subject_is_variable=True,
                object_is_variable=True,
                subject_variable="answer",
                object_variable="race",
            )
        )
        obligations = build_query_obligation_units(
            query="Who won the race in Phoenix?",
            query_triples=[
                ["?race", "held in", "Phoenix"],
                ["?answer", "won", "?race"],
            ],
        )

        self.assertEqual(len(obligations), 1)
        self.assertEqual(obligations[0]["raw_relation"], "held in")

    def test_answer_variable_with_concrete_endpoint_can_still_ground(self):
        obligations = build_query_obligation_units(
            query="Who directed Inception?",
            query_triples=[["?answer", "directed", "Inception"]],
        )

        self.assertEqual(len(obligations), 1)
        self.assertEqual(obligations[0]["subject_variable"], "answer")

    def test_lothair_mother_obligation_does_not_bind_daughter_fact(self):
        obligations = build_query_obligation_units(
            query="When did Lothair II's mother die?",
            query_triples=[["Lothair II", "mother", "?x"]],
        )
        self.assertEqual(len(obligations), 1)
        obligation = obligations[0]

        mother_fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-mother",
            "doc_index": 4,
            "subject": "Lothair II",
            "relation": "mother",
            "object": "Ermengarde of Tours",
        }
        daughter_fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-daughter",
            "doc_index": 6,
            "subject": "Bertha",
            "relation": "daughter of",
            "object": "Lothair II",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligation, fact_unit=mother_fact))
        self.assertFalse(obligation_matches_fact(obligation=obligation, fact_unit=daughter_fact))

    def test_object_bound_obligation_preserves_object_role(self):
        obligations = build_query_obligation_units(
            query="When was the person signed by Barcelona?",
            query_triples=[["?x", "signed by", "Barcelona"]],
        )
        obligation = obligations[0]

        signed_fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-signed",
            "doc_index": 2,
            "subject": "Player",
            "relation": "signed by",
            "object": "Barcelona",
        }
        reverse_fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-reverse",
            "doc_index": 7,
            "subject": "Barcelona",
            "relation": "signed by",
            "object": "Player",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligation, fact_unit=signed_fact))
        self.assertFalse(obligation_matches_fact(obligation=obligation, fact_unit=reverse_fact))

    def test_bound_subject_matches_title_qualified_openie_subject(self):
        obligations = build_query_obligation_units(
            query="What nationality is the director of film Blood Street?",
            query_triples=[["film Blood Street", "director", "?x1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-director",
            "doc_index": 1,
            "subject": "Blood Street",
            "relation": "directed by",
            "object": "Leo Fong",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_bound_subject_matches_parenthetical_title_alias(self):
        obligations = build_query_obligation_units(
            query="Who was Aleksander Koniecpolski's father?",
            query_triples=[["Aleksander Koniecpolski (1620-1659)", "father", "?x1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-father",
            "doc_index": 1,
            "subject": "Aleksander Koniecpolski",
            "relation": "father",
            "object": "Stanislaw Koniecpolski",
        }

        self.assertTrue(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_bound_subject_does_not_match_arbitrary_suffix_substring(self):
        obligations = build_query_obligation_units(
            query="Who founded New York University?",
            query_triples=[["New York University", "founded by", "?x1"]],
        )
        fact = {
            "unit_type": "openie_fact",
            "unit_id": "fact-other-school",
            "doc_index": 1,
            "subject": "York University",
            "relation": "founded by",
            "object": "Other Founder",
        }

        self.assertFalse(obligation_matches_fact(obligation=obligations[0], fact_unit=fact))

    def test_match_query_obligations_returns_seed_fact_ids(self):
        obligations = build_query_obligation_units(
            query="When did Lothair II's mother die?",
            query_triples=[
                ["Lothair II", "mother", "?x"],
                ["?x", "died in", "?date"],
            ],
        )
        candidate_units = [
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-mother",
                "doc_index": 4,
                "subject": "Lothair II",
                "relation": "mother",
                "object": "Ermengarde of Tours",
            },
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-died",
                "doc_index": 5,
                "subject": "Ermengarde of Tours",
                "relation": "died in",
                "object": "851",
            },
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-daughter",
                "doc_index": 6,
                "subject": "Bertha",
                "relation": "daughter of",
                "object": "Lothair II",
            },
        ]

        matches = match_query_obligations_to_sto_facts(
            obligations=obligations,
            candidate_units=candidate_units,
        )

        self.assertEqual(obligation_seed_unit_ids(matches), ["fact-mother", "fact-died"])

    def test_all_variable_obligation_does_not_seed_without_anchor_binding(self):
        obligations = build_query_obligation_units(
            query="Who won the race held in the city?",
            query_triples=[
                ["?race", "held in", "?city"],
                ["?answer", "won", "?race"],
            ],
        )
        candidate_units = [
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-held",
                "doc_index": 1,
                "subject": "Noise Race",
                "relation": "held in",
                "object": "Noise City",
            },
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-won",
                "doc_index": 2,
                "subject": "Noise Driver",
                "relation": "won",
                "object": "Noise Race",
            },
        ]

        matches = match_query_obligations_to_sto_facts(
            obligations=obligations,
            candidate_units=candidate_units,
        )

        self.assertEqual(obligation_seed_unit_ids(matches), [])

    def test_open_answer_assertion_is_filtered_after_anchor_binding(self):
        obligations = build_query_obligation_units(
            query="Who won the race held in Phoenix?",
            query_triples=[
                ["?race", "held in", "Phoenix"],
                ["?answer", "won", "?race"],
            ],
        )
        candidate_units = [
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-held",
                "doc_index": 1,
                "subject": "Phoenix Grand Prix",
                "relation": "held in",
                "object": "Phoenix",
            },
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-won",
                "doc_index": 2,
                "subject": "Correct Driver",
                "relation": "won",
                "object": "Phoenix Grand Prix",
            },
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-noise-won",
                "doc_index": 3,
                "subject": "Noise Driver",
                "relation": "won",
                "object": "Noise Race",
            },
        ]

        matches = match_query_obligations_to_sto_facts(
            obligations=obligations,
            candidate_units=candidate_units,
        )

        self.assertEqual(obligation_seed_unit_ids(matches), ["fact-held"])

    def test_variable_binding_uses_embedded_endpoint_for_composite_openie_object(self):
        obligations = build_query_obligation_units(
            query="Which city near Tucson is known for Indy car racing?",
            query_triples=[
                ["Tucson", "is located", "?city"],
                ["?city", "known for", "Indy car track"],
            ],
        )
        candidate_units = [
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-located",
                "doc_index": 1,
                "subject": "Tucson",
                "relation": "is located",
                "object": "108 miles southeast of Phoenix",
            },
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-known",
                "doc_index": 2,
                "subject": "Phoenix",
                "relation": "is known for",
                "object": "Indy car track",
            },
        ]

        matches = match_query_obligations_to_sto_facts(
            obligations=obligations,
            candidate_units=candidate_units,
        )

        self.assertEqual(obligation_seed_unit_ids(matches), ["fact-located", "fact-known"])

    def test_seed_match_alternatives_union_before_cross_obligation_intersection(self):
        obligations = build_query_obligation_units(
            query="Which person was born later?",
            query_triples=[["?x2", "born on", "?date2"]],
        )
        candidate_units = [
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-bob-born",
                "doc_index": 1,
                "subject": "Bob",
                "relation": "born on",
                "object": "2 February 2001",
            },
            {
                "unit_type": "openie_fact",
                "unit_id": "fact-alice-born",
                "doc_index": 2,
                "subject": "Alice",
                "relation": "born on",
                "object": "1 January 1990",
            },
        ]
        seed_matches = {
            "seed-obligation": [
                {"variable_bindings": {"x2": ["Bob"]}},
                {"variable_bindings": {"x2": ["Alias Without Birth Fact"]}},
            ]
        }

        matches = match_query_obligations_to_sto_facts(
            obligations=obligations,
            candidate_units=candidate_units,
            seed_matches=seed_matches,
        )

        self.assertEqual(obligation_seed_unit_ids(matches), ["fact-bob-born"])


if __name__ == "__main__":
    unittest.main()
