import unittest

from query_obligation_typing import (
    classify_query_triple,
    lower_query_triples_for_typed_program,
    typed_query_program,
)


class QueryObligationTypingTests(unittest.TestCase):
    def test_answer_target_is_not_retrieval_critical(self):
        typed = classify_query_triple(
            query="Who won the race held in Phoenix?",
            triple=["?answer", "won", "?race"],
        )

        self.assertEqual(typed["obligation_type"], "answer_target")
        self.assertFalse(typed["retrieval_critical"])
        self.assertEqual(typed["reason"], "answer_variable")
        self.assertTrue(typed["filtered_by_obligation_builder"])

    def test_temporal_answer_extraction_is_aggregation(self):
        typed = classify_query_triple(
            query="What year did the publisher of Labyrinth end?",
            triple=["?publisher", "ended in", "?year"],
        )

        self.assertEqual(typed["obligation_type"], "aggregation")
        self.assertFalse(typed["retrieval_critical"])
        self.assertEqual(typed["query_answer_shape"], "date")

    def test_when_answer_date_slot_is_aggregation_even_with_nonstandard_relation(self):
        typed = classify_query_triple(
            query="The Distribution of Industry act was passed by a man who was prime minister when?",
            triple=["?person", "was prime minister when", "?date"],
        )

        self.assertEqual(typed["obligation_type"], "aggregation")
        self.assertFalse(typed["retrieval_critical"])

    def test_comparison_is_not_plain_retrieval_evidence(self):
        typed = classify_query_triple(
            query="Which group is larger than the record label?",
            triple=["?group", "larger than", "?label"],
        )

        self.assertEqual(typed["obligation_type"], "comparison")
        self.assertFalse(typed["retrieval_critical"])

    def test_temporal_order_than_relation_is_comparison(self):
        typed = classify_query_triple(
            query="Which film was released first, Aas Ka Panchhi or Phoolwari?",
            triple=["?date1", "is earlier than", "?date2"],
        )

        self.assertEqual(typed["obligation_type"], "comparison")
        self.assertFalse(typed["retrieval_critical"])

    def test_longer_than_relation_is_comparison(self):
        typed = classify_query_triple(
            query="Who lived longer, A or B?",
            triple=["?person1", "lived longer than", "?person2"],
        )

        self.assertEqual(typed["obligation_type"], "comparison")
        self.assertFalse(typed["retrieval_critical"])

    def test_succeeded_by_is_factual_bridge_not_comparison_operator(self):
        typed = classify_query_triple(
            query="What company succeeded the owner of Empire Sports Network?",
            triple=["?owner", "succeeded by", "?company"],
        )

        self.assertEqual(typed["obligation_type"], "bridge")
        self.assertTrue(typed["retrieval_critical"])

    def test_constraint_relation_is_not_relation_scan(self):
        typed = classify_query_triple(
            query="Where is the group related to Randy Conrads?",
            triple=["?university", "related to", "Randy Conrads"],
        )

        self.assertEqual(typed["obligation_type"], "constraint")
        self.assertFalse(typed["retrieval_critical"])

    def test_variable_variable_type_descriptor_is_constraint_not_bridge_scan(self):
        typed = classify_query_triple(
            query="The fictional private detective appears in which story?",
            triple=["?x1", "is a fictional private detective", "?x2"],
        )

        self.assertEqual(typed["obligation_type"], "constraint")
        self.assertFalse(typed["retrieval_critical"])
        self.assertEqual(typed["reason"], "type_descriptor_constraint")

    def test_variable_variable_age_is_derived_attribute_not_bridge_scan(self):
        typed = classify_query_triple(
            query="Which film has the director who is older, Film A or Film B?",
            triple=["?x1", "age", "?age1"],
        )

        self.assertEqual(typed["obligation_type"], "comparison")
        self.assertFalse(typed["retrieval_critical"])
        self.assertEqual(typed["reason"], "derived_attribute_operand")

    def test_type_relation_with_variable_subject_is_constraint_not_bridge_scan(self):
        typed = classify_query_triple(
            query="Are both books about animals?",
            triple=["?x1", "type", "animal"],
        )

        self.assertEqual(typed["obligation_type"], "constraint")
        self.assertFalse(typed["retrieval_critical"])
        self.assertEqual(typed["reason"], "descriptor_relation_constraint")

    def test_variable_variable_generic_attribute_is_constraint_not_bridge_scan(self):
        typed = classify_query_triple(
            query="Which player from the same country had a left-handed batting style?",
            triple=["?x1", "batting style", "?x2"],
        )

        self.assertEqual(typed["obligation_type"], "constraint")
        self.assertFalse(typed["retrieval_critical"])
        self.assertEqual(typed["reason"], "descriptor_relation_constraint")

    def test_concrete_to_variable_binding_is_bridge(self):
        typed = classify_query_triple(
            query="When did Lothair II's mother die?",
            triple=["Lothair II", "mother", "?x1"],
        )

        self.assertEqual(typed["obligation_type"], "bridge")
        self.assertTrue(typed["retrieval_critical"])
        self.assertEqual(typed["reason"], "concrete_to_variable_binding")

    def test_fully_concrete_fact_is_retrieval_evidence(self):
        typed = classify_query_triple(
            query="What happened in Copa del Rey?",
            triple=["Messi", "goals in", "Copa del Rey"],
        )

        self.assertEqual(typed["obligation_type"], "retrieval_evidence")
        self.assertTrue(typed["retrieval_critical"])

    def test_program_counts_types(self):
        program = typed_query_program(
            query="Who won the race in Phoenix?",
            query_triples=[
                ["?race", "held in", "Phoenix"],
                ["?answer", "won", "?race"],
                ["?race", "larger than", "?other"],
            ],
        )

        self.assertEqual(program["type_counts"]["bridge"], 1)
        self.assertEqual(program["type_counts"]["answer_target"], 1)
        self.assertEqual(program["type_counts"]["comparison"], 1)
        self.assertEqual(program["retrieval_critical_count"], 1)

    def test_temporal_comparison_exports_variable_type_constraints(self):
        program = typed_query_program(
            query="Which film has the director born first, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["?x1", "was born in", "?date1"],
                ["Film B", "directed by", "?x2"],
                ["?x2", "was born in", "?date2"],
                ["?date1", "earlier than", "?date2"],
            ],
        )

        self.assertEqual(
            program["variable_type_constraints"],
            {"date1": "temporal", "date2": "temporal"},
        )
        self.assertEqual(
            set(program["variable_type_constraint_reasons"].values()),
            {"temporal_comparison_operand"},
        )

    def test_older_than_comparison_lowers_to_birth_date_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film has the director who is older than the other, Airheads or Return To Cabin By The Lake?",
            query_triples=[
                ["Airheads", "directed by", "?x1"],
                ["Return To Cabin By The Lake", "directed by", "?x2"],
                ["?x1", "is older than", "?x2"],
            ],
        )

        self.assertIn(["?x1", "born on", "?date1"], lowered)
        self.assertIn(["?x2", "born on", "?date2"], lowered)
        self.assertIn(["?date1", "earlier than", "?date2"], lowered)

    def test_older_than_lowering_does_not_duplicate_existing_birth_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which director is older?",
            query_triples=[
                ["?x1", "born on", "?date1"],
                ["?x2", "born on", "?date2"],
                ["?x1", "is older than", "?x2"],
            ],
        )

        self.assertEqual(lowered.count(["?x1", "born on", "?date1"]), 1)
        self.assertEqual(lowered.count(["?x2", "born on", "?date2"]), 1)

    def test_older_than_lowering_requires_person_age_context(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film is older, Film A or Film B?",
            query_triples=[["?film1", "is older than", "?film2"]],
        )

        self.assertNotIn(["?film1", "born on", "?date1"], lowered)
        self.assertNotIn(["?film2", "born on", "?date2"], lowered)

    def test_born_later_than_comparison_lowers_to_birth_date_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film has the director who was born later, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x2"],
                ["?x1", "was born later than", "?x2"],
            ],
        )

        self.assertIn(["?x1", "born on", "?date1"], lowered)
        self.assertIn(["?x2", "born on", "?date2"], lowered)
        self.assertIn(["?date1", "earlier than", "?date2"], lowered)

    def test_temporal_birth_relation_normalizes_in_to_on_for_date_variables(self):
        lowered = lower_query_triples_for_typed_program(
            query="Who was born first, Alice or Bob?",
            query_triples=[
                ["Alice", "was born in", "?date1"],
                ["Bob", "was born in", "?date2"],
                ["?date1", "earlier than", "?date2"],
            ],
        )

        self.assertIn(["Alice", "born on", "?date1"], lowered)
        self.assertIn(["Bob", "born on", "?date2"], lowered)
        self.assertNotIn(["Alice", "was born in", "?date1"], lowered)
        self.assertNotIn(["Bob", "was born in", "?date2"], lowered)

    def test_birthplace_relation_does_not_normalize_place_variable(self):
        lowered = lower_query_triples_for_typed_program(
            query="Where was Alice born?",
            query_triples=[["Alice", "was born in", "?place"]],
        )

        self.assertIn(["Alice", "was born in", "?place"], lowered)
        self.assertNotIn(["Alice", "born on", "?place"], lowered)

    def test_temporal_death_relation_normalizes_in_to_on_for_date_variables(self):
        lowered = lower_query_triples_for_typed_program(
            query="Who died first, Alice or Bob?",
            query_triples=[
                ["Alice", "died in", "?date1"],
                ["Bob", "died in", "?date2"],
                ["?date1", "earlier than", "?date2"],
            ],
        )

        self.assertIn(["Alice", "died on", "?date1"], lowered)
        self.assertIn(["Bob", "died on", "?date2"], lowered)
        self.assertNotIn(["Alice", "died in", "?date1"], lowered)
        self.assertNotIn(["Bob", "died in", "?date2"], lowered)

    def test_born_later_duplicate_branch_variable_is_split_to_declared_operand(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film has the director who was born later, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x1"],
                ["?x1", "was born in", "?date"],
                ["?x1", "was born later than", "?x2"],
            ],
        )

        self.assertIn(["Film A", "directed by", "?x1"], lowered)
        self.assertIn(["Film B", "directed by", "?x2"], lowered)
        self.assertNotIn(["Film B", "directed by", "?x1"], lowered)
        self.assertIn(["?x1", "born on", "?date"], lowered)
        self.assertNotIn(["?x1", "was born in", "?date"], lowered)
        self.assertIn(["?x2", "born on", "?date2"], lowered)
        self.assertIn(["?date", "earlier than", "?date2"], lowered)
        self.assertNotIn(["?date1", "earlier than", "?date2"], lowered)

    def test_duplicate_branch_variable_is_not_split_when_second_operand_is_materialized(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film has the director who was born later, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x1"],
                ["?x2", "was born in", "?date2"],
                ["?x1", "was born later than", "?x2"],
            ],
        )

        self.assertIn(["Film B", "directed by", "?x1"], lowered)
        self.assertNotIn(["Film B", "directed by", "?x2"], lowered)

    def test_shared_object_age_comparison_lowers_subjects_as_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Who is older, Cheryl Saban or Dmitry Grigorieff?",
            query_triples=[
                ["Cheryl Saban", "is older than", "?x1"],
                ["Dmitry Grigorieff", "is older than", "?x1"],
            ],
        )

        self.assertIn(["Cheryl Saban", "born on", "?date1"], lowered)
        self.assertIn(["Dmitry Grigorieff", "born on", "?date2"], lowered)
        self.assertNotIn(["?x1", "born on", "?date2"], lowered)
        self.assertIn(["?date1", "earlier than", "?date2"], lowered)

    def test_release_comparison_lowers_to_release_date_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film was released first, Film A or Film B?",
            query_triples=[["Film A", "earlier than", "Film B"]],
        )

        self.assertIn(["Film A", "released in", "?date1"], lowered)
        self.assertIn(["Film B", "released in", "?date2"], lowered)
        self.assertIn(["?date1", "earlier than", "?date2"], lowered)

    def test_release_lowering_does_not_duplicate_existing_release_date_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film was released first, Film A or Film B?",
            query_triples=[
                ["Film A", "was released in", "?date1"],
                ["Film B", "was released in", "?date2"],
                ["?date1", "earlier than", "?date2"],
            ],
        )

        self.assertNotIn(["Film A", "released in", "?date1"], lowered)
        self.assertNotIn(["Film B", "released in", "?date2"], lowered)
        self.assertNotIn(["?date1", "released in", "?date1"], lowered)
        self.assertNotIn(["?date2", "released in", "?date2"], lowered)

    def test_established_comparison_lowers_to_established_date_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which museum was established first, Museum A or Museum B?",
            query_triples=[["Museum A", "earlier than", "Museum B"]],
        )

        self.assertIn(["Museum A", "established in", "?date1"], lowered)
        self.assertIn(["Museum B", "established in", "?date2"], lowered)
        self.assertIn(["?date1", "earlier than", "?date2"], lowered)

    def test_established_lowering_does_not_relower_date_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which museum was established first, Museum A or Museum B?",
            query_triples=[
                ["Museum A", "was established in", "?date1"],
                ["Museum B", "was established in", "?date2"],
                ["?date1", "earlier than", "?date2"],
            ],
        )

        self.assertNotIn(["?date1", "established in", "?date1"], lowered)
        self.assertNotIn(["?date2", "established in", "?date2"], lowered)

    def test_died_later_than_lowers_to_death_date_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Which film has the director who died later, Film A or Film B?",
            query_triples=[
                ["Film A", "directed by", "?x1"],
                ["Film B", "directed by", "?x2"],
                ["?x1", "died later than", "?x2"],
            ],
        )

        self.assertIn(["?x1", "died on", "?date1"], lowered)
        self.assertIn(["?x2", "died on", "?date2"], lowered)
        self.assertIn(["?date1", "earlier than", "?date2"], lowered)

    def test_shared_object_death_comparison_lowers_subjects_as_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Who died earlier, Alice or Bob?",
            query_triples=[
                ["Alice", "died earlier than", "?x1"],
                ["Bob", "died earlier than", "?x1"],
            ],
        )

        self.assertIn(["Alice", "died on", "?date1"], lowered)
        self.assertIn(["Bob", "died on", "?date2"], lowered)
        self.assertIn(["?date1", "earlier than", "?date2"], lowered)

    def test_lived_longer_comparison_lowers_to_lifespan_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Who lived longer, Alice or Bob?",
            query_triples=[["Alice", "lived longer than", "Bob"]],
        )

        self.assertIn(["Alice", "born on", "?date1"], lowered)
        self.assertIn(["Alice", "died on", "?date2"], lowered)
        self.assertIn(["Bob", "born on", "?date3"], lowered)
        self.assertIn(["Bob", "died on", "?date4"], lowered)

    def test_shared_object_lived_longer_comparison_lowers_subjects_as_operands(self):
        lowered = lower_query_triples_for_typed_program(
            query="Who lived longer, Ludwig Elsbett or Pamela Ann Rymer?",
            query_triples=[
                ["Ludwig Elsbett", "lived longer than", "?x1"],
                ["Pamela Ann Rymer", "lived longer than", "?x1"],
            ],
        )

        self.assertIn(["Ludwig Elsbett", "born on", "?date1"], lowered)
        self.assertIn(["Ludwig Elsbett", "died on", "?date2"], lowered)
        self.assertIn(["Pamela Ann Rymer", "born on", "?date3"], lowered)
        self.assertIn(["Pamela Ann Rymer", "died on", "?date4"], lowered)
        self.assertNotIn(["?x1", "born on", "?date3"], lowered)
        self.assertNotIn(["?x1", "died on", "?date4"], lowered)


if __name__ == "__main__":
    unittest.main()
