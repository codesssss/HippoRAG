from build_minimal_evidence_units import build_minimal_evidence_units_for_doc


def test_minimal_evidence_units_include_sentence_and_openie_provenance():
    doc = {
        "idx": "doc-1",
        "passage": "Lionel Messi\nLionel Messi signed for Barcelona in 2000.",
        "extracted_entities": ["Lionel Messi", "Barcelona"],
        "extracted_triples": [["Lionel Messi", "joined", "Barcelona"]],
    }

    units = build_minimal_evidence_units_for_doc(doc, doc_index=3)

    sentence_units = [unit for unit in units if unit["source"] == "sentence"]
    openie_units = [unit for unit in units if unit["source"] == "openie"]
    assert sentence_units
    assert openie_units
    assert sentence_units[0]["doc_index"] == 3
    assert sentence_units[0]["title"] == "Lionel Messi"
    assert "Lionel Messi" in sentence_units[0]["mention_surfaces"]
    assert "Barcelona" in sentence_units[0]["mention_surfaces"]
    assert "2000" in sentence_units[0]["mention_surfaces"]
    assert openie_units[0]["predicate_text"] == "joined"
    assert openie_units[0]["argument_texts"] == ["Lionel Messi", "Barcelona"]


def test_minimal_evidence_units_do_not_require_openie_relation_label():
    doc = {
        "idx": "doc-2",
        "passage": "Ada Lovelace\nAda Lovelace wrote notes about the Analytical Engine.",
        "extracted_entities": ["Ada Lovelace", "Analytical Engine"],
        "extracted_triples": [],
    }

    units = build_minimal_evidence_units_for_doc(doc, doc_index=0)

    assert len(units) == 1
    assert units[0]["source"] == "sentence"
    assert units[0]["predicate_text"] == ""
    assert "Ada Lovelace" in units[0]["mention_surfaces"]
    assert "Analytical Engine" in units[0]["mention_surfaces"]
