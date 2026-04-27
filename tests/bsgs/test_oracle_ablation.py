from importlib import util
from pathlib import Path

from src.bsgs.slots import Slot


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bsgs_run_oracle_ablation.py"
_SPEC = util.spec_from_file_location("bsgs_run_oracle_ablation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_diagnostic_propositions = _MODULE.build_diagnostic_propositions
likelihood_for_variant = _MODULE.likelihood_for_variant
rewrite_slot_with_gold_bindings = _MODULE.rewrite_slot_with_gold_bindings


def test_rewrite_slot_with_gold_bindings_replaces_musique_refs():
    slot = Slot(
        slot_id="s2",
        slot_text="When was #1 signed by Barcelona?",
        input_variables=["x1"],
        output_variable="answer",
        gold_answer="1982",
    )

    rewritten = rewrite_slot_with_gold_bindings(slot, ["Diego Maradona"])

    assert rewritten.slot_text == "When was Diego Maradona signed by Barcelona?"
    assert rewritten.output_variable == "answer"


def test_oracle_pool_adds_full_support_paragraph_proposition():
    sample = {
        "paragraphs": [
            {
                "idx": 0,
                "title": "Gold Doc",
                "paragraph_text": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth answer sentence.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Distractor",
                "paragraph_text": "Distractor sentence.",
                "is_supporting": False,
            },
        ]
    }

    props = build_diagnostic_propositions(sample, oracle_pool=True, base_max_props=1)

    full_props = [prop for prop in props.values() if prop.metadata.get("oracle_full_paragraph")]
    assert len(full_props) == 1
    assert "Fifth answer sentence" in full_props[0].text


def test_oracle_likelihood_uses_slot_support_idx():
    sample = {
        "paragraphs": [
            {
                "idx": 3,
                "title": "Gold Doc",
                "paragraph_text": "Gold support.",
                "is_supporting": True,
            },
            {
                "idx": 4,
                "title": "Distractor",
                "paragraph_text": "Distractor.",
                "is_supporting": False,
            },
        ]
    }
    props = build_diagnostic_propositions(sample, oracle_pool=True)
    gold_prop = next(prop for prop in props.values() if prop.metadata.get("paragraph_idx") == 3)
    distractor_prop = next(prop for prop in props.values() if prop.metadata.get("paragraph_idx") == 4)
    slot = Slot(slot_id="s1", slot_text="Gold support?", input_variables=[], output_variable="x1")

    assert likelihood_for_variant(slot, gold_prop, "oracle_pool_binding_likelihood", 3) == 0.95
    assert likelihood_for_variant(slot, distractor_prop, "oracle_pool_binding_likelihood", 3) == 0.05
