from src.bsgs.slots import Slot, evaluate_slot_quality


def test_slot_quality_greedy_alignment():
    gold = [
        Slot("s1", "director of Film X", [], "x1"),
        Slot("s2", "birthplace of x1", ["x1"], "answer"),
    ]
    pred = [
        Slot("p1", "who directed Film X", [], "x1"),
        Slot("p2", "where was x1 born", ["x1"], "answer"),
    ]
    scores = evaluate_slot_quality(pred, gold, threshold=0.2)
    assert scores["slot_recall"] == 1.0
    assert scores["slot_precision"] == 1.0
    assert scores["variable_grounding_accuracy"] == 1.0
