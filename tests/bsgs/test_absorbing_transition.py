from src.bsgs.transition import semi_absorbing_transition


def test_absorbing_transition_keeps_supported_node_mass():
    base = {("p1", "p2"): 1.0, ("p2", "p1"): 1.0}
    trans = semi_absorbing_transition(base, {"p1": 0.8, "p2": 0.0}, nodes=["p1", "p2"])
    assert abs(trans[("p1", "p1")] - 0.8) < 1e-9
    assert abs(trans[("p1", "p2")] - 0.2) < 1e-9
    assert abs(sum(v for (src, _), v in trans.items() if src == "p1") - 1.0) < 1e-9
