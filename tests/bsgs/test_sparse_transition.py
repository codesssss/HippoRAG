from src.bsgs.transition import sparse_transition


def test_sparse_transition_row_normalizes_by_likelihood():
    edges = {("p1", "p2"): 2.0, ("p1", "p3"): 1.0, ("p2", "p1"): 1.0}
    likelihood = {"p2": 0.5, "p3": 1.0, "p1": 1.0}
    trans = sparse_transition(edges, likelihood, nodes=["p1", "p2", "p3"])
    assert abs(trans[("p1", "p2")] - 0.5) < 1e-9
    assert abs(trans[("p1", "p3")] - 0.5) < 1e-9
    assert trans[("p3", "p3")] == 1.0
