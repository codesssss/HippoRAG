from src.bsgs.state import BeliefStateGraph, PropositionNode


def test_bsgs_tracks_node_marginal_not_set_posterior():
    props = {
        "p1": PropositionNode("p1", "parent evidence", "d1", "parent evidence"),
        "p2": PropositionNode("p2", "child evidence", "d2", "child evidence"),
    }
    graph = BeliefStateGraph.uniform(props, edges={("p1", "p2"): 1.0, ("p2", "p2"): 1.0})
    predicted = graph.predict({("p1", "p2"): 1.0, ("p2", "p2"): 1.0})
    assert set(predicted) == {"p1", "p2"}
    assert abs(sum(predicted.values()) - 1.0) < 1e-9
    assert predicted["p2"] == 1.0
