from src.bsgs.state import BeliefStateGraph, PropositionNode


def test_belief_normalization_uniform_fallback():
    props = {
        "p1": PropositionNode("p1", "A", "d1", "A"),
        "p2": PropositionNode("p2", "B", "d2", "B"),
    }
    graph = BeliefStateGraph(propositions=props, belief={"p1": 0.0, "p2": 0.0}, edges={})
    graph.normalize()
    assert graph.belief == {"p1": 0.5, "p2": 0.5}


def test_observe_renormalizes():
    props = {
        "p1": PropositionNode("p1", "A", "d1", "A"),
        "p2": PropositionNode("p2", "B", "d2", "B"),
    }
    graph = BeliefStateGraph.uniform(props)
    graph.observe({"p1": 0.9, "p2": 0.1})
    assert abs(sum(graph.belief.values()) - 1.0) < 1e-9
    assert graph.belief["p1"] > graph.belief["p2"]
