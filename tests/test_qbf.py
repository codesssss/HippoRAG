import numpy as np

from src.hipporag.qbf import (
    QBFFactEntityIndex,
    infer_terminal_schema,
    rerank_doc_ids_by_scores,
)


class DummyEmbeddingModel:
    def batch_encode(self, texts, instruction=None, norm=True):
        rows = []
        for text in texts:
            text = str(text).lower()
            if "country" in text or "nationality" in text:
                rows.append(np.array([1.0, 0.0], dtype=float))
            elif "director" in text:
                rows.append(np.array([0.0, 1.0], dtype=float))
            else:
                rows.append(np.array([0.5, 0.5], dtype=float))
        return rows


def test_infer_terminal_schema_uses_relation_type_not_entity():
    schema = infer_terminal_schema("What country is the director of Inception from?")

    assert schema.label == "country"
    assert "country" in schema.phrase
    assert "Christopher Nolan" not in schema.phrase


def test_ppr_chi_rerank_keeps_baseline_rank_tie_break():
    doc_ids, scores = rerank_doc_ids_by_scores([3, 1, 2], np.array([0.0, 0.5, 0.2, 0.5]), top_k=3)

    assert doc_ids.tolist() == [3, 1, 2]
    assert scores.tolist() == [0.5, 0.5, 0.2]


def test_qbf_backward_flow_propagates_terminal_schema_over_entity_bridge():
    index = QBFFactEntityIndex(
        fact_ids=["f0", "f1"],
        fact_triples=[
            ("film a", "directed by", "person x"),
            ("person x", "country of citizenship", "country y"),
        ],
        fact_embeddings=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
        source_fact_indices=[0, 1],
        fact_doc_indices=[[0], [1]],
        num_docs=2,
    )
    fact_chi = index.score_terminal_schema(DummyEmbeddingModel(), "country nationality citizenship")
    beta, trace = index.backward_potential(fact_chi=fact_chi, alpha=0.5, max_iter=20)

    assert fact_chi.tolist() == [0.0, 1.0]
    assert beta[0] > fact_chi[0]
    assert beta[1] >= beta[0]
    assert trace["qbf_iterations"] >= 1


def test_qbf_retrieve_scores_terminal_document_first_when_query_matches_terminal_fact():
    index = QBFFactEntityIndex(
        fact_ids=["f0", "f1"],
        fact_triples=[
            ("film a", "directed by", "person x"),
            ("person x", "country of citizenship", "country y"),
        ],
        fact_embeddings=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
        source_fact_indices=[0, 1],
        fact_doc_indices=[[0], [1]],
        num_docs=2,
    )
    fact_chi = np.array([0.0, 1.0], dtype=float)
    query_fact_scores = np.array([0.2, 0.9], dtype=float)
    result = index.qbf_retrieve(query_fact_scores=query_fact_scores, fact_chi=fact_chi, top_k=2)

    assert result.sorted_doc_ids[0] == 1
    assert result.trace["operator"] == "qbf_schema_relational"


def test_qbf_doc_scores_exposes_full_doc_score_vector_for_pool_rerank():
    index = QBFFactEntityIndex(
        fact_ids=["f0", "f1"],
        fact_triples=[
            ("film a", "directed by", "person x"),
            ("person x", "country of citizenship", "country y"),
        ],
        fact_embeddings=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
        source_fact_indices=[0, 1],
        fact_doc_indices=[[0], [1]],
        num_docs=3,
    )
    doc_scores, trace = index.qbf_doc_scores(
        query_fact_scores=np.array([0.2, 0.9], dtype=float),
        fact_chi=np.array([0.0, 1.0], dtype=float),
    )

    assert doc_scores.shape == (3,)
    assert doc_scores[1] > doc_scores[0]
    assert doc_scores[2] == 0.0
    assert "qbf_iterations" in trace
