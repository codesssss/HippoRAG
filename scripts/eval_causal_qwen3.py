import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.utils.causal_utils import route_query_type
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import QuerySolution, string_to_bool


def get_gold_docs(samples: List, dataset_name: str = None, corpus: List | None = None) -> List:
    gold_docs = []
    corpus = corpus or []
    for sample in samples:
        if 'supporting_facts' in sample:
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        elif 'document' in sample and isinstance(sample['document'], dict) and corpus:
            document_id = str(sample['document'].get('id', '')).strip()
            gold_doc = [
                item['title'] + '\n' + item['text']
                for item in corpus
                if str(item.get('idx', '')).startswith(f"{document_id}_")
            ]
        else:
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample in samples:
        gold_ans = None
        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = list(set([sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']]))
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])
        gold_answers.append(list(gold_ans))
    return gold_answers


def subset_by_indices(values: List, indices: List[int]) -> List:
    return [values[idx] for idx in indices]


def compute_slice_metrics(config: BaseConfig,
                          query_solutions: List[QuerySolution],
                          gold_docs: List[List[str]],
                          gold_answers: List[List[str]]) -> Dict[str, Dict[str, float]]:
    queries = [query_solution.question for query_solution in query_solutions]
    predicted_answers = [query_solution.answer for query_solution in query_solutions]
    retrieved_docs = [query_solution.docs for query_solution in query_solutions]

    qa_em = QAExactMatch(global_config=config)
    qa_f1 = QAF1Score(global_config=config)
    retrieval = RetrievalRecall(global_config=config)

    retrieval_metrics, _ = retrieval.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=retrieved_docs,
        k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
    )
    em_metrics, _ = qa_em.calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )
    f1_metrics, _ = qa_f1.calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )

    causal_indices = [idx for idx, query in enumerate(queries) if route_query_type(query) != "non_causal"]
    if causal_indices:
        causal_retrieval_metrics, _ = retrieval.calculate_metric_scores(
            gold_docs=subset_by_indices(gold_docs, causal_indices),
            retrieved_docs=subset_by_indices(retrieved_docs, causal_indices),
            k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
        )
        causal_em_metrics, _ = qa_em.calculate_metric_scores(
            gold_answers=subset_by_indices(gold_answers, causal_indices),
            predicted_answers=subset_by_indices(predicted_answers, causal_indices),
            aggregation_fn=np.max,
        )
        causal_f1_metrics, _ = qa_f1.calculate_metric_scores(
            gold_answers=subset_by_indices(gold_answers, causal_indices),
            predicted_answers=subset_by_indices(predicted_answers, causal_indices),
            aggregation_fn=np.max,
        )
    else:
        causal_retrieval_metrics, causal_em_metrics, causal_f1_metrics = {}, {}, {}

    return {
        "overall": {
            **retrieval_metrics,
            **em_metrics,
            **f1_metrics,
            "num_queries": len(queries),
        },
        "causal_slice": {
            **causal_retrieval_metrics,
            **causal_em_metrics,
            **causal_f1_metrics,
            "num_queries": len(causal_indices),
        },
    }


def build_config(args, corpus_len: int) -> BaseConfig:
    return BaseConfig(
        save_dir=args.save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_base_url=args.embedding_base_url,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=args.embedding_batch_size,
        max_new_tokens=None,
        max_retry_attempts=args.max_retry_attempts,
        corpus_len=corpus_len,
        openie_mode=args.openie_mode,
        planner_enabled=string_to_bool(args.planner_enabled),
        planner_mode=args.planner_mode,
        planner_max_steps=args.planner_max_steps,
        causal_enabled=string_to_bool(args.causal_enabled),
        causal_query_only=string_to_bool(args.causal_query_only),
        causal_seed_top_k=args.causal_seed_top_k,
        causal_confidence_threshold=args.causal_confidence_threshold,
        causal_damping=args.causal_damping,
        causal_blend_dense_weight=args.causal_blend_dense_weight,
        causal_blend_fact_weight=args.causal_blend_fact_weight,
        causal_blend_graph_weight=args.causal_blend_graph_weight,
        structure_rerank_enabled=string_to_bool(args.structure_rerank_enabled),
        structure_rerank_top_n=args.structure_rerank_top_n,
        structure_rerank_bonus_weight=args.structure_rerank_bonus_weight,
        structure_rerank_min_edge_support=args.structure_rerank_min_edge_support,
        structure_rerank_max_top5_swaps=args.structure_rerank_max_top5_swaps,
        structure_rerank_seed_top_k=args.structure_rerank_seed_top_k,
        structure_rerank_max_hops=args.structure_rerank_max_hops,
        structure_rerank_margin_threshold=args.structure_rerank_margin_threshold,
        rerank_require_non_empty=string_to_bool(args.rerank_require_non_empty),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate HippoRAG with local Qwen3 causal retrieval.")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--planner_enabled", type=str, default="false")
    parser.add_argument("--planner_mode", choices=["none", "myopic"], default="none")
    parser.add_argument("--planner_max_steps", type=int, default=3)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--causal_enabled", type=str, default="true")
    parser.add_argument("--causal_query_only", type=str, default="true")
    parser.add_argument("--causal_seed_top_k", type=int, default=20)
    parser.add_argument("--causal_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--causal_damping", type=float, default=0.7)
    parser.add_argument("--causal_blend_dense_weight", type=float, default=0.35)
    parser.add_argument("--causal_blend_fact_weight", type=float, default=0.15)
    parser.add_argument("--causal_blend_graph_weight", type=float, default=0.50)
    parser.add_argument("--structure_rerank_enabled", type=str, default="true")
    parser.add_argument("--structure_rerank_top_n", type=int, default=40)
    parser.add_argument("--structure_rerank_bonus_weight", type=float, default=0.08)
    parser.add_argument("--structure_rerank_min_edge_support", type=int, default=2)
    parser.add_argument("--structure_rerank_max_top5_swaps", type=int, default=2)
    parser.add_argument("--structure_rerank_seed_top_k", type=int, default=4)
    parser.add_argument("--structure_rerank_max_hops", type=int, default=2)
    parser.add_argument("--structure_rerank_margin_threshold", type=float, default=0.02)
    parser.add_argument("--rerank_require_non_empty", type=str, default="true")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    if save_dir == "outputs":
        save_dir = os.path.join(save_dir, dataset_name)
    else:
        save_dir = f"{save_dir}_{dataset_name}"
    args.save_dir = save_dir

    corpus_path = Path(f"reproduce/dataset/{dataset_name}_corpus.json")
    sample_path = Path(f"reproduce/dataset/{dataset_name}.json")
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset_name, corpus=corpus)

    config = build_config(args, corpus_len=len(corpus))
    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    query_solutions, responses, metadata, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )

    slice_metrics = compute_slice_metrics(
        config=config,
        query_solutions=query_solutions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )
    result = {
        "dataset": dataset_name,
        "limit": len(samples),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "config": {
            "causal_enabled": config.causal_enabled,
            "causal_query_only": config.causal_query_only,
            "causal_seed_top_k": config.causal_seed_top_k,
            "causal_confidence_threshold": config.causal_confidence_threshold,
            "causal_damping": config.causal_damping,
            "causal_blend_dense_weight": config.causal_blend_dense_weight,
            "causal_blend_fact_weight": config.causal_blend_fact_weight,
            "causal_blend_graph_weight": config.causal_blend_graph_weight,
            "structure_rerank_enabled": config.structure_rerank_enabled,
            "structure_rerank_top_n": config.structure_rerank_top_n,
            "structure_rerank_bonus_weight": config.structure_rerank_bonus_weight,
            "structure_rerank_min_edge_support": config.structure_rerank_min_edge_support,
            "structure_rerank_max_top5_swaps": config.structure_rerank_max_top5_swaps,
            "structure_rerank_seed_top_k": config.structure_rerank_seed_top_k,
            "structure_rerank_max_hops": config.structure_rerank_max_hops,
            "structure_rerank_margin_threshold": config.structure_rerank_margin_threshold,
        },
        "overall_from_pipeline": {
            **(overall_retrieval_result or {}),
            **(overall_qa_results or {}),
        },
        "overall_recomputed": slice_metrics["overall"],
        "causal_slice": slice_metrics["causal_slice"],
        "examples": [
            {
                "question": query_solution.question,
                "query_type": route_query_type(query_solution.question),
                "answer": query_solution.answer,
                "gold_answers": query_solution.gold_answers,
                "docs": query_solution.docs[:3],
            }
            for query_solution in query_solutions
        ],
    }

    output_json = args.output_json
    if output_json is None:
        report_dir = Path(save_dir) / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_json = report_dir / f"causal_eval_{dataset_name}_{len(samples)}_{args.llm_name.replace('/', '_')}.json"
    else:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(json.dumps({
        "output_json": str(output_json),
        "overall_recomputed": result["overall_recomputed"],
        "causal_slice": result["causal_slice"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
