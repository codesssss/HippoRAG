import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.causal_utils import route_query_type

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import build_config, compute_slice_metrics, get_gold_answers, get_gold_docs


logger = logging.getLogger(__name__)


def build_examples(config: BaseConfig,
                   query_solutions,
                   gold_docs: list[list[str]],
                   gold_answers: list[list[str]]) -> list[dict]:
    retrieval = RetrievalRecall(global_config=config)
    qa_em = QAExactMatch(global_config=config)
    qa_f1 = QAF1Score(global_config=config)

    retrieved_docs = [query_solution.docs for query_solution in query_solutions]
    predicted_answers = [query_solution.answer for query_solution in query_solutions]

    _, recall_examples = retrieval.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=retrieved_docs,
        k_list=[1, 2, 5, 10, 20],
    )
    _, em_examples = qa_em.calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )
    _, f1_examples = qa_f1.calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )

    examples = []
    for idx, query_solution in enumerate(query_solutions):
        examples.append({
            "question": query_solution.question,
            "query_type": route_query_type(query_solution.question),
            "answer": query_solution.answer,
            "gold_answers": query_solution.gold_answers,
            "docs": query_solution.docs[:3],
            "metrics": {
                **recall_examples[idx],
                **em_examples[idx],
                **f1_examples[idx],
            },
            "retrieval_trace": query_solution.retrieval_trace or {},
            "qa_trace": query_solution.qa_trace or {},
        })
    return examples


def summarize_structure_traces(examples: list[dict]) -> dict:
    noop_reason_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    final_failure_reason_counts: dict[str, int] = {}
    facts_empty_stage_counts: dict[str, int] = {}
    enabled_examples = []
    effective_enabled_examples = []
    top5_membership_changed_examples = []
    top5_order_changed_examples = []
    parse_failure_examples = 0
    schema_failure_examples = 0
    empty_output_examples = 0
    semantic_empty_examples = 0
    repair_failure_examples = 0
    non_empty_fallback_examples = 0
    final_facts_empty_examples = 0
    use_causal_path_count = 0
    causal_doc_non_empty_count = 0
    route_causal_intent_scores = []
    causal_doc_counts_when_used = []
    causal_weights_after_attenuation_when_used = []

    for example in examples:
        trace = example.get("retrieval_trace") or {}
        noop_reason = str(trace.get("noop_reason", "unknown"))
        noop_reason_counts[noop_reason] = noop_reason_counts.get(noop_reason, 0) + 1
        fallback_reason = trace.get("rerank_fallback_reason")
        if fallback_reason is not None:
            fallback_reason = str(fallback_reason)
            fallback_reason_counts[fallback_reason] = fallback_reason_counts.get(fallback_reason, 0) + 1
        final_failure_reason = trace.get("rerank_final_failure_reason")
        if final_failure_reason is not None:
            final_failure_reason = str(final_failure_reason)
            final_failure_reason_counts[final_failure_reason] = (
                final_failure_reason_counts.get(final_failure_reason, 0) + 1
            )
        facts_empty_stage = trace.get("rerank_facts_empty_stage")
        if facts_empty_stage is not None:
            facts_empty_stage = str(facts_empty_stage)
            facts_empty_stage_counts[facts_empty_stage] = facts_empty_stage_counts.get(facts_empty_stage, 0) + 1
        if trace.get("applied"):
            enabled_examples.append(trace)
            if int(trace.get("num_swaps_top5", 0)) > 0:
                effective_enabled_examples.append(trace)
        if float(trace.get("top5_jaccard", 1.0)) < 1.0 or int(trace.get("moved_out_of_top5", 0)) > 0:
            top5_membership_changed_examples.append(trace)
        if int(trace.get("num_swaps_top5", 0)) > 0:
            top5_order_changed_examples.append(trace)
        if int(trace.get("rerank_parse_failure_count", 0)) > 0:
            parse_failure_examples += 1
        if int(trace.get("rerank_schema_failure_count", 0)) > 0:
            schema_failure_examples += 1
        if int(trace.get("rerank_empty_output_count", 0)) > 0:
            empty_output_examples += 1
        if int(trace.get("rerank_model_semantic_empty_count", 0)) > 0:
            semantic_empty_examples += 1
        if int(trace.get("rerank_repair_failure_count", 0)) > 0:
            repair_failure_examples += 1
        if int(trace.get("rerank_non_empty_fallback_count", 0)) > 0:
            non_empty_fallback_examples += 1
        if int(trace.get("rerank_final_facts_empty_count", 0)) > 0:
            final_facts_empty_examples += 1
        if trace.get("use_causal_path"):
            use_causal_path_count += 1
            causal_doc_counts_when_used.append(float(trace.get("causal_doc_count", 0)))
            causal_weights_after_attenuation_when_used.append(
                float(trace.get("causal_weight_after_attenuation", 0.0))
            )
        if trace.get("causal_doc_non_empty"):
            causal_doc_non_empty_count += 1
        if "route_causal_intent_score" in trace:
            route_causal_intent_scores.append(float(trace.get("route_causal_intent_score", 0.0)))

    def _avg(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "num_queries": len(examples),
        "structure_enabled_count": len(enabled_examples),
        "structure_enabled_rate": (len(enabled_examples) / len(examples)) if examples else 0.0,
        "effective_applied_count": len(effective_enabled_examples),
        "effective_applied_rate": (len(effective_enabled_examples) / len(examples)) if examples else 0.0,
        "top5_membership_changed_count": len(top5_membership_changed_examples),
        "top5_membership_changed_rate": (len(top5_membership_changed_examples) / len(examples)) if examples else 0.0,
        "top5_order_changed_count": len(top5_order_changed_examples),
        "top5_order_changed_rate": (len(top5_order_changed_examples) / len(examples)) if examples else 0.0,
        "parse_failure_example_count": parse_failure_examples,
        "schema_failure_example_count": schema_failure_examples,
        "empty_output_example_count": empty_output_examples,
        "semantic_empty_example_count": semantic_empty_examples,
        "repair_failure_example_count": repair_failure_examples,
        "non_empty_fallback_example_count": non_empty_fallback_examples,
        "final_facts_empty_example_count": final_facts_empty_examples,
        "use_causal_path_count": use_causal_path_count,
        "use_causal_path_rate": (use_causal_path_count / len(examples)) if examples else 0.0,
        "causal_doc_non_empty_count": causal_doc_non_empty_count,
        "causal_doc_non_empty_rate": (causal_doc_non_empty_count / len(examples)) if examples else 0.0,
        "avg_route_causal_intent_score": _avg(route_causal_intent_scores),
        "avg_causal_doc_count_when_used": _avg(causal_doc_counts_when_used),
        "avg_causal_weight_after_attenuation_when_used": _avg(causal_weights_after_attenuation_when_used),
        "facts_empty_stage_counts": facts_empty_stage_counts,
        "noop_reason_counts": noop_reason_counts,
        "fallback_reason_counts": fallback_reason_counts,
        "final_failure_reason_counts": final_failure_reason_counts,
        "avg_top5_jaccard_when_applied": _avg([float(trace.get("top5_jaccard", 1.0)) for trace in enabled_examples]),
        "avg_num_swaps_top5_when_applied": _avg([float(trace.get("num_swaps_top5", 0.0)) for trace in enabled_examples]),
        "avg_moved_out_of_top5_when_applied": _avg([float(trace.get("moved_out_of_top5", 0.0)) for trace in enabled_examples]),
        "avg_scored_doc_count_when_applied": _avg([float(trace.get("scored_doc_count", 0.0)) for trace in enabled_examples]),
        "avg_num_swaps_top5_when_effective": _avg(
            [float(trace.get("num_swaps_top5", 0.0)) for trace in effective_enabled_examples]
        ),
    }


def compute_paired_summary(baseline_examples: list[dict], structure_examples: list[dict]) -> dict:
    assert len(baseline_examples) == len(structure_examples)

    summary = {
        "num_queries": len(baseline_examples),
        "improve_em_count": 0,
        "hurt_em_count": 0,
        "tie_em_count": 0,
        "improve_f1_count": 0,
        "hurt_f1_count": 0,
        "tie_f1_count": 0,
        "structure_triggered_count": 0,
        "triggered_improve_em_count": 0,
        "triggered_hurt_em_count": 0,
        "triggered_improve_f1_count": 0,
        "triggered_hurt_f1_count": 0,
        "effective_triggered_count": 0,
        "reader_prompt_changed_count": 0,
        "reader_response_changed_count": 0,
        "effective_triggered_improve_f1_count": 0,
        "effective_triggered_hurt_f1_count": 0,
    }

    for baseline_example, structure_example in zip(baseline_examples, structure_examples):
        baseline_em = float(baseline_example["metrics"]["ExactMatch"])
        structure_em = float(structure_example["metrics"]["ExactMatch"])
        baseline_f1 = float(baseline_example["metrics"]["F1"])
        structure_f1 = float(structure_example["metrics"]["F1"])
        structure_trace = structure_example.get("retrieval_trace") or {}
        triggered = bool(structure_trace.get("applied"))
        effective_triggered = triggered and int(structure_trace.get("num_swaps_top5", 0)) > 0
        baseline_qa_trace = baseline_example.get("qa_trace") or {}
        structure_qa_trace = structure_example.get("qa_trace") or {}
        prompt_changed = (
            baseline_qa_trace.get("reader_prompt_hash") is not None
            and structure_qa_trace.get("reader_prompt_hash") is not None
            and baseline_qa_trace.get("reader_prompt_hash") != structure_qa_trace.get("reader_prompt_hash")
        )
        response_changed = (
            baseline_qa_trace.get("reader_response_hash") is not None
            and structure_qa_trace.get("reader_response_hash") is not None
            and baseline_qa_trace.get("reader_response_hash") != structure_qa_trace.get("reader_response_hash")
        )

        if structure_em > baseline_em:
            summary["improve_em_count"] += 1
            if triggered:
                summary["triggered_improve_em_count"] += 1
        elif structure_em < baseline_em:
            summary["hurt_em_count"] += 1
            if triggered:
                summary["triggered_hurt_em_count"] += 1
        else:
            summary["tie_em_count"] += 1

        if structure_f1 > baseline_f1:
            summary["improve_f1_count"] += 1
            if triggered:
                summary["triggered_improve_f1_count"] += 1
            if effective_triggered:
                summary["effective_triggered_improve_f1_count"] += 1
        elif structure_f1 < baseline_f1:
            summary["hurt_f1_count"] += 1
            if triggered:
                summary["triggered_hurt_f1_count"] += 1
            if effective_triggered:
                summary["effective_triggered_hurt_f1_count"] += 1
        else:
            summary["tie_f1_count"] += 1

        if triggered:
            summary["structure_triggered_count"] += 1
        if effective_triggered:
            summary["effective_triggered_count"] += 1
        if prompt_changed:
            summary["reader_prompt_changed_count"] += 1
        if response_changed:
            summary["reader_response_changed_count"] += 1

    return summary


def build_paired_examples(baseline_examples: list[dict], structure_examples: list[dict]) -> list[dict]:
    paired_examples = []
    for baseline_example, structure_example in zip(baseline_examples, structure_examples):
        paired_examples.append({
            "question": baseline_example["question"],
            "query_type": baseline_example["query_type"],
            "baseline": {
                "answer": baseline_example["answer"],
                "metrics": baseline_example["metrics"],
                "qa_trace": baseline_example.get("qa_trace") or {},
            },
            "structure": {
                "answer": structure_example["answer"],
                "metrics": structure_example["metrics"],
                "retrieval_trace": structure_example.get("retrieval_trace") or {},
                "qa_trace": structure_example.get("qa_trace") or {},
            },
            "delta": {
                "ExactMatch": float(structure_example["metrics"]["ExactMatch"]) - float(baseline_example["metrics"]["ExactMatch"]),
                "F1": float(structure_example["metrics"]["F1"]) - float(baseline_example["metrics"]["F1"]),
                "Recall@1": float(structure_example["metrics"]["Recall@1"]) - float(baseline_example["metrics"]["Recall@1"]),
                "Recall@2": float(structure_example["metrics"]["Recall@2"]) - float(baseline_example["metrics"]["Recall@2"]),
                "Recall@5": float(structure_example["metrics"]["Recall@5"]) - float(baseline_example["metrics"]["Recall@5"]),
            },
        })
    return paired_examples


def run_single(label: str,
               config: BaseConfig,
               docs: list[str],
               queries: list[str],
               gold_docs: list[list[str]],
               gold_answers: list[list[str]]) -> dict:
    logger.info("Starting run: %s", label)
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    query_solutions, _, _, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(
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
    examples = build_examples(
        config=config,
        query_solutions=query_solutions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )
    return {
        "config": {
            "causal_enabled": config.causal_enabled,
            "causal_query_only": config.causal_query_only,
            "causal_gate_mode": config.causal_gate_mode,
            "structure_rerank_enabled": config.structure_rerank_enabled,
            "structure_rerank_top_n": config.structure_rerank_top_n,
            "structure_rerank_bonus_weight": config.structure_rerank_bonus_weight,
            "structure_rerank_min_edge_support": config.structure_rerank_min_edge_support,
            "structure_rerank_max_top5_swaps": config.structure_rerank_max_top5_swaps,
            "structure_rerank_seed_top_k": config.structure_rerank_seed_top_k,
            "structure_rerank_max_hops": config.structure_rerank_max_hops,
            "structure_rerank_margin_threshold": config.structure_rerank_margin_threshold,
            "rerank_require_non_empty": config.rerank_require_non_empty,
        },
        "overall_from_pipeline": {
            **(overall_retrieval_result or {}),
            **(overall_qa_results or {}),
        },
        "overall_recomputed": slice_metrics["overall"],
        "causal_slice": slice_metrics["causal_slice"],
        "structure_analysis": summarize_structure_traces(examples),
        "examples": examples,
    }


def compute_delta(baseline: dict, structure: dict) -> dict:
    keys = sorted(set(baseline["overall_recomputed"]) | set(structure["overall_recomputed"]))
    delta = {}
    for key in keys:
        baseline_value = baseline["overall_recomputed"].get(key)
        structure_value = structure["overall_recomputed"].get(key)
        if isinstance(baseline_value, (int, float)) and isinstance(structure_value, (int, float)):
            delta[key] = structure_value - baseline_value
    return delta


def main():
    parser = argparse.ArgumentParser(description="Compare baseline retrieval vs structure rerank on the same dataset slice.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", type=str, default=None)
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
    parser.add_argument("--causal_gate_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--causal_seed_top_k", type=int, default=20)
    parser.add_argument("--causal_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--causal_damping", type=float, default=0.7)
    parser.add_argument("--causal_blend_dense_weight", type=float, default=0.35)
    parser.add_argument("--causal_blend_fact_weight", type=float, default=0.15)
    parser.add_argument("--causal_blend_graph_weight", type=float, default=0.50)
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
    if args.save_dir == "outputs":
        args.save_dir = f"outputs/{dataset_name}"
    else:
        args.save_dir = f"{args.save_dir}_{dataset_name}"

    logging.basicConfig(level=logging.INFO)

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

    base_args = deepcopy(args)
    base_args.structure_rerank_enabled = "false"
    base_config = build_config(base_args, corpus_len=len(corpus))

    structure_args = deepcopy(args)
    structure_args.structure_rerank_enabled = "true"
    structure_config = build_config(structure_args, corpus_len=len(corpus))

    baseline_result = run_single(
        label="baseline",
        config=base_config,
        docs=docs,
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )
    structure_result = run_single(
        label="structure",
        config=structure_config,
        docs=docs,
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )

    report = {
        "dataset": dataset_name,
        "limit": len(samples),
        "baseline": baseline_result,
        "structure": structure_result,
        "delta": compute_delta(baseline_result, structure_result),
        "paired_summary": compute_paired_summary(
            baseline_result["examples"],
            structure_result["examples"],
        ),
        "paired_examples": build_paired_examples(
            baseline_result["examples"],
            structure_result["examples"],
        ),
    }

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = Path(args.save_dir) / "eval_reports" / (
            f"struct_compare_{dataset_name}_{len(samples)}_{args.llm_name.replace('/', '_')}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, output_path.open("w"), ensure_ascii=False, indent=2)
    print(json.dumps({"output_json": str(output_path), "delta": report["delta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
