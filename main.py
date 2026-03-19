import os
from typing import List
import json

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--embedding_base_url', type=str, default=None, help='Embedding base URL')
    parser.add_argument('--max_retry_attempts', type=int, default=5, help='Max retry attempts for each LLM API call')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    parser.add_argument('--planner_enabled', type=str, default='false', help='Enable planner-based retrieval path.')
    parser.add_argument('--planner_mode', choices=['none', 'myopic'], default='none', help='Planner mode to use.')
    parser.add_argument('--planner_max_steps', type=int, default=3, help='Maximum planner steps per query.')
    parser.add_argument('--causal_enabled', type=str, default='true', help='Enable causal extraction and causal-aware retrieval.')
    parser.add_argument('--causal_query_only', type=str, default='true', help='Only use causal retrieval for causal queries.')
    parser.add_argument('--causal_gate_mode', choices=['hard', 'soft'], default='hard', help='Causal gate mode: hard keeps routed gating, soft uses a rule-based intent score.')
    parser.add_argument('--causal_seed_top_k', type=int, default=20, help='How many fact seeds to use for the causal retriever.')
    parser.add_argument('--causal_confidence_threshold', type=float, default=0.5, help='Minimum confidence for causal edges.')
    parser.add_argument('--causal_damping', type=float, default=0.7, help='Damping factor for causal graph propagation.')
    parser.add_argument('--causal_blend_dense_weight', type=float, default=0.35, help='Dense score weight in causal blending.')
    parser.add_argument('--causal_blend_fact_weight', type=float, default=0.15, help='Original fact-graph score weight in causal blending.')
    parser.add_argument('--causal_blend_graph_weight', type=float, default=0.50, help='Causal graph score weight in causal blending.')
    parser.add_argument('--use_local_qwen3', type=str, default='false', help='Use local Qwen3-8B chat server and local embedding server presets.')
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    embedding_name = args.embedding_name
    embedding_base_url = args.embedding_base_url
    use_local_qwen3 = string_to_bool(args.use_local_qwen3)

    if use_local_qwen3:
        llm_base_url = 'http://localhost:8039/v1'
        llm_name = 'qwen3-8b'
        embedding_name = 'VLLM//mnt/nvme/Qwen3-Embedding-8B'
        embedding_base_url = 'http://localhost:8018/v1/embeddings'

    if save_dir == 'outputs':
        save_dir = save_dir + '/' + dataset_name
    else:
        save_dir = save_dir + '_' + dataset_name

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)
    planner_enabled = string_to_bool(args.planner_enabled)
    causal_enabled = string_to_bool(args.causal_enabled)
    causal_query_only = string_to_bool(args.causal_query_only)

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        embedding_base_url=embedding_base_url,
        dataset=dataset_name,
        embedding_model_name=embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        max_retry_attempts=args.max_retry_attempts,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
        planner_enabled=planner_enabled,
        planner_mode=args.planner_mode,
        planner_max_steps=args.planner_max_steps,
        causal_enabled=causal_enabled,
        causal_query_only=causal_query_only,
        causal_gate_mode=args.causal_gate_mode,
        causal_seed_top_k=args.causal_seed_top_k,
        causal_confidence_threshold=args.causal_confidence_threshold,
        causal_damping=args.causal_damping,
        causal_blend_dense_weight=args.causal_blend_dense_weight,
        causal_blend_fact_weight=args.causal_blend_fact_weight,
        causal_blend_graph_weight=args.causal_blend_graph_weight,
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)

    # Retrieval and QA
    hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)

if __name__ == "__main__":
    main()
