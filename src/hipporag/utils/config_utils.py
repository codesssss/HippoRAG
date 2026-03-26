import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online", "Transformers-offline"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    causal_enabled: bool = field(
        default=True,
        metadata={"help": "Enable causal extraction and causal-aware retrieval."}
    )
    causal_query_only: bool = field(
        default=True,
        metadata={"help": "Only run the causal retriever for queries routed as causal."}
    )
    causal_gate_mode: Literal["hard", "soft"] = field(
        default="hard",
        metadata={"help": "Causal gate mode: hard keeps the existing routed gate, soft attenuates causal usage by a rule-based intent score."}
    )
    causal_seed_top_k: int = field(
        default=20,
        metadata={"help": "How many proposition seeds to use for causal graph retrieval."}
    )
    causal_confidence_threshold: float = field(
        default=0.5,
        metadata={"help": "Minimum confidence required to keep an extracted causal relation."}
    )
    causal_damping: float = field(
        default=0.7,
        metadata={"help": "Damping factor for causal graph personalized PageRank."}
    )
    causal_blend_dense_weight: float = field(
        default=0.35,
        metadata={"help": "Dense retrieval weight in causal-aware score blending."}
    )
    causal_blend_fact_weight: float = field(
        default=0.15,
        metadata={"help": "Original fact-graph retrieval weight in causal-aware score blending."}
    )
    causal_blend_graph_weight: float = field(
        default=0.50,
        metadata={"help": "Causal graph retrieval weight in causal-aware score blending."}
    )
    causal_margin_gate_enabled: bool = field(
        default=False,
        metadata={"help": "Only allow causal score blending when the non-causal baseline ranking is uncertain."}
    )
    causal_margin_threshold: float = field(
        default=0.02,
        metadata={"help": "Minimum normalized baseline top1-top2 margin that disables causal blending when margin gating is enabled."}
    )
    causal_blend_top_k: int = field(
        default=0,
        metadata={"help": "Only allow the top-K causal documents to contribute causal scores. Set to 0 to keep all causal docs."}
    )
    causal_engine_version: Literal["legacy", "v2"] = field(
        default="legacy",
        metadata={"help": "Which causal engine to use. `legacy` keeps the old OpenIE+graph pipeline, `v2` uses schema extraction + semantic routing + serialized subgraph context."}
    )
    causal_router_type: Literal["embedding_anchor"] = field(
        default="embedding_anchor",
        metadata={"help": "Semantic router backend for causal V2."}
    )
    causal_router_anchor_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a JSON anchor bank for the causal V2 semantic router."}
    )
    causal_router_causal_threshold: float = field(
        default=0.62,
        metadata={"help": "Minimum causal-anchor similarity required for causal V2 routing."}
    )
    causal_router_standard_threshold: float = field(
        default=0.62,
        metadata={"help": "Minimum standard-anchor similarity that keeps a query on the standard path when the causal margin is weak."}
    )
    causal_router_margin_threshold: float = field(
        default=0.02,
        metadata={"help": "Minimum causal-vs-standard similarity margin required to route a query into the causal V2 path."}
    )
    causal_v2_probe_mode: Literal["router", "always"] = field(
        default="router",
        metadata={"help": "V2 subgraph probing mode. `router` only probes routed causal queries, `always` probes every query while keeping router scores for analysis."}
    )
    causal_v2_graph_mode: Literal["causal", "general"] = field(
        default="causal",
        metadata={"help": "V2 graph mode. `causal` keeps cause/enable/prevent extraction, `general` uses a minimal schema-constrained relation graph for multi-hop retrieval."}
    )
    causal_context_injection_mode: Literal["all", "selective"] = field(
        default="all",
        metadata={"help": "How V2 serialized causal chains are injected. `all` keeps the current top-K behavior, `selective` applies query-entity overlap filtering and top-1 gating."}
    )
    causal_context_min_chain_score: float = field(
        default=0.1,
        metadata={"help": "Minimum chain score required for V2 causal context injection in selective mode."}
    )
    causal_v2_base_retrieval_mode: Literal["dense", "legacy_fact_graph"] = field(
        default="dense",
        metadata={"help": "Base ranking used before V2 graph candidate injection. `dense` keeps the current passage-only ranking, `legacy_fact_graph` reuses the original HippoRAG fact-graph retrieval when legacy artifacts are available."}
    )
    causal_v2_legacy_preferred_embedding_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Preferred legacy embedding workspace to reuse when `causal_v2_base_retrieval_mode=legacy_fact_graph`. V2 will try to keep graph and fact/entity embeddings aligned to this workspace when available."}
    )
    causal_v2_candidate_injection_enabled: bool = field(
        default=False,
        metadata={"help": "Enable V2 retrieval candidate injection by expanding dense top-N documents through the V2 graph and inserting newly discovered documents into the candidate pool."}
    )
    causal_v2_candidate_injection_top_n: int = field(
        default=20,
        metadata={"help": "Use the dense top-N documents as the seed candidate set for V2 candidate injection."}
    )
    causal_v2_candidate_injection_max_docs: int = field(
        default=5,
        metadata={"help": "Maximum number of newly injected documents to insert into the reranked candidate list."}
    )
    causal_v2_candidate_injection_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Keep the dense top-K documents fixed ahead of injected documents during V2 candidate injection."}
    )
    causal_v2_candidate_injection_hops: int = field(
        default=1,
        metadata={"help": "Maximum graph hops used when proposing new documents for V2 candidate injection."}
    )
    causal_v2_candidate_injection_blend_weight: float = field(
        default=0.15,
        metadata={"help": "Small graph-score blend weight used when reranking the expanded V2 candidate pool."}
    )
    causal_v2_candidate_injection_require_query_entity_overlap: bool = field(
        default=True,
        metadata={"help": "Only allow V2 candidate injection to expand from dense seed documents that overlap extracted query entities."}
    )
    causal_v2_doc_rerank_enabled: bool = field(
        default=False,
        metadata={"help": "Enable conservative V2 retrieval reranking using selected causal chains as small document bonuses."}
    )
    causal_v2_doc_rerank_top_n: int = field(
        default=20,
        metadata={"help": "Only rerank the top-N dense documents with V2 causal document bonuses."}
    )
    causal_v2_doc_rerank_boost_weight: float = field(
        default=0.05,
        metadata={"help": "Small additive bonus weight for V2 causal document reranking."}
    )
    causal_v2_doc_rerank_protect_top1: bool = field(
        default=True,
        metadata={"help": "Keep the dense top-1 document fixed during V2 causal document reranking."}
    )
    causal_v2_doc_rerank_max_top5_swaps: int = field(
        default=2,
        metadata={"help": "Maximum number of top-5 order changes allowed for V2 causal document reranking."}
    )
    causal_v2_extraction_max_tokens: int = field(
        default=768,
        metadata={"help": "Maximum completion tokens for causal V2 schema-constrained extraction."}
    )
    causal_v2_extraction_retry_attempts: int = field(
        default=2,
        metadata={"help": "How many times to retry causal V2 extraction when schema parsing fails."}
    )
    causal_v2_extraction_workers: int = field(
        default=4,
        metadata={"help": "Parallel worker count for causal V2 extraction."}
    )
    causal_event_top_k: int = field(
        default=8,
        metadata={"help": "Top-K canonical event seeds to retrieve for causal V2 subgraph retrieval."}
    )
    causal_v2_max_hops: int = field(
        default=2,
        metadata={"help": "Maximum directed hops when expanding causal V2 subgraphs."}
    )
    causal_chain_top_k: int = field(
        default=6,
        metadata={"help": "Maximum number of serialized causal chains to pass to the generator."}
    )
    causal_context_max_items: int = field(
        default=6,
        metadata={"help": "Maximum number of causal context items injected into the generator prompt."}
    )
    causal_er_similarity_threshold: float = field(
        default=0.92,
        metadata={"help": "Embedding similarity threshold used by causal V2 event resolution."}
    )
    causal_er_text_threshold: float = field(
        default=0.55,
        metadata={"help": "Textual similarity threshold used by causal V2 event resolution."}
    )
    causal_v2_min_edge_confidence: float = field(
        default=0.7,
        metadata={"help": "Minimum edge confidence kept in the causal V2 graph."}
    )
    structure_rerank_enabled: bool = field(
        default=True,
        metadata={"help": "Enable lightweight directed-structure reranking over the top candidate documents."}
    )
    structure_rerank_top_n: int = field(
        default=40,
        metadata={"help": "Only rerank the top-N candidate docs with structure signals."}
    )
    structure_rerank_bonus_weight: float = field(
        default=0.08,
        metadata={"help": "Small additive bonus weight for structure-based doc reranking."}
    )
    structure_rerank_min_edge_support: int = field(
        default=2,
        metadata={"help": "Minimum number of candidate docs that must receive structure support before reranking is applied."}
    )
    structure_rerank_max_top5_swaps: int = field(
        default=2,
        metadata={"help": "Maximum number of top-5 order swaps allowed for structure reranking; larger perturbations are skipped."}
    )
    structure_rerank_seed_top_k: int = field(
        default=4,
        metadata={"help": "How many top facts to use when deriving structure rerank seed entities."}
    )
    structure_rerank_max_hops: int = field(
        default=2,
        metadata={"help": "How many directed hops to expand when building structure rerank signals."}
    )
    structure_rerank_margin_threshold: float = field(
        default=0.02,
        metadata={"help": "Only apply structure reranking when the top doc-score margin is below this threshold."}
    )
    rerank_require_non_empty: bool = field(
        default=True,
        metadata={"help": "Require reranker to return at least one fact when candidates are non-empty; otherwise keep legacy empty-output behavior."}
    )
    planner_enabled: bool = field(
        default=False,
        metadata={"help": "Enable planner-based retrieval path instead of the default one-shot retriever."}
    )
    planner_mode: Literal["none", "myopic"] = field(
        default="none",
        metadata={"help": "Planner mode to use for retrieval."}
    )
    planner_max_steps: int = field(
        default=3,
        metadata={"help": "Maximum planner steps per query."}
    )
    planner_seed_doc_budget: int = field(
        default=24,
        metadata={"help": "How many docs a seed action can add into the planner candidate pool."}
    )
    planner_entity_doc_budget: int = field(
        default=10,
        metadata={"help": "How many docs an entity expansion can add into the planner candidate pool."}
    )
    planner_max_entity_actions: int = field(
        default=6,
        metadata={"help": "Maximum entity-expansion actions considered by the planner."}
    )
    planner_max_inspect_passages: int = field(
        default=4,
        metadata={"help": "Maximum inspect-passage actions considered by the planner."}
    )
    planner_info_gain_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for planner information gain."}
    )
    planner_relevance_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for planner relevance reward."}
    )
    planner_novelty_weight: float = field(
        default=0.25,
        metadata={"help": "Weight for planner novelty reward."}
    )
    planner_cost_weight: float = field(
        default=0.35,
        metadata={"help": "Weight for planner action cost penalty."}
    )
    planner_min_action_score: float = field(
        default=0.1,
        metadata={"help": "Minimum score required to keep exploring when enough documents are already collected."}
    )
    planner_belief_stop_threshold: float = field(
        default=0.72,
        metadata={"help": "Stop planning once top belief and evidence count are both sufficient."}
    )
    planner_dense_fallback_weight: float = field(
        default=0.92,
        metadata={"help": "Score discount applied when dense retrieval is used only as planner fallback."}
    )
    
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
