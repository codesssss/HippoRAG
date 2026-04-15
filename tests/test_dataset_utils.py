import json
import sys
import types
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.completion = lambda **kwargs: None
    sys.modules["litellm"] = litellm_stub
if "gritlm" not in sys.modules:
    gritlm_stub = types.ModuleType("gritlm")
    gritlm_stub.GritLM = object
    sys.modules["gritlm"] = gritlm_stub
if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers_stub
if "igraph" not in sys.modules:
    igraph_stub = types.ModuleType("igraph")
    igraph_stub.Graph = object
    sys.modules["igraph"] = igraph_stub
if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = object
    vllm_stub.LLM = object
    sys.modules["vllm"] = vllm_stub
if "outlines" not in sys.modules:
    outlines_stub = types.ModuleType("outlines")
    outlines_generate_stub = types.ModuleType("outlines.generate")
    outlines_generate_stub.json = lambda *args, **kwargs: (lambda prompts, **inner_kwargs: [])
    outlines_models_stub = types.ModuleType("outlines.models")
    outlines_models_stub.Transformers = object
    sys.modules["outlines"] = outlines_stub
    sys.modules["outlines.generate"] = outlines_generate_stub
    sys.modules["outlines.models"] = outlines_models_stub

from eval_causal_qwen3 import get_gold_answers, get_gold_docs
from src.hipporag.utils.dataset_utils import resolve_dataset_file_stem, resolve_dataset_paths


def test_resolve_dataset_file_stem_maps_nq_aliases():
    assert resolve_dataset_file_stem("nq") == "nq_rear"
    assert resolve_dataset_file_stem("naturalquestions") == "nq_rear"
    assert resolve_dataset_file_stem("popqa") == "popqa"


def test_resolve_dataset_paths_points_nq_to_existing_rear_files():
    corpus_path, sample_path = resolve_dataset_paths("nq", ROOT_DIR / "reproduce" / "dataset")
    assert corpus_path.name == "nq_rear_corpus.json"
    assert sample_path.name == "nq_rear.json"
    assert corpus_path.exists()
    assert sample_path.exists()


def test_popqa_gold_docs_and_answers_extract_from_real_dataset_files():
    corpus_path, sample_path = resolve_dataset_paths("popqa", ROOT_DIR / "reproduce" / "dataset")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    samples = json.loads(sample_path.read_text(encoding="utf-8"))[:3]

    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, "popqa", corpus=corpus)

    assert len(gold_answers) == len(samples)
    assert len(gold_docs) == len(samples)
    assert all(gold_answers)
    assert all(gold_docs)


def test_nq_gold_docs_and_answers_extract_from_alias_dataset_files():
    corpus_path, sample_path = resolve_dataset_paths("nq", ROOT_DIR / "reproduce" / "dataset")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    samples = json.loads(sample_path.read_text(encoding="utf-8"))[:3]

    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, "nq", corpus=corpus)

    assert len(gold_answers) == len(samples)
    assert len(gold_docs) == len(samples)
    assert all(gold_answers)
    assert all(gold_docs)
