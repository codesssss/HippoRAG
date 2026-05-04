import importlib.util
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_script_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


eval_causal_qwen3 = load_script_module("eval_causal_qwen3", "scripts/eval_causal_qwen3.py")
export_hipporag_pool = load_script_module("export_hipporag_pool", "scripts/export_hipporag_pool.py")


def test_nq_dataset_alias_resolves_to_rear_files():
    assert eval_causal_qwen3.resolve_dataset_file_stem("nq") == "nq_rear"
    assert eval_causal_qwen3.resolve_dataset_file_stem("natural_questions") == "nq_rear"
    assert export_hipporag_pool.resolve_dataset_file_stem("nq") == "nq_rear"


def test_popqa_string_aliases_are_parsed_for_eval_answers():
    answers = eval_causal_qwen3.get_gold_answers([
        {
            "obj": "politician",
            "possible_answers": '["politician", "political leader"]',
            "o_wiki_title": "Politician",
            "o_aliases": '["political figure", "pol"]',
        }
    ])[0]

    assert set(answers) == {
        "politician",
        "political leader",
        "Politician",
        "political figure",
        "pol",
    }


def test_popqa_string_aliases_are_parsed_for_pool_export_answers():
    answers = export_hipporag_pool.get_gold_answers([
        {
            "obj": "politician",
            "possible_answers": '["politician", "political leader"]',
            "o_wiki_title": "Politician",
            "o_aliases": '["political figure", "pol"]',
        }
    ])[0]

    assert set(answers) == {
        "politician",
        "political leader",
        "Politician",
        "political figure",
        "pol",
    }
