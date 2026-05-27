from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def toy_pool(tmp_path: Path) -> Path:
    path = tmp_path / "pool.json"
    write_json(
        path,
        {
            "records": [
                {
                    "qid": "q1",
                    "query_idx": 0,
                    "question": "What city was Alice born in?",
                    "gold_answers": ["Paris"],
                    "gold_titles": ["Alice", "Paris"],
                    "initial_answer": "Paris",
                    "pool_docs": [
                        "Alice\nAlice was born in Paris.",
                        "Paris\nParis is a city in France.",
                        "Noise\nUnrelated text.",
                    ],
                    "pool_titles": ["Alice", "Paris", "Noise"],
                    "pool_doc_scores": [1.0, 0.8, 0.1],
                }
            ]
        },
    )
    return path


def toy_obligations(tmp_path: Path, name: str = "obligations.jsonl") -> Path:
    path = tmp_path / name
    write_jsonl(
        path,
        [
            {
                "qid": "q1",
                "obligations": [
                    {
                        "id": "o1",
                        "claim": "Paris is the city where Alice was born.",
                        "type": "factual",
                        "retrieval_active": True,
                    }
                ],
            }
        ],
    )
    return path


def toy_cot_queries(tmp_path: Path) -> Path:
    path = tmp_path / "cot.jsonl"
    write_jsonl(path, [{"qid": "q1", "query": "Find the city where Alice was born."}])
    return path


def run_script(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT_DIR, check=True)


def test_oracle_script_writes_template_when_oracle_missing(tmp_path: Path) -> None:
    pool = toy_pool(tmp_path)
    oracle = tmp_path / "missing_oracle.jsonl"
    out_json = tmp_path / "oracle.json"
    out_md = tmp_path / "oracle.md"

    run_script(
        [
            "scripts/arec_rag_oracle_obligation_ceiling.py",
            "--pool_json",
            str(pool),
            "--dataset",
            "toy",
            "--oracle_obligations_jsonl",
            str(oracle),
            "--ircot_prompt_path",
            "official_ircot_prompt.txt",
            "--verifier_backend",
            "lexical_smoke",
            "--output_json",
            str(out_json),
            "--output_md",
            str(out_md),
            "--rows_jsonl",
            str(tmp_path / "oracle_rows.jsonl"),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "needs_oracle_obligations"
    assert oracle.exists()


def test_closure_and_residual_scripts_run_with_lexical_smoke(tmp_path: Path) -> None:
    pool = toy_pool(tmp_path)
    obligations = toy_obligations(tmp_path)
    oracle = toy_obligations(tmp_path, "oracle.jsonl")
    cot = toy_cot_queries(tmp_path)

    closure_json = tmp_path / "closure.json"
    closure_rows = tmp_path / "closure_rows.jsonl"
    run_script(
        [
            "scripts/arec_rag_smoke_closure.py",
            "--pool_json",
            str(pool),
            "--generated_obligations_jsonl",
            str(obligations),
            "--oracle_obligations_jsonl",
            str(oracle),
            "--dataset",
            "toy",
            "--verifier_backend",
            "lexical_smoke",
            "--output_json",
            str(closure_json),
            "--output_md",
            str(tmp_path / "closure.md"),
            "--rows_jsonl",
            str(closure_rows),
        ]
    )
    closure_payload = json.loads(closure_json.read_text(encoding="utf-8"))
    assert closure_payload["status"] == "completed"
    assert closure_payload["summary"]["rows"] == 1

    residual_json = tmp_path / "residual.json"
    residual_rows = tmp_path / "residual_rows.jsonl"
    run_script(
        [
            "scripts/arec_rag_smoke_residual_retrieval.py",
            "--pool_json",
            str(pool),
            "--generated_obligations_jsonl",
            str(obligations),
            "--cot_queries_jsonl",
            str(cot),
            "--dataset",
            "toy",
            "--verifier_backend",
            "lexical_smoke",
            "--output_json",
            str(residual_json),
            "--output_md",
            str(tmp_path / "residual.md"),
            "--rows_jsonl",
            str(residual_rows),
        ]
    )
    residual_payload = json.loads(residual_json.read_text(encoding="utf-8"))
    assert residual_payload["status"] == "completed"
    assert residual_payload["summary"]["rows"] == 1
    row = json.loads(residual_rows.read_text(encoding="utf-8").splitlines()[0])
    assert "arec_residual_titles" in row

