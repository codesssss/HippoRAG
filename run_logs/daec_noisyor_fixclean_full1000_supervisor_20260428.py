#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/mnt/nvme/code/HippoRAG")
RUN_TAG = "fixclean_full1000_20260428"
PORTS = [8041, 8042, 8043]
PYTHON = os.environ.get("PYTHON_BIN", "python")
SUPERVISOR_LOG = ROOT / "run_logs" / f"daec_noisyor_{RUN_TAG}_supervisor.log"


JOBS = [
    {
        "name": "daec_noisyor_dense_pool100_2wiki",
        "dataset": "2wikimultihopqa",
        "pool_json": "run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json",
        "pool_source": "dense_pool100",
        "output_json": "run_logs/daec_noisyor_dense_pool100_2wiki_fixclean_full1000_20260428.json",
    },
    {
        "name": "daec_noisyor_dense_pool100_hotpotqa",
        "dataset": "hotpotqa",
        "pool_json": "run_logs/dense_pool_exports_full1000_20260424/hotpotqa_dense_pool100.json",
        "pool_source": "dense_pool100",
        "output_json": "run_logs/daec_noisyor_dense_pool100_hotpotqa_fixclean_full1000_20260428.json",
    },
    {
        "name": "daec_noisyor_dense_pool100_musique",
        "dataset": "musique",
        "pool_json": "run_logs/dense_pool_exports_full1000_20260424/musique_dense_pool100.json",
        "pool_source": "dense_pool100",
        "output_json": "run_logs/daec_noisyor_dense_pool100_musique_fixclean_full1000_20260428.json",
    },
    {
        "name": "daec_noisyor_proprag_pool100_2wiki",
        "dataset": "2wikimultihopqa",
        "pool_json": "run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json",
        "pool_source": "proprag_pool100",
        "output_json": "run_logs/daec_noisyor_proprag_pool100_2wiki_fixclean_full1000_20260428.json",
    },
]


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def output_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    sw = data.get("setwise_selector_qa", {})
    return sw.get("selector_EM") is not None and sw.get("selector_F1") is not None


def port_has_external_client(port: int) -> bool:
    needle = f"localhost:{port}/v1"
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except Exception:
        return False
    self_pid = os.getpid()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        pid = int(parts[0]) if parts and parts[0].isdigit() else -1
        cmd = parts[1] if len(parts) > 1 else ""
        if pid == self_pid:
            continue
        if needle in cmd and "vllm.entrypoints.openai.api_server" not in cmd:
            return True
    return False


def build_command(job: dict[str, str], port: int) -> list[str]:
    return [
        PYTHON,
        "scripts/eval_causal_qwen3.py",
        "--dataset",
        job["dataset"],
        "--limit",
        "1000",
        "--save_dir",
        "outputs_step0_general_nvembed",
        "--llm_name",
        "qwen3-8b",
        "--llm_request_name",
        "qwen3-8b-train",
        "--llm_base_url",
        f"http://localhost:{port}/v1",
        "--embedding_name",
        "VLLM/nvidia/NV-Embed-v2",
        "--embedding_base_url",
        "http://localhost:8019/v1/embeddings",
        "--external_pool_json",
        job["pool_json"],
        "--external_pool_source_name",
        job["pool_source"],
        "--external_pool_strict_questions",
        "true",
        "--setwise_selector",
        "daec_noisyor",
        "--setwise_pool_k",
        "100",
        "--qa_top_k",
        "5",
        "--dtc_decomposition_mode",
        "llm",
        "--dtc_binding_max_candidates",
        "5",
        "--output_json",
        job["output_json"],
    ]


def launch(job: dict[str, str], port: int) -> tuple[subprocess.Popen[bytes], object]:
    log_path = ROOT / "run_logs" / f"{job['name']}_{RUN_TAG}.log"
    pid_path = ROOT / "run_logs" / f"{job['name']}_{RUN_TAG}.pid"
    handle = log_path.open("wb")
    process = subprocess.Popen(
        build_command(job, port),
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    log(f"launched {job['name']} on {port}: pid={process.pid} log={log_path} json={job['output_json']}")
    return process, handle


def main() -> int:
    os.chdir(ROOT)
    log("DAEC noisy-OR full1000 supervisor started")
    pending: list[dict[str, str]] = []
    for job in JOBS:
        out = ROOT / job["output_json"]
        if output_complete(out):
            log(f"skip complete output for {job['name']}: {out}")
            continue
        job = dict(job)
        job["attempt"] = "0"
        pending.append(job)

    running: dict[int, tuple[dict[str, str], subprocess.Popen[bytes], object]] = {}
    completed: list[str] = []
    failed: list[str] = []

    while pending or running:
        for port, (job, process, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            out = ROOT / job["output_json"]
            if code == 0 and output_complete(out):
                completed.append(job["name"])
                log(f"completed {job['name']} on {port}: code=0 json={out}")
            else:
                attempt = int(job.get("attempt", "0"))
                if attempt < 1:
                    retry_job = dict(job)
                    retry_job["attempt"] = str(attempt + 1)
                    pending.insert(0, retry_job)
                    log(f"retrying {job['name']} after code={code}; attempt={attempt + 1}")
                else:
                    failed.append(job["name"])
                    log(f"failed {job['name']} on {port}: code={code} json_complete={output_complete(out)}")
            del running[port]

        for port in PORTS:
            if not pending:
                break
            if port in running:
                continue
            if port_has_external_client(port):
                continue
            job = pending.pop(0)
            process, handle = launch(job, port)
            running[port] = (job, process, handle)
            time.sleep(5)

        if pending or running:
            active = ", ".join(f"{port}:{job['name']}:{proc.pid}" for port, (job, proc, _) in running.items())
            log(f"heartbeat pending={len(pending)} running=[{active}]")
            time.sleep(60)

    log(f"DAEC noisy-OR full1000 supervisor finished completed={completed} failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
