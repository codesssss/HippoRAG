#!/usr/bin/env python3
"""Serve NV-Embed-v2 behind a minimal OpenAI-compatible embeddings API.

This server is intentionally narrow:
- local-files-only load from an already cached NV-Embed-v2 snapshot
- single-process / single-GPU service
- OpenAI-compatible `GET /v1/models` and `POST /v1/embeddings`

Why the tokenizer runtime dir exists:
- NV-Embed-v2 remote code internally calls `AutoTokenizer.from_pretrained`
  against `config.text_config._name_or_path`.
- In an offline local-cache setup, the top-level model config points back to
  the composite NV-Embed config instead of a plain Mistral tokenizer config.
- We materialize a tiny runtime tokenizer directory with tokenizer files plus a
  patched `config.json` so the model can load fully offline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_CACHE_ROOT = Path("/mnt/nvme/hf/models--nvidia--NV-Embed-v2")
DEFAULT_RUNTIME_ROOT = Path("/tmp/nv_embed_v2_runtime")
DEFAULT_SERVED_MODEL_NAME = "nvidia/NV-Embed-v2"
MIB = 1024 * 1024

logger = logging.getLogger("uvicorn.error")

REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "configuration_nvembed.py",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

TOKENIZER_RUNTIME_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


class EmbeddingsRequest(BaseModel):
    input: str | List[str] = Field(..., description="One string or a list of strings.")
    model: str | None = Field(default=None)
    encoding_format: str | None = Field(default="float")
    user: str | None = Field(default=None)


def _resolve_snapshot_dir(explicit_snapshot: str | None) -> Path:
    if explicit_snapshot:
        snapshot_dir = Path(explicit_snapshot).expanduser().resolve()
    else:
        snapshots_root = DEFAULT_CACHE_ROOT / "snapshots"
        if not snapshots_root.exists():
            raise FileNotFoundError(f"Missing snapshots root: {snapshots_root}")
        candidates = sorted([path for path in snapshots_root.iterdir() if path.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"No NV-Embed-v2 snapshots found under {snapshots_root}")
        snapshot_dir = candidates[-1].resolve()

    missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (snapshot_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Snapshot is incomplete: {snapshot_dir}. Missing files: {', '.join(missing)}"
        )
    return snapshot_dir


def _materialize_tokenizer_runtime_dir(snapshot_dir: Path, runtime_root: Path) -> Path:
    runtime_dir = runtime_root / "tokenizer"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for name in TOKENIZER_RUNTIME_FILES:
        src = snapshot_dir / name
        dst = runtime_dir / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            dst.unlink()
        dst.symlink_to(src)

    composite_config = json.loads((snapshot_dir / "config.json").read_text(encoding="utf-8"))
    text_config = dict(composite_config.get("text_config") or {})
    if not text_config:
        raise RuntimeError(f"Missing text_config in {snapshot_dir / 'config.json'}")

    # Force tokenizer loading through a standard Mistral tokenizer config rather
    # than the composite NV-Embed model config.
    text_config["model_type"] = "mistral"
    text_config["_name_or_path"] = str(runtime_dir)
    (runtime_dir / "config.json").write_text(
        json.dumps(text_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return runtime_dir


def _normalize_texts(raw_input: str | Sequence[str]) -> List[str]:
    if isinstance(raw_input, str):
        texts = [raw_input]
    else:
        texts = [str(text) for text in raw_input]
    normalized: List[str] = []
    for text in texts:
        text = text.replace("\n", " ").strip()
        normalized.append(text if text else " ")
    return normalized


def _chunked(values: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _count_prompt_tokens(tokenizer: Any, texts: Sequence[str]) -> int:
    try:
        encoded = tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = encoded.get("input_ids") or []
        return int(sum(len(ids) for ids in input_ids))
    except Exception:
        return 0


def _bytes_to_mib(num_bytes: int) -> int:
    return int(num_bytes // MIB)


def _get_cuda_memory_stats(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "cuda_device_index": None,
            "free_mib": None,
            "total_mib": None,
            "allocated_mib": None,
            "reserved_mib": None,
            "max_allocated_mib": None,
            "max_reserved_mib": None,
        }

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {
        "cuda_device_index": int(device_index),
        "free_mib": _bytes_to_mib(free_bytes),
        "total_mib": _bytes_to_mib(total_bytes),
        "allocated_mib": _bytes_to_mib(torch.cuda.memory_allocated(device_index)),
        "reserved_mib": _bytes_to_mib(torch.cuda.memory_reserved(device_index)),
        "max_allocated_mib": _bytes_to_mib(torch.cuda.max_memory_allocated(device_index)),
        "max_reserved_mib": _bytes_to_mib(torch.cuda.max_memory_reserved(device_index)),
    }


def _format_cuda_memory_stats(device: torch.device) -> str:
    stats = _get_cuda_memory_stats(device)
    if stats["cuda_device_index"] is None:
        return "device=cpu"
    return (
        f"cuda:{stats['cuda_device_index']} "
        f"free={stats['free_mib']}MiB "
        f"allocated={stats['allocated_mib']}MiB "
        f"reserved={stats['reserved_mib']}MiB "
        f"peak_allocated={stats['max_allocated_mib']}MiB "
        f"peak_reserved={stats['max_reserved_mib']}MiB "
        f"total={stats['total_mib']}MiB"
    )


def _build_warmup_prompt(max_length: int) -> str:
    token_budget = max(int(max_length), 1)
    # Repeating short words reliably produces a prompt longer than max_length
    # tokens so NV-Embed hits its real truncation and workspace path.
    return "warmup " * (token_budget * 3)


def _run_startup_warmup(
    model: Any,
    device: torch.device,
    instruction: str,
    max_length: int,
    warmup_batch_size: int,
) -> None:
    if device.type != "cuda" or warmup_batch_size <= 0:
        return

    prompts = [_build_warmup_prompt(max_length) for _ in range(warmup_batch_size)]
    logger.info(
        "Running NV-Embed startup warmup batch_size=%d max_length=%d",
        warmup_batch_size,
        max_length,
    )
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        warmup_embeddings = model.encode(
            prompts=prompts,
            instruction=instruction,
            max_length=max_length,
        )
        if isinstance(warmup_embeddings, torch.Tensor):
            _ = warmup_embeddings.shape
        else:
            _ = np.asarray(warmup_embeddings).shape
    torch.cuda.synchronize(device)
    logger.info("Startup warmup complete: %s", _format_cuda_memory_stats(device))


def _prime_cuda_cache(device: torch.device, reserve_mib: int) -> None:
    if device.type != "cuda" or reserve_mib <= 0:
        return

    chunk_mib = 512
    chunks: List[torch.Tensor] = []
    remaining_mib = int(reserve_mib)
    logger.info("Priming CUDA allocator reserve=%dMiB", reserve_mib)
    while remaining_mib > 0:
        current_mib = min(remaining_mib, chunk_mib)
        chunks.append(
            torch.empty(current_mib * MIB, dtype=torch.uint8, device=device)
        )
        remaining_mib -= current_mib
    del chunks
    torch.cuda.synchronize(device)
    logger.info("CUDA allocator primed: %s", _format_cuda_memory_stats(device))


def build_app(args: argparse.Namespace) -> FastAPI:
    snapshot_dir = _resolve_snapshot_dir(args.model_snapshot)
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    tokenizer_runtime_dir = _materialize_tokenizer_runtime_dir(snapshot_dir, runtime_root)
    dtype = _resolve_torch_dtype(args.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_runtime_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    config.text_config._name_or_path = str(tokenizer_runtime_dir)
    model = AutoModel.from_pretrained(
        str(snapshot_dir),
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True,
    ).eval().to(device)

    warmup_batch_size = max(0, min(int(args.startup_warmup_batch_size), int(args.max_batch_size)))
    _run_startup_warmup(
        model=model,
        device=device,
        instruction=str(args.instruction or ""),
        max_length=int(args.max_length),
        warmup_batch_size=warmup_batch_size,
    )
    _prime_cuda_cache(device=device, reserve_mib=int(args.cuda_reserve_mib))
    logger.info("NV-Embed service ready with memory profile: %s", _format_cuda_memory_stats(device))

    model_lock = threading.Lock()
    started_at = int(time.time())
    service_state: dict[str, Any] = {
        "startup_warmup_batch_size": warmup_batch_size,
        "cuda_reserve_mib": int(args.cuda_reserve_mib),
        "oom_count": 0,
        "last_oom": None,
    }

    app = FastAPI(title="NV-Embed-v2 Local Embeddings Server")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        payload = {
            "status": "ok",
            "model": args.served_model_name,
            "snapshot_dir": str(snapshot_dir),
            "device": str(device),
            "dtype": str(dtype),
            "max_length": int(args.max_length),
            "max_batch_size": int(args.max_batch_size),
            "startup_warmup_batch_size": int(service_state["startup_warmup_batch_size"]),
            "cuda_reserve_mib": int(service_state["cuda_reserve_mib"]),
            "oom_count": int(service_state["oom_count"]),
            "last_oom": service_state["last_oom"],
        }
        if device.type == "cuda":
            payload["cuda"] = _get_cuda_memory_stats(device)
        return payload

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": started_at,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/embeddings")
    def create_embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
        if request.model and request.model != args.served_model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model={request.model!r}; served model is {args.served_model_name!r}",
            )

        texts = _normalize_texts(request.input)
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")

        embeddings: List[List[float]] = []
        try:
            with model_lock, torch.inference_mode():
                for batch in _chunked(texts, int(args.max_batch_size)):
                    batch_embeddings = model.encode(
                        prompts=list(batch),
                        instruction=str(args.instruction or ""),
                        max_length=int(args.max_length),
                    )
                    if isinstance(batch_embeddings, torch.Tensor):
                        batch_embeddings = batch_embeddings.detach().cpu().numpy()
                    batch_embeddings = np.asarray(batch_embeddings)
                    embeddings.extend(batch_embeddings.astype(np.float32).tolist())
        except torch.OutOfMemoryError as exc:
            service_state["oom_count"] = int(service_state["oom_count"]) + 1
            service_state["last_oom"] = {
                "ts": int(time.time()),
                "batch_size": min(len(texts), int(args.max_batch_size)),
                "num_inputs": len(texts),
                "max_length": int(args.max_length),
                "cuda": _get_cuda_memory_stats(device) if device.type == "cuda" else None,
            }
            logger.error(
                "NV-Embed OOM while handling num_inputs=%d max_batch_size=%d max_length=%d: %s | %s",
                len(texts),
                int(args.max_batch_size),
                int(args.max_length),
                exc,
                _format_cuda_memory_stats(device),
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "NV-Embed CUDA OOM. The service GPU lost required headroom. "
                    "Reduce GPU contention or restart the service on a less contended GPU."
                ),
            ) from exc

        prompt_tokens = _count_prompt_tokens(tokenizer, texts)
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": index,
                }
                for index, embedding in enumerate(embeddings)
            ],
            "model": args.served_model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve NV-Embed-v2 on a local GPU.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--model-snapshot", default="")
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--startup-warmup-batch-size", type=int, default=0)
    parser.add_argument("--cuda-reserve-mib", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
