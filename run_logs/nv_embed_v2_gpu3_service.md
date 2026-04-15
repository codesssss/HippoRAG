# NV-Embed-v2 GPU3 Service

## Purpose

This repo now includes a local OpenAI-compatible embeddings service for the
original `nvidia/NV-Embed-v2` model on GPU 3.

Files:

- `scripts/serve_nv_embed_v2.py`
- `run_logs/launch_nv_embed_v2_gpu3.sh`

Default endpoint:

- `http://localhost:8019/v1`

Default model id:

- `nvidia/NV-Embed-v2`

## Why this service is custom

`NV-Embed-v2` is loaded via `transformers` remote code rather than vLLM.
In offline local-cache mode, the model can fail to initialize because the
top-level composite config points tokenizer loading back at the composite model
directory. The service fixes this by:

1. loading from the complete local snapshot under:
   `/mnt/nvme/hf/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef`
2. materializing a tiny runtime tokenizer directory under:
   `/tmp/nv_embed_v2_runtime/tokenizer`
3. patching `config.text_config._name_or_path` to that tokenizer directory

This keeps the service fully offline and reproducible.

## Start

```bash
bash run_logs/launch_nv_embed_v2_gpu3.sh
```

## Stop

```bash
kill "$(cat run_logs/launch_nv_embed_v2_gpu3.pid)"
```

If you want a clean restart:

```bash
kill "$(cat run_logs/launch_nv_embed_v2_gpu3.pid)" || true
sleep 2
bash run_logs/launch_nv_embed_v2_gpu3.sh
```

## Health Checks

```bash
curl -s http://127.0.0.1:8019/healthz
curl -s http://127.0.0.1:8019/v1/models
curl -s http://127.0.0.1:8019/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/NV-Embed-v2","input":["hello world","second example"]}'
```

## How to Use from HippoRAG

When you want the OpenAI-compatible embedding client path:

```bash
--embedding_name nvidia/NV-Embed-v2 \
--embedding_base_url http://localhost:8019/v1
```

Do not use the VLLM-style `.../v1/embeddings` URL with this service.

## Adjustable Knobs

The launcher exposes these env vars:

- `CUDA_DEVICE`
  Default: `3`
- `PORT`
  Default: `8019`
- `MODEL_SNAPSHOT`
  Default: `/mnt/nvme/hf/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef`
- `RUNTIME_ROOT`
  Default: `/tmp/nv_embed_v2_runtime`
- `MAX_LENGTH`
  Default: `2048`
- `MAX_BATCH_SIZE`
  Default: `4`
- `TORCH_DTYPE`
  Default: `float16`
- `SERVED_MODEL_NAME`
  Default: `nvidia/NV-Embed-v2`
- `STARTUP_WARMUP_BATCH_SIZE`
  Default: `4`
- `CUDA_RESERVE_MIB`
  Default: `8192`
- `PYTORCH_CUDA_ALLOC_CONF_VALUE`
  Default: `expandable_segments:True`

Example:

```bash
PORT=8021 CUDA_DEVICE=3 MAX_BATCH_SIZE=8 MAX_LENGTH=1024 CUDA_RESERVE_MIB=4096 \
  bash run_logs/launch_nv_embed_v2_gpu3.sh
```

## Runtime Files

- log: `run_logs/launch_nv_embed_v2_gpu3.log`
- status: `run_logs/launch_nv_embed_v2_gpu3.status`
- pid: `run_logs/launch_nv_embed_v2_gpu3.pid`

## Operational Notes

- The service is single-process and single-GPU.
- It uses `torch.inference_mode()` and a single in-process model lock.
- At startup it now runs a synthetic warmup request and primes the CUDA
  allocator with an additional reserve so later requests can often reuse this
  process-local memory instead of competing for fresh device allocations.
- This improves resilience on a shared GPU, but it is not a hard isolation
  guarantee. True guarantees still require a dedicated GPU, MIG, or scheduler
  isolation.
- Returned vectors are `float32` JSON arrays.
- The model itself is loaded in `float16` by default.
- On this machine, the live GPU-3 process settled at roughly **16.3 GiB**
  (`pid=3142571`, bus `00000000:44:00.0`) after startup.
