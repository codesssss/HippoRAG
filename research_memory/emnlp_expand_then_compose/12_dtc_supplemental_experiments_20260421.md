# DtC Supplemental Experiments - 2026-04-21

## Purpose

This run supplements the DtC-Embed method evidence for the fixed-pool evidence composition story.

The current main-method candidate is DtC-Embed v1:

- Fixed expanded pool: `pool_k=100`
- Reader budget: `qa_top_k=5`
- One-shot LLM decomposition: `dtc_decomposition_mode=llm`
- Dependency constraints enabled: `dtc_enforce_dependencies=true`
- Dependency binding disabled by default: `dtc_enable_dependency_binding=false`
- Reserve baseline prefix through `setwise_anchor_count=2`

The typed binding version remains an ablation/failure-analysis variant because pilot100 improved 2Wiki but weakened Hotpot and MuSiQue relative to v1.

## Code Additions

Added ablation controls without changing default DtC v1 behavior:

- `--dtc_decomposition_mode {llm,query}`
  - `llm`: default DtC decomposition.
  - `query`: query-only ablation, using one fallback requirement equal to the original question.
- `--dtc_enforce_dependencies {true,false}`
  - `true`: default, respects LLM-declared dependency order.
  - `false`: strips `depends_on` before coverage selection.

Validation:

- `python -m py_compile scripts/dtc_embed_utils.py scripts/eval_causal_qwen3.py`
- `.venv-hipporag/bin/python -m pytest -q tests/test_dtc_embed_utils.py`
- Result: `4 passed`.

## Running Queues

### Queue A: DtC v1 limit1000

Script:

- `run_logs/run_dtc_v1_limit1000_20260421.sh`

Log:

- `run_logs/dtc_v1_limit1000_20260421.log`

PID file:

- `run_logs/dtc_v1_limit1000_20260421.pid`

Protocol:

- Datasets: `2wikimultihopqa`, `hotpotqa`, `musique`
- Limit: `1000`
- Reader: `qwen3-8b-train` at `http://localhost:8043/v1`
- Output paths:
  - `outputs_step0_general_2wikimultihopqa/eval_reports/dtc_embed_limit1000_v1_anchor2_fresh8043.json`
  - `outputs_step0_general_hotpotqa/eval_reports/dtc_embed_limit1000_v1_anchor2_fresh8043.json`
  - `outputs_step0_general_musique/eval_reports/dtc_embed_limit1000_v1_anchor2_fresh8043.json`

### Queue B: DtC pilot100 ablations

Script:

- `run_logs/run_dtc_ablation_pilot100_20260421.sh`

Log:

- `run_logs/dtc_ablation_pilot100_20260421.log`

PID file:

- `run_logs/dtc_ablation_pilot100_20260421.pid`

Protocol:

- Datasets: `2wikimultihopqa`, `hotpotqa`, `musique`
- Limit: `100`
- Reader: `qwen3-8b-train` at `http://localhost:8042/v1`
- Variants:
  - `nodecomp`: `--dtc_decomposition_mode query`
  - `nodep`: `--dtc_enforce_dependencies false`
  - `nored`: `--dtc_redundancy_weight 0.0`
  - `noreserve`: `--setwise_anchor_count 0 --setwise_reserve_top_m 0`

Expected output pattern:

- `outputs_step0_general_<dataset>/eval_reports/dtc_embed_ablate_<variant>_pilot100_anchor2_fresh8042.json`

Note: the `noreserve` filename still includes `anchor2` due to the shared naming template; the JSON config records the actual `setwise_anchor_count=0`.

## First Completed Result

The first ablation, `2Wiki/nodecomp`, completed:

- File: `outputs_step0_general_2wikimultihopqa/eval_reports/dtc_embed_ablate_nodecomp_pilot100_anchor2_fresh8042.json`
- Baseline EM/F1: `0.3600 / 0.4008`
- Selector EM/F1: `0.3600 / 0.4008`
- Delta EM/F1: `0.0000 / 0.0000`

Interpretation:

- Query-only coverage does not reproduce DtC v1's earlier 2Wiki pilot100 gain.
- This supports the claim that LLM decomposition is not cosmetic; explicit sub-demand coverage is doing useful work.

## Monitoring Commands

```bash
ps -p "$(cat run_logs/dtc_v1_limit1000_20260421.pid)" -o pid,ppid,sid,stat,etime,cmd
ps -p "$(cat run_logs/dtc_ablation_pilot100_20260421.pid)" -o pid,ppid,sid,stat,etime,cmd
tail -f run_logs/dtc_v1_limit1000_20260421.log
tail -f run_logs/dtc_ablation_pilot100_20260421.log
```

Quick summary once files finish:

```bash
.venv-hipporag/bin/python - <<'PY'
import json
from pathlib import Path

for root in [
    Path("outputs_step0_general_2wikimultihopqa/eval_reports"),
    Path("outputs_step0_general_hotpotqa/eval_reports"),
    Path("outputs_step0_general_musique/eval_reports"),
]:
    for p in sorted(root.glob("dtc_embed*_fresh804*.json")):
        r = json.loads(p.read_text())
        qa = r.get("setwise_selector_qa", {})
        if not qa:
            continue
        print(
            p,
            "base", qa.get("baseline_EM"), qa.get("baseline_F1"),
            "sel", qa.get("selector_EM"), qa.get("selector_F1"),
            "delta", qa.get("EM_delta"), qa.get("F1_delta"),
        )
PY
```
