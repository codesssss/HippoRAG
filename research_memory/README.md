# Research Memory

This directory is the canonical memory for ongoing research directions in this repo.

Use it to prevent idea drift across session compaction, agent handoff, and long-running experiment cycles.

Rules:
- Treat these files as the source of truth for research framing.
- Update `01_claim_ledger.md` when new evidence changes the support status of a claim.
- Update `02_decision_log.md` whenever the main framing changes.
- Update `03_experiment_board.md` when tasks move between done, running, blocked, and next.
- Update `04_result_registry.md` only with concrete results that have a file path.

Current active direction:
- `emnlp_expand_then_compose/`

## Runtime Notes For Future Agents

If you touch the current `coverage` or `width-matched CE control` line, read this note before launching new runs:
- `emnlp_expand_then_compose/10_coverage_probe_and_control_status_20260409.md`
- `emnlp_expand_then_compose/11_nq_popqa_bridge_threshold_diagnosis_20260410.md`

Operational rules for the current full-scale control family:
- Ignore the old `20260409` pseudo-full outputs. They accidentally inherited `--limit 20`.
- Use `20260409fullfix` outputs as the first valid full-scale control runs.
- For repeated controls on the same dataset and `qa_top_k`, prefer reuse:
  - baseline QA reuse: `--baseline_report_json`
  - retrieval reuse: `--retrieval_cache_json`
  - fresh baseline cache write: `--save_retrieval_cache_json`

Control semantics to keep straight:
- `baseline_top5_plus_ce` is baseline-only CE rerank, not the same-pool append control.
- `bridge_append_plus_ce` is the same-pool pure CE rerank control: top-10 baseline prefix plus 3 bridge-appended docs, then CE sort only.
- `random3_deep_plus_ce` is the matched random-append CE control.
- `baseline_top10_plus_ce` is the width-matched no-append CE control.
- For this family, the `100`-query and `20260409fullfix` full-scale runs keep the same method definition and differ only in evaluation size.

The current reusable runners are:
- `run_logs/fullscale_width_matched_control_20260409.sh`
- `run_logs/fullscale_width_matched_control_musique_top5_priority_20260409.sh`
- `run_logs/fullscale_width_matched_control_musique_top5_remaining_20260409.sh`

External LLM endpoint note:
- Public Qwen3 mirrors verified from outside the host on 2026-04-10:
  - `http://36.133.236.142:8002/v1`
  - `http://36.133.236.142:8003/v1`
- Both return `qwen3-8b` from `/v1/models` and produce valid `/v1/chat/completions` responses.
- Current generation format still includes a lightweight `<think>...</think>` wrapper before the visible answer text, so future agents should not assume raw plain-text completions.

Dataset naming note:
- `nq` in this repo resolves to the packaged `nq_rear` files already stored under `reproduce/dataset/`.
- `popqa` and `nq_rear` here are packaged `1000`-query evaluation subsets shipped with the HippoRAG data bundle, not the raw full upstream benchmarks.
