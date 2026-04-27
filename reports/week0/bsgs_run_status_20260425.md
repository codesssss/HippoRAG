# BSGS Run Status 2026-04-25

## Running Jobs

| Job | Session / PID | Status | Output |
|---|---:|---|---|
| IRCoT MuSiQue-1000 | `78249` / `144518` | running; first retrieval round observed at `93/1000`, about `10-11s/example`; many rerank parse retries but process continues | `reports/week0/ircot_musique1000.json`, `reports/week0/ircot_baseline.md` |

## Completed Jobs

| Job | Status | Output | Key result |
|---|---|---|---|
| IRCoT smoke/cache warmup | completed | `/tmp/bsgs_ircot_smoke.json`, `/tmp/bsgs_ircot_smoke.md` | cache warmup succeeded; full1000 restarted with elevated localhost access |
| Qwen slot quality MuSiQue-1000 | completed | `reports/week0/qwen_slot_quality.json`, `data/processed/musique_qwen_slots.jsonl` | slot recall `0.6769`; exact variable grounding `0.0019` |
| Qwen slot post-hoc diagnostics | completed | `reports/week0/qwen_slot_quality_posthoc.json`, `reports/week0/qwen_slot_quality_posthoc.md` | slot count within one `0.9428`; position-chain accuracy `0.5575` |
| BSGS oracle-slot MuSiQue-1000 | completed | `reports/week1/operator_validation.json`, `reports/week1/operator_validation.md` | supporting paragraph recall `0.2268`; avg entropy `2.4719` |
| BSGS mechanism analysis | completed | `reports/week1/operator_mechanism_analysis.json`, `reports/week1/operator_mechanism_analysis.md` | full support covered `54/1000`; partial `400/1000`; none `546/1000` |
| Week0/Week1 summaries | refreshed | `reports/week0/decision.json`, `reports/week0/decision.md`, `reports/week1/operator_validation_summary.json`, `reports/week1/operator_validation.md` | IRCoT remains missing/running until full1000 output is written |

## Current Interpretation

- Avoid launching new jobs that use `8041` or `8019`; IRCoT full1000 is currently the critical path.
- Qwen latent-slot route should remain diagnostic-only for now: slot text is near the threshold, but dependency wiring is weak.
- BSGS oracle operator is not yet strong enough as a main route: over half the examples cover no supporting paragraph under the current minimal operator.
