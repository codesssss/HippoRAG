# Week 0 Decision Report

## Split audit
- Variant: `MuSiQue-Ans-local-subset`
- Dev/local size: `1000`
- Current MuSiQue-1000 source: `reproduce/dataset/musique.json`
- Recommended evaluation split: `reproduce/dataset/musique.json`

## IRCoT baseline
- Status: `missing`
- MuSiQue EM/F1: `` / ``
- Latency: ``
- LLM calls/query: ``

## NLI calibration
- Status: `model_unavailable`
- ECE: `1.0`
- Brier: `1.0`
- Absorbing status: `ablation`

## Oracle slot pipeline
- Status: `completed`
- Built examples: `1000`
- Issues: `Oracle slots are only for BSGS-oracle-slot diagnostics; do not tune latent-slot prompts on the same split.`

## Qwen slot quality
- Status: `completed`
- Slot recall: `0.6768639251086587`
- Slot precision: `0.6551248985050385`
- Variable grounding accuracy: `0.0019224339685723836`

## Qwen slot quality post-hoc
- Status: `completed`
- Position-chain accuracy: `0.5575058508859912`
- Dependency presence accuracy: `0.6908224674022063`
- Slot count within one: `0.9428284854563691`

## Decision
- Week 1 config: `{'dataset': 'musique', 'slot_mode': 'oracle-slot', 'binding': 'hard normalized string / alias match', 'selector': 'posterior top-k'}`
- Components enabled: `MuSiQue answerable-subset diagnostics, oracle-slot Week-1 diagnostic`
- Components disabled: `latent-slot main path, soft binding, DPP, LLM verifier, semi-absorbing transition as main`

## Risks
- NLI calibration did not pass or was unavailable; absorbing remains ablation/fallback.
- Qwen slot recall is 0.6769, below the 0.70 Week-2 main-route gate; keep latent slots diagnostic-only.
- Post-hoc position-chain accuracy is 0.5575; Qwen often misses explicit dependency wiring even when slot text matches.
