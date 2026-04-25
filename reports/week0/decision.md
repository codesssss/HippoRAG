# Week 0 Decision Report

## Split audit
- Variant: `MuSiQue-Ans-local-subset`
- Dev/local size: `1000`
- Current MuSiQue-1000 source: `reproduce/dataset/musique.json`
- Recommended evaluation split: `reproduce/dataset/musique.json`

## IRCoT baseline
- Status: `not_run`
- MuSiQue EM/F1: `` / ``
- Latency: ``
- LLM calls/query: ``

## NLI calibration
- Status: `completed`
- ECE: `0.08528021153070182`
- Brier: `0.057685430095094414`
- Absorbing status: `ablation`

## Oracle slot pipeline
- Status: `completed`
- Built examples: `1000`
- Issues: `Oracle slots are only for BSGS-oracle-slot diagnostics; do not tune latent-slot prompts on the same split.`

## Qwen slot quality
- Status: `not_run`
- Slot recall: ``
- Slot precision: ``
- Variable grounding accuracy: ``

## Decision
- Week 1 config: `{'dataset': 'musique', 'slot_mode': 'oracle-slot', 'binding': 'hard normalized string / alias match', 'selector': 'posterior top-k'}`
- Components enabled: `MuSiQue answerable-subset diagnostics, oracle-slot Week-1 diagnostic`
- Components disabled: `latent-slot main path, soft binding, DPP, LLM verifier, semi-absorbing transition as main`

## Risks
- Formal DeBERTa calibration has not run; lexical smoke is not a main verifier result, so absorbing remains ablation/fallback.
