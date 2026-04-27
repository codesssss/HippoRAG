# DAEC-ALR Day-0 Reader Consistency Probe - 2026-04-27

## Purpose

This note records the first, deliberately narrow test for the proposed `DAEC-Inner-Loop Active Retrieval` direction.

The tested question was:

> Does reader answer self-consistency under DAEC evidence perturbations separate correct from wrong DAEC contexts?

This is only a signal probe. It does not implement active retrieval, single-edit repair, or reader-gated admission yet.

## Artifacts

Implementation:
- `scripts/run_daec_consistency_probe.py`
- `tests/dpathrag/test_daec_consistency_probe.py`

Primary report:
- `reports/daec_alr/consistency_probe.md`
- `reports/daec_alr/consistency_probe.json`
- `reports/daec_alr/consistency_probe.rows.jsonl`
- `reports/daec_alr/consistency_probe.predictions.jsonl`

Additional controls:
- `reports/daec_alr/consistency_probe_hf_flan.md`
- `reports/daec_alr/consistency_probe_hf_flan.json`
- `reports/daec_alr/consistency_probe_qwen_simple_prompt.md`
- `reports/daec_alr/consistency_probe_qwen_simple_prompt.json`

Validation:

```text
.venv-hipporag/bin/python -m pytest tests/dpathrag/test_daec_consistency_probe.py
4 passed, 2 warnings
```

## Protocol

Input:
- DAEC report: `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- pool JSON: `run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json`
- rows: first `200` queries
- evidence budget: DAEC selected top-5 from `selector_trace.selected_pool_positions`

Perturbation variants:
- `original`
- `swap01`
- `reverse`
- `rotate_left`
- `drop_last`

Reader:
- primary: local OpenAI-compatible `qwen3-8b-train` at `http://localhost:8043/v1`
- prompt: HippoRAG `rag_qa_musique` one-shot QA template
- generations: `200 x 5 = 1000`
- concurrency: `8`

Correctness label:
- primary label: original prediction `F1 >= 0.5`
- secondary labels: original EM, original F1 positive

Consistency features:
- `majority_fraction`
- `inverse_entropy`
- `inverse_distinct`
- `mean_pairwise_f1`

## Primary Result: Qwen Reader With Matched HippoRAG QA Prompt

Base reader result on the reconstructed DAEC top-5:

| Metric | Value |
|---|---:|
| rows | 200 |
| original EM | 0.4900 |
| original F1 | 0.5882 |
| support recall | 0.9413 |
| support complete | 0.8550 |

Consistency means:

| Metric | All | Correct F1>=0.5 | Wrong F1<0.5 |
|---|---:|---:|---:|
| majority_fraction | 0.8030 | 0.8992 | 0.6617 |
| inverse_entropy | 0.7377 | 0.8649 | 0.5507 |
| inverse_distinct | 0.7800 | 0.8950 | 0.6111 |
| mean_pairwise_f1 | 0.7766 | 0.8658 | 0.6456 |
| normalized_entropy | 0.2623 | 0.1351 | 0.4493 |
| distinct_answer_count | 1.8800 | 1.4202 | 2.5556 |

AUC against original `F1 >= 0.5`:

| Consistency Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.7572 | [0.6875, 0.8202] |
| inverse_entropy | 0.7598 | [0.6902, 0.8226] |
| inverse_distinct | 0.7599 | [0.6898, 0.8218] |
| mean_pairwise_f1 | 0.7317 | [0.6601, 0.7983] |

AUC against original EM:

| Consistency Metric | AUC | 95% CI |
|---|---:|---:|
| majority_fraction | 0.7683 | [0.7056, 0.8232] |
| inverse_entropy | 0.7700 | [0.7063, 0.8264] |
| inverse_distinct | 0.7648 | [0.7028, 0.8235] |
| mean_pairwise_f1 | 0.7447 | [0.6778, 0.8045] |

Decision:

```text
PASS_SIGNAL_PROBE
```

The signal is strong enough to justify a bounded Step-1 single-edit probe.

## Controls

### Qwen Reader With Simpler QA Prompt

This was run before switching to the HippoRAG one-shot QA template.

Result:
- original EM/F1: `0.4050 / 0.5166`
- majority_fraction AUC vs F1>=0.5: `0.8107`, CI `[0.7501, 0.8667]`
- inverse_entropy AUC vs F1>=0.5: `0.8153`, CI `[0.7555, 0.8716]`

Interpretation:
- Even with a weaker simplified prompt, Qwen self-consistency was highly discriminative.
- The matched prompt is the primary result because it is closer to the DAEC reader protocol.

### Fine-Tuned Flan-T5 Reader

Result:
- original EM/F1: `0.4500 / 0.5129`
- majority_fraction AUC vs F1>=0.5: `0.5666`, CI `[0.4986, 0.6315]`
- inverse_entropy AUC vs F1>=0.5: `0.5658`, CI `[0.4982, 0.6317]`

Interpretation:
- The consistency signal is reader-dependent.
- The local fine-tuned Flan-T5 reader is much more stable across perturbations for both correct and wrong answers, so consistency does not separate well.
- For DAEC-ALR, Qwen is the relevant reader because DAEC's main positive results were generated with Qwen-style readers.

## Interpretation

This is the first recent residual-route probe that produces a genuinely positive new signal:

```text
Qwen reader answer stability under DAEC evidence perturbation separates correct from wrong original contexts.
```

This differs from previous failed routes:

- C-CEE used reader likelihood under answer uncertainty; this probe uses answer agreement across evidence perturbations.
- D-PathRAG edited freely and imported hard negatives; this probe does not edit yet.
- CAPS scored candidate proofs locally; this probe asks whether the final reader answer is stable under evidence changes.
- CPAG/RRF rewarded cross-pool agreement; this probe operates after DAEC evidence composition and measures reader behavior.

The positive signal should not be overclaimed:

- consistency is not correctness;
- consistent wrong answers still exist;
- the Flan-T5 control shows the signal is not universal across readers;
- the probe does not yet prove that an edit loop improves F1.

## Next Step

Proceed to `DAEC-ALR Step-1: consistency-gated single-edit probe`, but keep it tightly bounded.

Required admission rule:

```text
accept an edit only if:
1. original consistency is below a threshold;
2. the replacement improves DAEC demand/binding coverage for the weakest selected demand;
3. already satisfied high-confidence demands are not reduced;
4. reader consistency improves by a margin after replacement;
5. at most one edit is accepted per query.
```

Do not jump directly to active retrieval. First test single-edit repair inside the existing PropRAG top100 pool.

Suggested Step-1 metrics:
- edit trigger rate;
- accepted edit rate;
- support_complete delta;
- reader F1 delta;
- correct-to-wrong flips vs wrong-to-correct flips;
- added non-gold / added gold ratio;
- consistency improvement among accepted edits.

Stop Step-1 if:
- accepted edits mostly increase consistency for wrong answers;
- support_complete drops below DAEC;
- F1 drops;
- non-gold/gold import ratio begins to resemble D-PathRAG.
