# Diversity Selector Baselines for Fixed-Pool Evidence Composition

Date: 2026-04-21

## Purpose

This note records the first full-scale comparison between fixed top-k truncation and structure-blind setwise diversity selectors for the `expand + assemble` line.

The question was:

> If the expanded pool already contains useful evidence, can generic set-level diversity selection close the reader-utility gap, or do we need task / graph / bridge-aware evidence composition?

This is the missing baseline family for the current paper story. Prior analysis showed large oracle headroom under a fixed expanded pool, but without MMR / DPP baselines we could not distinguish "set-level diversity is enough" from "multi-hop composition needs structure-aware selection."

## Protocol

All runs used frozen pools from the existing oracle reports and re-ran the reader for both baseline and selectors under the same reader protocol.

Important protocol details:

- Dataset size: 1000 queries for each dataset.
- Pool: fixed `pool_k=100` from the corresponding oracle report.
- Reader context size: `qa_top_k=5`.
- Baseline: original top-5 from the frozen pool, freshly passed through `rag_qa`.
- Selectors: `MMR` and greedy `DPP`, both selecting 5 documents from the same top-100 pool.
- Ordering: `original_rank`, so diversity baselines do not get extra chain-ordering help.
- Embeddings: existing Qwen3 passage embeddings from the `qwen3-8b` workdir.
- Reader endpoint: `qwen3-8b-train` served at `http://localhost:8043/v1`.
- Working directory model name: `--llm_name qwen3-8b`.
- Request model name: `--llm_request_name qwen3-8b-train`.

The `llm_name` / `llm_request_name` split matters. Using `--llm_name qwen3-8b-train` changes the HippoRAG working directory and can fall back to a different retrieval / graph state. The valid protocol is to keep the existing `qwen3-8b` graph/cache workdir and only route live reader calls to the `qwen3-8b-train` service.

## Implementation Notes

The experiment code is in:

- `scripts/diversity_selector_study.py`
- `tests/test_diversity_selector_study.py`

The script supports checkpointing and resume via `--resume_from_output`. This was needed for the MuSiQue DPP run.

The first implementation compared frozen baseline answers against live selector answers, which was not a fair comparison. This was fixed: baseline, MMR, and DPP are now all evaluated with fresh `rag_qa` calls under the same reader endpoint.

MMR uses a rank-prior relevance score and Qwen3 embedding cosine similarity as the diversity penalty.

DPP uses a rank-prior quality term and a cosine-similarity kernel, selected greedily by log-determinant gain.

## Output Files

- `outputs_step0_general_2wikimultihopqa/eval_reports/diversity_selector_study_select100_fresh8043.json`
- `outputs_step0_general_2wikimultihopqa/eval_reports/diversity_selector_study_select100_fresh8043.md`
- `outputs_step0_general_hotpotqa/eval_reports/diversity_selector_study_select100_fresh8043.json`
- `outputs_step0_general_hotpotqa/eval_reports/diversity_selector_study_select100_fresh8043.md`
- `outputs_step0_general_musique/eval_reports/diversity_selector_study_select100_fresh8043.json`
- `outputs_step0_general_musique/eval_reports/diversity_selector_study_select100_fresh8043.md`

## Main Results

| Dataset | Method | EM | F1 | Delta F1 vs. baseline | Recall@5 | Query deltas |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | Baseline top-5 | 0.432 | 0.4868 | - | 0.7865 | - |
| 2Wiki | MMR@100 -> 5 | 0.426 | 0.4805 | -0.0063 | 0.7820 | 7 up / 16 down / 977 same |
| 2Wiki | DPP@100 -> 5 | 0.432 | 0.4863 | -0.0005 | 0.7847 | 9 up / 11 down / 980 same |
| 2Wiki | Oracle@100 | 0.560 | 0.6318 | +0.1450 | - | full support in pool: 0.752 |
| HotpotQA | Baseline top-5 | 0.569 | 0.6839 | - | 0.8835 | - |
| HotpotQA | MMR@100 -> 5 | 0.567 | 0.6830 | -0.0009 | 0.8855 | 12 up / 15 down / 973 same |
| HotpotQA | DPP@100 -> 5 | 0.567 | 0.6829 | -0.0010 | 0.8840 | 12 up / 12 down / 976 same |
| HotpotQA | Oracle@100 | 0.645 | 0.7707 | +0.0868 | - | full support in pool: 0.986 |
| MuSiQue | Baseline top-5 | 0.264 | 0.3439 | - | 0.6177 | - |
| MuSiQue | MMR@100 -> 5 | 0.270 | 0.3545 | +0.0106 | 0.6298 | 30 up / 18 down / 952 same |
| MuSiQue | DPP@100 -> 5 | 0.272 | 0.3569 | +0.0129 | 0.6247 | 30 up / 10 down / 960 same |
| MuSiQue | Oracle@100 | 0.436 | 0.5401 | +0.1962 | - | full support in pool: 0.757 |

## Bucket Results

| Dataset | Bucket | Baseline F1 | MMR F1 | MMR Delta | DPP F1 | DPP Delta |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 2-doc | 0.5436 | 0.5341 | -0.0096 | 0.5405 | -0.0032 |
| 2Wiki | 4-doc | 0.3016 | 0.3062 | +0.0045 | 0.3098 | +0.0081 |
| HotpotQA | 2-doc | 0.6839 | 0.6830 | -0.0009 | 0.6829 | -0.0010 |
| MuSiQue | 2-doc | 0.4125 | 0.4155 | +0.0030 | 0.4168 | +0.0043 |
| MuSiQue | 3-doc | 0.3194 | 0.3414 | +0.0220 | 0.3459 | +0.0265 |
| MuSiQue | 4-doc | 0.1768 | 0.1895 | +0.0126 | 0.1907 | +0.0139 |

## Interpretation

### 1. Structure-blind diversity does not solve the 2Wiki / Hotpot composition gap

2Wiki and Hotpot both have clear oracle headroom under the same fixed pool:

- 2Wiki oracle gap: +0.1450 F1.
- Hotpot oracle gap: +0.0868 F1.

MMR and DPP do not close this gap:

- 2Wiki MMR is negative, and DPP is essentially neutral but slightly negative.
- Hotpot MMR and DPP are both essentially neutral / slightly negative.

This means "set-level interaction" alone is not enough. Generic embedding-space diversity is too weak a proxy for multi-hop evidence utility.

### 2. Diversity helps more on deeper MuSiQue questions, but only modestly

MuSiQue is the only dataset where generic diversity gives a stable positive signal:

- MMR: +0.0106 F1.
- DPP: +0.0129 F1.

The gain is concentrated in deeper buckets:

- 3-doc bucket: DPP +0.0265 F1.
- 4-doc bucket: DPP +0.0139 F1.

This suggests generic diversity can help when the task needs broader evidence coverage, but it captures only a small fraction of the oracle headroom:

- MuSiQue oracle gap: +0.1962 F1.
- DPP closes about 6.6% of that F1 gap.

So diversity is a useful baseline, not a sufficient method.

### 3. The result refines the paper story

The story should not be:

> Any setwise diversity selector fixes fixed-pool composition.

The data says the opposite. A better claim is:

> Fixed-pool evidence composition has real oracle headroom, but structure-blind diversity only helps in deeper multi-hop cases and fails to recover most of the gap. Multi-hop QA needs task- or structure-aware composition rather than generic embedding diversity.

This supports a three-level framing:

1. Pointwise top-k ranking is insufficient.
2. Structure-blind diversity is a limited partial fix.
3. The next method must use query / graph / bridge structure to select reader-useful evidence.

### 4. Current bridge-beam numbers should not be over-compared yet

The diversity reports include old bridge-beam summaries for context, but those bridge reports were not freshly re-run under the exact same `qwen3-8b` workdir plus `qwen3-8b-train@8043` reader protocol.

Therefore:

- It is safe to say MMR / DPP do not beat the same-protocol top-k baseline on 2Wiki / Hotpot and only modestly help MuSiQue.
- It is not yet safe to claim a definitive apples-to-apples ordering of `bridge_beam > MMR/DPP` until bridge-beam is re-run under the same fresh-reader protocol.

Prior bridge-beam reports remain useful as directional evidence, especially on 2Wiki, but they should be treated as prior results rather than final comparison numbers.

## Consequences for Expand + Assemble

This experiment is directly relevant to the `expand + assemble` narrative.

`Expand` is still justified: oracle@100 headroom remains large across all datasets.

`Assemble` cannot be reduced to generic diversity: MMR/DPP do not reliably improve reader utility from the expanded pool.

The method direction should be:

> structure-aware evidence composition under a fixed expanded pool.

But the implementation must be careful. A generic diversity selector is not enough, and an overly aggressive bridge selector can hurt Hotpot / MuSiQue. The method likely needs a conservative gate or task-aware bridge criterion rather than unconditional re-selection.

## Recommended Next Experiments

1. Re-run the best bridge-aware selector under the exact same fresh-reader protocol.

   Use the same frozen pools, `pool_k=100`, `qa_top_k=5`, `order_mode=original_rank` or a separately controlled order mode, `--llm_name qwen3-8b`, and `--llm_request_name qwen3-8b-train --llm_base_url http://localhost:8043/v1`.

   This is required before making a final paper claim that structure-aware composition beats structure-blind diversity.

2. Compare against a very simple "DPP + structure gate" or "MMR + bridge prior" hybrid.

   The MuSiQue result says generic diversity has some real signal. The 2Wiki/Hotpot result says it is not enough. A minimal hybrid can test whether the right method is not pure bridge-beam or pure diversity, but structure-guided diversity.

3. Keep the oracle@100 line as the upper bound, not as a method result.

   The oracle gaps are still the main motivation:

   - 2Wiki: +0.1450 F1.
   - HotpotQA: +0.0868 F1.
   - MuSiQue: +0.1962 F1.

   The method should be judged by how much of this gap it recovers under the same reader protocol.

## Bottom Line

The diversity baseline study does not kill the `expand + assemble` direction. It strengthens the core distinction:

> Expanded pools contain useful evidence, but generic embedding diversity is not enough to assemble it into reader-useful context.

For the paper, the cleanest claim is:

> Fixed-pool composition is a real bottleneck; structure-blind diversity gives at most modest gains, so multi-hop QA needs structure-aware evidence composition.

