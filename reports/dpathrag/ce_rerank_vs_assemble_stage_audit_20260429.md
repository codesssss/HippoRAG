# CE Rerank vs Assemble-Stage CE Audit

Date: 2026-04-29

## Decision

Do not write that "CE is dead" without qualification.

There are two different CE paths in the current code and artifacts:

1. `--cross_encoder_rerank`: pointwise CE reranking over the existing baseline
   top-k/window. This is negative on the available 2Wiki-100 run.
2. `bridge_append + assemble_mode=cross_encoder`: expand the candidate set first,
   then use `bge-reranker-v2-m3` as the assemble-stage scorer. This is positive
   in the 100-query runs and must be retained as a strong baseline.

Correct shorthand:

```text
Pointwise CE on the original top-k is not sufficient.
CE after structured expansion is a strong assemble-stage baseline.
```

## Baseline Top-K CE Rerank

Source:

- `outputs/2wikimultihopqa/eval_reports/2wiki100_ce_rerank_m3_alpha07_w20.json`
- `outputs/2wikimultihopqa/eval_reports/2wiki100_ce_rerank_m3_alpha09_w20.json`

| Dataset | Run | Base EM | CE EM | Delta EM | Base F1 | CE F1 | Delta F1 | Base R@5 | CE R@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | alpha=0.7, window=20 | 0.4600 | 0.3900 | -0.0700 | 0.5188 | 0.4694 | -0.0494 | 0.8450 | 0.8250 |
| 2Wiki | alpha=0.9, window=20 | 0.4600 | 0.4100 | -0.0500 | 0.5188 | 0.4850 | -0.0338 | 0.8450 | 0.8500 |

Interpretation:

- CE can improve or preserve retrieval recall in one setting (`alpha=0.9` R@5
  is 0.8500 vs 0.8450), while still hurting reader EM/F1.
- The failure is not simply "CE cannot find relevant documents"; it is that
  pointwise reranking of the original context does not reliably assemble a
  reader-useful multi-hop evidence set.

## Bridge-Append Assemble-Stage CE

Source pattern:

- `outputs_step0_general_{dataset}/eval_reports/bridge_append_cross_encoder_*.json`

All rows below are limit=100 runs using `bge-reranker-v2-m3` as
`assemble_mode=cross_encoder`.

| Dataset | Variant | k | Base EM | Method EM | Delta EM | Base F1 | Method F1 | Delta F1 | Base R@5 | Method R@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | append0 CE | 5 | 0.3600 | 0.4200 | +0.0600 | 0.4008 | 0.4553 | +0.0545 | 0.7800 | 0.7750 |
| 2Wiki | append3 CE | 5 | 0.3600 | 0.4300 | +0.0700 | 0.4008 | 0.4728 | +0.0720 | 0.7800 | 0.7800 |
| 2Wiki | control top-k CE | 5 | 0.3600 | 0.3700 | +0.0100 | 0.4008 | 0.4083 | +0.0075 | 0.7800 | 0.7800 |
| 2Wiki | append0 CE | 7 | 0.4200 | 0.4400 | +0.0200 | 0.4642 | 0.5152 | +0.0510 | 0.7800 | 0.7750 |
| 2Wiki | append3 CE | 7 | 0.4200 | 0.4400 | +0.0200 | 0.4642 | 0.5192 | +0.0550 | 0.7800 | 0.7800 |
| 2Wiki | control top-k CE | 7 | 0.4200 | 0.4300 | +0.0100 | 0.4642 | 0.4860 | +0.0218 | 0.7800 | 0.7900 |
| 2Wiki | append0 CE | 10 | 0.4400 | 0.4600 | +0.0200 | 0.5018 | 0.4992 | -0.0026 | 0.7800 | 0.7750 |
| 2Wiki | append3 CE | 10 | 0.4400 | 0.5000 | +0.0600 | 0.5018 | 0.5403 | +0.0385 | 0.7800 | 0.7800 |
| 2Wiki | control top-k CE | 10 | 0.4500 | 0.4600 | +0.0100 | 0.5118 | 0.4992 | -0.0126 | 0.7800 | 0.7750 |
| HotpotQA | append0 CE | 5 | 0.5800 | 0.6200 | +0.0400 | 0.6967 | 0.7173 | +0.0206 | 0.9150 | 0.9350 |
| HotpotQA | append3 CE | 5 | 0.5800 | 0.6200 | +0.0400 | 0.6967 | 0.7173 | +0.0206 | 0.9150 | 0.9400 |
| HotpotQA | control top-k CE | 5 | 0.5800 | 0.5600 | -0.0200 | 0.6967 | 0.6746 | -0.0221 | 0.9150 | 0.9150 |
| HotpotQA | append0 CE | 7 | 0.5800 | 0.6000 | +0.0200 | 0.6953 | 0.7136 | +0.0183 | 0.9150 | 0.9350 |
| HotpotQA | append3 CE | 7 | 0.5800 | 0.6000 | +0.0200 | 0.6953 | 0.7136 | +0.0183 | 0.9150 | 0.9400 |
| HotpotQA | control top-k CE | 7 | 0.5800 | 0.6000 | +0.0200 | 0.6953 | 0.7222 | +0.0269 | 0.9150 | 0.9300 |
| HotpotQA | append0 CE | 10 | 0.6000 | 0.5900 | -0.0100 | 0.7295 | 0.6981 | -0.0314 | 0.9150 | 0.9350 |
| HotpotQA | append3 CE | 10 | 0.6000 | 0.5800 | -0.0200 | 0.7295 | 0.6881 | -0.0414 | 0.9150 | 0.9400 |
| HotpotQA | control top-k CE | 10 | 0.6000 | 0.5900 | -0.0100 | 0.7295 | 0.6981 | -0.0314 | 0.9150 | 0.9350 |
| MuSiQue | append0 CE | 5 | 0.2700 | 0.3100 | +0.0400 | 0.3359 | 0.3698 | +0.0339 | 0.6150 | 0.6500 |
| MuSiQue | append3 CE | 5 | 0.2700 | 0.3200 | +0.0500 | 0.3348 | 0.3810 | +0.0462 | 0.6150 | 0.6550 |
| MuSiQue | control top-k CE | 5 | 0.2700 | 0.3100 | +0.0400 | 0.3359 | 0.3577 | +0.0218 | 0.6150 | 0.6150 |
| MuSiQue | append0 CE | 7 | 0.3200 | 0.3400 | +0.0200 | 0.3924 | 0.4034 | +0.0110 | 0.6150 | 0.6500 |
| MuSiQue | append3 CE | 7 | 0.3200 | 0.3800 | +0.0600 | 0.3924 | 0.4385 | +0.0461 | 0.6150 | 0.6550 |
| MuSiQue | control top-k CE | 7 | 0.3200 | 0.3400 | +0.0200 | 0.3924 | 0.4007 | +0.0083 | 0.6150 | 0.6375 |
| MuSiQue | append0 CE | 10 | 0.3400 | 0.3900 | +0.0500 | 0.4235 | 0.4412 | +0.0177 | 0.6150 | 0.6500 |
| MuSiQue | append3 CE | 10 | 0.3400 | 0.3700 | +0.0300 | 0.4235 | 0.4422 | +0.0187 | 0.6150 | 0.6550 |
| MuSiQue | control top-k CE | 10 | 0.3400 | 0.3900 | +0.0500 | 0.4235 | 0.4412 | +0.0177 | 0.6150 | 0.6500 |

Interpretation:

- The user's recollection is correct: assemble-stage CE is often above baseline
  by +0.02 to +0.07 EM and +0.01 to +0.07 F1 in these 100-query runs.
- The strongest 2Wiki k=5 result is not just "CE alone": `append3 CE` gives
  +0.0700 EM / +0.0720 F1, while `control top-k CE` gives only +0.0100 EM /
  +0.0075 F1.
- On MuSiQue, CE controls are also positive in some settings, so the method
  story must not rely on claiming that generic CE is uniformly weak.
- HotpotQA shows a k sensitivity: k=5/7 positive, k=10 negative. This is a
  reader-context composition effect, not a simple retrieval-recall effect.

## Paper Implication

This audit changes the wording, not the overall caution.

Do not use:

```text
CE reranking is dead.
```

Use:

```text
Pointwise CE over the original context is not enough, but CE becomes a strong
baseline when used as the assemble-stage scorer after expansion.
```

For the DAEC-L1 / Expand-then-Compose paper, this means:

1. `bridge_append + cross_encoder` must be treated as a strong baseline, not as
   a negative control.
2. DAEC-L1 must be compared against assemble-stage CE in a matched pool/budget
   setting before claiming a stronger set-selection method.
3. If DAEC-L1 only matches assemble-stage CE, the contribution should be framed
   as a diagnostic / negative-study paper rather than a method-novelty paper.
4. If DAEC-L1 beats assemble-stage CE on full1000 or in hard buckets, the claim
   becomes stronger: structure-aware set selection beats a strong CE assembly
   baseline under the same expanded pool.

