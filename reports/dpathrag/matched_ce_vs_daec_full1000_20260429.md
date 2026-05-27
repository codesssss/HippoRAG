# Matched CE vs DAEC Full1000 Gate

Date: 2026-04-29

## Status

The queue completed successfully.

Artifacts:

- `run_logs/matched_ce_daec_full1000_20260429/2wikimultihopqa_dense_pool100_ce100_qatopk5.json`
- `run_logs/matched_ce_daec_full1000_20260429/hotpotqa_dense_pool100_ce100_qatopk5.json`
- `run_logs/matched_ce_daec_full1000_20260429/musique_dense_pool100_ce100_qatopk7.json`
- `run_logs/matched_ce_daec_full1000_20260429/2wikimultihopqa_dense_pool100_bridge_append_ce_qatopk5.json`
- `run_logs/matched_ce_daec_full1000_20260429/hotpotqa_dense_pool100_bridge_append_ce_qatopk5.json`
- `run_logs/matched_ce_daec_full1000_20260429/musique_dense_pool100_bridge_append_ce_qatopk7.json`
- `run_logs/matched_ce_daec_full1000_20260429/musique_dense_pool100_daec_noisyor_qatopk7.json`
- `run_logs/matched_ce_daec_full1000_20260429/musique_dense_pool100_oracle100_qatopk7.json`
- `run_logs/matched_ce_daec_full1000_20260429/2wikimultihopqa_dense_pool100_mmr_dpp_qatopk5.json`
- `run_logs/matched_ce_daec_full1000_20260429/hotpotqa_dense_pool100_mmr_dpp_qatopk5.json`
- `run_logs/matched_ce_daec_full1000_20260429/musique_dense_pool100_mmr_dpp_qatopk7.json`

Existing DAEC-L1 k=5 artifacts reused:

- `run_logs/daec_noisyor_dense_pool100_2wiki_fixclean_full1000_20260428.json`
- `run_logs/daec_noisyor_dense_pool100_hotpotqa_fixclean_full1000_20260428.json`

Existing oracle k=5 artifacts reused:

- `run_logs/layer1_dense_pool_eval_20260424/2wikimultihopqa_dense_pool_daec_oracle.json`
- `run_logs/layer1_dense_pool_eval_20260424/hotpotqa_dense_pool_daec_oracle.json`

## Main Table

| Dataset | k | Method | EM | F1 | Delta EM | Delta F1 | R@5 |
|---|---:|---|---:|---:|---:|---:|---:|
| 2Wiki | 5 | baseline dense top-k | 0.4550 | 0.4991 | 0.0000 | 0.0000 | 0.7238 |
| 2Wiki | 5 | CE@100 | 0.4600 | 0.5045 | +0.0050 | +0.0054 | 0.7282 |
| 2Wiki | 5 | bridge_append+CE | 0.4890 | 0.5423 | +0.0340 | +0.0432 | 0.7642 |
| 2Wiki | 5 | DAEC-L1 | 0.4980 | 0.5541 | +0.0430 | +0.0550 | 0.7855 |
| 2Wiki | 5 | MMR | 0.4545 | 0.4993 | +0.0000 | +0.0000 | 0.7260 |
| 2Wiki | 5 | DPP | 0.4545 | 0.4993 | +0.0000 | +0.0000 | 0.7260 |
| 2Wiki | 5 | oracle@100 | 0.5810 | 0.6396 | +0.1260 | +0.1412 | - |
| HotpotQA | 5 | baseline dense top-k | 0.5950 | 0.7106 | 0.0000 | 0.0000 | 0.9305 |
| HotpotQA | 5 | CE@100 | 0.5910 | 0.7047 | -0.0040 | -0.0059 | 0.9365 |
| HotpotQA | 5 | bridge_append+CE | 0.6100 | 0.7271 | +0.0150 | +0.0165 | 0.9490 |
| HotpotQA | 5 | DAEC-L1 | 0.6120 | 0.7267 | +0.0170 | +0.0161 | 0.9360 |
| HotpotQA | 5 | MMR | 0.6009 | 0.7149 | +0.0000 | +0.0000 | 0.9351 |
| HotpotQA | 5 | DPP | 0.6009 | 0.7149 | +0.0000 | +0.0000 | 0.9351 |
| HotpotQA | 5 | oracle@100 | 0.6500 | 0.7669 | +0.0550 | +0.0563 | - |
| MuSiQue | 7 | baseline dense top-k | 0.3110 | 0.4079 | 0.0000 | 0.0000 | 0.6628 |
| MuSiQue | 7 | CE@100 | 0.2840 | 0.3846 | -0.0270 | -0.0233 | 0.6314 |
| MuSiQue | 7 | bridge_append+CE | 0.3160 | 0.4144 | +0.0050 | +0.0065 | 0.6757 |
| MuSiQue | 7 | DAEC-L1 | 0.3040 | 0.3982 | -0.0070 | -0.0097 | 0.6501 |
| MuSiQue | 7 | MMR | 0.2930 | 0.3871 | -0.0180 | -0.0208 | 0.6467 |
| MuSiQue | 7 | DPP | 0.3150 | 0.4108 | +0.0040 | +0.0029 | 0.6629 |
| MuSiQue | 7 | oracle@100 | 0.4260 | 0.5375 | +0.1150 | +0.1296 | - |

MMR/DPP caveat:

- 2Wiki diversity study used 988 aligned queries.
- HotpotQA diversity study used 932 aligned queries.
- MuSiQue diversity study used 1000 aligned queries.

So MMR/DPP are useful as a sanity baseline, but not a primary matched
conclusion until the alignment issue is fixed.

## Paired Bootstrap

Bootstrap uses per-query deltas, 2000 resamples, 95% percentile CI.

| Dataset | k | Method | n | Delta EM | 95% CI | Delta F1 | 95% CI | Wrong->Correct | Correct->Wrong |
|---|---:|---|---:|---:|---|---:|---|---:|---:|
| 2Wiki | 5 | CE@100 | 1000 | +0.0050 | [-0.0210, +0.0300] | +0.0054 | [-0.0190, +0.0309] | 82 | 77 |
| 2Wiki | 5 | bridge_append+CE | 1000 | +0.0340 | [+0.0120, +0.0550] | +0.0432 | [+0.0216, +0.0643] | 80 | 46 |
| 2Wiki | 5 | DAEC-L1 | 1000 | +0.0430 | [+0.0190, +0.0660] | +0.0551 | [+0.0337, +0.0783] | 95 | 52 |
| HotpotQA | 5 | CE@100 | 1000 | -0.0040 | [-0.0250, +0.0170] | -0.0059 | [-0.0251, +0.0143] | 57 | 61 |
| HotpotQA | 5 | bridge_append+CE | 1000 | +0.0150 | [-0.0050, +0.0350] | +0.0165 | [-0.0003, +0.0337] | 58 | 43 |
| HotpotQA | 5 | DAEC-L1 | 1000 | +0.0170 | [+0.0000, +0.0340] | +0.0161 | [+0.0005, +0.0316] | 45 | 28 |
| MuSiQue | 7 | CE@100 | 1000 | -0.0270 | [-0.0520, -0.0020] | -0.0234 | [-0.0479, +0.0014] | 67 | 94 |
| MuSiQue | 7 | bridge_append+CE | 1000 | +0.0050 | [-0.0150, +0.0250] | +0.0064 | [-0.0142, +0.0273] | 56 | 51 |
| MuSiQue | 7 | DAEC-L1 | 1000 | -0.0070 | [-0.0270, +0.0120] | -0.0098 | [-0.0306, +0.0106] | 48 | 55 |

## Direct Bridge-vs-DAEC Comparison

Positive means bridge_append+CE is better than DAEC-L1.

| Dataset | k | n | Delta EM | 95% CI | Delta F1 | 95% CI |
|---|---:|---:|---:|---|---:|---|
| 2Wiki | 5 | 1000 | -0.0090 | [-0.0330, +0.0160] | -0.0118 | [-0.0352, +0.0133] |
| HotpotQA | 5 | 1000 | -0.0020 | [-0.0210, +0.0160] | +0.0004 | [-0.0168, +0.0174] |
| MuSiQue | 7 | 999 | +0.0120 | [-0.0090, +0.0340] | +0.0157 | [-0.0062, +0.0394] |

No bridge-vs-DAEC difference is significant under this bootstrap.

## Interpretation

1. **Pure CE@100 is not the strong method.**
   It is nearly flat on 2Wiki, slightly negative on HotpotQA, and clearly
   negative on MuSiQue k=7. This falsifies the broad claim that a stronger
   semantic reranker alone solves the assembly problem.

2. **Expansion is the source of most CE gains.**
   `bridge_append+CE` improves over baseline on 2Wiki and is directionally
   positive on HotpotQA and MuSiQue. The difference between CE@100 and
   bridge_append+CE shows that the expansion substrate matters more than
   applying CE to the full top100.

3. **DAEC-L1 and bridge_append+CE are statistically tied.**
   DAEC-L1 is numerically higher on 2Wiki and HotpotQA, while bridge_append+CE
   is numerically higher on MuSiQue k=7. All direct bridge-vs-DAEC confidence
   intervals cross zero.

4. **The paper should not claim DAEC-L1 clearly beats CE assembly.**
   The defensible framing is that structured expansion plus reader-budget
   assembly matters, while selector choice is dataset-sensitive.

5. **Oracle headroom remains large.**
   Top100 oracle gains are still +0.126 EM on 2Wiki, +0.055 EM on HotpotQA,
   and +0.115 EM on MuSiQue k=7. Current train-free selectors recover only a
   small fraction of this headroom.

## Paper-Level Decision

This gate does not support a main-method claim of:

```text
DAEC-L1 significantly outperforms strong CE assembly.
```

It supports a narrower diagnostic claim:

```text
On strong top100 pools, pure pointwise CE is insufficient; structured expansion
and reader-budget assembly are both useful, but current train-free selectors
remain statistically tied and far below oracle headroom.
```

Practical consequence:

- DAEC-L1 can remain a positive method variant, especially on 2Wiki.
- `bridge_append+CE` must be a strong baseline or co-primary variant.
- The paper should be framed as an evidence-composition / diagnostic study
  unless a new selector significantly separates from bridge_append+CE.

