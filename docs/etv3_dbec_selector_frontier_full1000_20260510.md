# ETv3 + DBEC Selector Frontier P1

Scope: selector-level title metrics only. Reader QA is not mixed into this frontier table.

## Overall Frontier

| Dataset | Variant | Preserve / Admit | Title-all@5 | Δ all@5 | Title recall@5 | Δ recall | Changed | Swaps | Rescue | Regression | Gold-out queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | top5_max0 | 5/0 | 0.7060 | +0.0000 | 0.9015 | +0.0000 | 0 | 0 | 0 | 0 | 0 |
| 2wikimultihopqa | top4_max1 | 4/1 | 0.8560 | +0.1500 | 0.9503 | +0.0488 | 483 | 483 | 155 | 5 | 13 |
| 2wikimultihopqa | top3_max2 | 3/2 | 0.8790 | +0.1730 | 0.9563 | +0.0548 | 487 | 539 | 179 | 6 | 19 |
| 2wikimultihopqa | top2_max3 | 2/3 | 0.8660 | +0.1600 | 0.9495 | +0.0480 | 487 | 540 | 178 | 18 | 39 |
| 2wikimultihopqa | top1_max2 | 1/2 | 0.8200 | +0.1140 | 0.9307 | +0.0293 | 487 | 539 | 154 | 40 | 94 |
| hotpotqa | top5_max0 | 5/0 | 0.9050 | +0.0000 | 0.9505 | +0.0000 | 0 | 0 | 0 | 0 | 0 |
| hotpotqa | top4_max1 | 4/1 | 0.9320 | +0.0270 | 0.9645 | +0.0140 | 203 | 203 | 28 | 1 | 7 |
| hotpotqa | top3_max2 | 3/2 | 0.9260 | +0.0210 | 0.9610 | +0.0105 | 208 | 224 | 29 | 8 | 15 |
| hotpotqa | top2_max3 | 2/3 | 0.9060 | +0.0010 | 0.9500 | -0.0005 | 209 | 228 | 28 | 27 | 37 |
| hotpotqa | top1_max2 | 1/2 | 0.8880 | -0.0170 | 0.9390 | -0.0115 | 209 | 226 | 28 | 45 | 57 |
| musique | top5_max0 | 5/0 | 0.4600 | +0.0000 | 0.7391 | +0.0000 | 0 | 0 | 0 | 0 | 0 |
| musique | top4_max1 | 4/1 | 0.4850 | +0.0250 | 0.7551 | +0.0160 | 522 | 539 | 47 | 22 | 137 |
| musique | top3_max2 | 3/2 | 0.4810 | +0.0210 | 0.7548 | +0.0157 | 546 | 717 | 55 | 34 | 174 |
| musique | top2_max3 | 2/3 | 0.4620 | +0.0020 | 0.7357 | -0.0034 | 548 | 784 | 52 | 50 | 241 |
| musique | top1_max2 | 1/2 | 0.4520 | -0.0080 | 0.7322 | -0.0068 | 548 | 740 | 55 | 63 | 228 |

## MuSiQue Depth Frontier

| Variant | Slice | Title-all@5 | Δ all@5 | Title recall@5 | Δ recall | Changed | Swaps | Rescue | Regression | Gold-out queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top5_max0 | 2_doc | 0.6737 | +0.0000 | 0.8292 | +0.0000 | 0 | 0 | 0 | 0 | 0 |
| top5_max0 | 3_doc | 0.3323 | +0.0000 | 0.7257 | +0.0000 | 0 | 0 | 0 | 0 | 0 |
| top5_max0 | 4_doc | 0.0361 | +0.0000 | 0.4834 | +0.0000 | 0 | 0 | 0 | 0 | 0 |
| top4_max1 | 2_doc | 0.7046 | +0.0309 | 0.8456 | +0.0164 | 207 | 215 | 23 | 7 | 49 |
| top4_max1 | 3_doc | 0.3513 | +0.0190 | 0.7384 | +0.0127 | 188 | 195 | 20 | 14 | 61 |
| top4_max1 | 4_doc | 0.0542 | +0.0181 | 0.5045 | +0.0211 | 127 | 129 | 4 | 1 | 27 |
| top3_max2 | 2_doc | 0.6911 | +0.0174 | 0.8388 | +0.0097 | 216 | 245 | 26 | 17 | 62 |
| top3_max2 | 3_doc | 0.3576 | +0.0253 | 0.7405 | +0.0148 | 197 | 272 | 23 | 15 | 81 |
| top3_max2 | 4_doc | 0.0602 | +0.0241 | 0.5196 | +0.0361 | 133 | 200 | 6 | 2 | 31 |
| top2_max3 | 2_doc | 0.6737 | +0.0000 | 0.8282 | -0.0010 | 214 | 256 | 25 | 25 | 81 |
| top2_max3 | 3_doc | 0.3259 | -0.0063 | 0.7110 | -0.0148 | 199 | 293 | 20 | 22 | 108 |
| top2_max3 | 4_doc | 0.0602 | +0.0241 | 0.4940 | +0.0105 | 135 | 235 | 7 | 3 | 52 |
| top1_max2 | 2_doc | 0.6525 | -0.0212 | 0.8195 | -0.0097 | 216 | 251 | 24 | 35 | 83 |
| top1_max2 | 3_doc | 0.3323 | +0.0000 | 0.7152 | -0.0105 | 198 | 281 | 25 | 25 | 90 |
| top1_max2 | 4_doc | 0.0542 | +0.0181 | 0.4925 | +0.0090 | 134 | 208 | 6 | 3 | 55 |

## Immediate Read

- `top3/max2` gives the highest 2Wiki title-all@5, so the frontier is not a simple top4-only story.
- `top4/max1` is the most robust preservation point: it is best on HotpotQA and MuSiQue overall and keeps gold-out regressions much lower than looser variants.
- Loosening preservation beyond top4 (`top2/max3`, `top1/max2`) increases edit volume and gold-out risk faster than it improves set completeness.
- The frontier is non-monotonic: admission capacity cannot compensate for weaker preservation.
- This supports preservation-constrained residual admission as the ETv4-composition prior, but it does not prove a state-binding mechanism.
