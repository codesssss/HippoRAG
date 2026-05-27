# PCEC m=4 Mechanism Audit (8B Full1000)

This audit uses existing 8B frontier artifacts. It is title-level unless explicitly stated otherwise.

Important caveat: `pool_doc_scores` are rank-coded in these artifacts (`200..196`, then `95..`), so a rank-4/5 score-gap plot would be circular and is not used as evidence.

## Baseline Gold Slot Distribution

| Dataset | Top4 all-gold | Top5 all-gold | Rank4 gold | Rank5 gold | Rank4 critical | Rank5 critical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 68.6% | 70.6% | 7.9% | 3.5% | 6.5% | 2.6% |
| hotpotqa | 88.0% | 90.5% | 12.6% | 4.2% | 10.2% | 3.0% |
| musique | 40.8% | 46.0% | 29.8% | 22.6% | 22.7% | 14.5% |

Definitions: `Rank5 critical` means rank 5 is gold and the top-4 prefix is not already all-gold. This is the direct gold-out risk of replacing slot 5. `Rank4 critical` is the analogous risk introduced by allowing replacement of slot 4.

## Swap-Out Hazard by Variant

| Dataset | Variant | Swaps | Gold-out queries | Complete rescues | Complete regressions | Slot hazards |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2wikimultihopqa | top4_max1 | 483 | 13 | 155 | 5 | r5: 13/483 (2.7%) |
| 2wikimultihopqa | top3_max2 | 539 | 19 | 179 | 6 | r4: 19/459 (4.1%)<br>r5: 0/80 (0.0%) |
| 2wikimultihopqa | top2_max3 | 540 | 39 | 178 | 18 | r3: 27/217 (12.4%)<br>r4: 13/269 (4.8%)<br>r5: 0/54 (0.0%) |
| 2wikimultihopqa | top1_max2 | 539 | 94 | 154 | 40 | r2: 78/231 (33.8%)<br>r3: 11/134 (8.2%)<br>r4: 7/170 (4.1%)<br>r5: 0/4 (0.0%) |
| hotpotqa | top4_max1 | 203 | 7 | 28 | 1 | r5: 7/203 (3.4%) |
| hotpotqa | top3_max2 | 224 | 15 | 29 | 8 | r4: 13/194 (6.7%)<br>r5: 2/30 (6.7%) |
| hotpotqa | top2_max3 | 228 | 37 | 28 | 27 | r3: 29/128 (22.7%)<br>r4: 7/83 (8.4%)<br>r5: 1/17 (5.9%) |
| hotpotqa | top1_max2 | 226 | 57 | 28 | 45 | r2: 44/126 (34.9%)<br>r3: 12/66 (18.2%)<br>r4: 1/27 (3.7%)<br>r5: 0/7 (0.0%) |
| musique | top4_max1 | 539 | 137 | 47 | 22 | r5: 137/539 (25.4%) |
| musique | top3_max2 | 717 | 174 | 55 | 34 | r4: 138/495 (27.9%)<br>r5: 65/222 (29.3%) |
| musique | top2_max3 | 784 | 241 | 52 | 50 | r3: 174/434 (40.1%)<br>r4: 75/220 (34.1%)<br>r5: 41/130 (31.5%) |
| musique | top1_max2 | 740 | 228 | 55 | 63 | r2: 124/423 (29.3%)<br>r3: 81/189 (42.9%)<br>r4: 37/91 (40.7%)<br>r5: 16/37 (43.2%) |

## Oracle Variant View

`Conservative best m` chooses the largest preservation prefix among variants that achieve the maximum title coverage for that query. It asks: if an oracle picked among the already-run frontier variants, how often would it need to relax below m=4?

| Dataset | m4 matches max coverage | m4 unique best | m4 improves over baseline | m4 worsens vs baseline | Conservative-best m distribution |
| --- | ---: | ---: | ---: | ---: | --- |
| 2wikimultihopqa | 969 (96.9%) | 0 (0.0%) | 160 (16.0%) | 5 (0.5%) | m=5: 816 (81.6%), m=4: 158 (15.8%), m=3: 25 (2.5%), m=1: 1 (0.1%) |
| hotpotqa | 998 (99.8%) | 0 (0.0%) | 29 (2.9%) | 1 (0.1%) | m=5: 970 (97.0%), m=4: 29 (2.9%), m=3: 1 (0.1%) |
| musique | 923 (92.3%) | 0 (0.0%) | 84 (8.4%) | 41 (4.1%) | m=5: 883 (88.3%), m=4: 79 (7.9%), m=3: 26 (2.6%), m=2: 10 (1.0%), m=1: 2 (0.2%) |

## Non-Baseline Oracle Need

This table removes queries for which the no-op ETv3 top-5 already ties the best title coverage among the frontier variants. It measures where the composition layer actually matters.

| Dataset | Non-baseline oracle need | m4 share among non-baseline | Below-m4 share among non-baseline |
| --- | ---: | ---: | ---: |
| 2wikimultihopqa | 184 (18.4%) | 158 (85.9%) | 26 (14.1%) |
| hotpotqa | 30 (3.0%) | 29 (96.7%) | 1 (3.3%) |
| musique | 117 (11.7%) | 79 (67.5%) | 38 (32.5%) |

## Interpretation

1. `m=4` is not explained by a score cliff: the stored pool scores are rank-coded, so score-gap analysis would be circular.
2. The strongest mechanism evidence is slot vulnerability. Rank 4 is more often critical than rank 5 in the baseline top-5, and allowing replacement of earlier slots sharply raises gold-out risk, especially once rank 3 or rank 2 becomes replaceable.
3. `m=4` captures most non-baseline oracle need on 2Wiki and HotpotQA, but it is not a unique oracle optimum. The right claim is conservative: with K=5, `m=4` is the robust one-slot boundary that preserves most selector gains while limiting gold displacement and reader-context risk.
4. MuSiQue remains the warning case: even rank-5 replacement has high gold-out risk, and below-m4 variants add many swaps with limited extra rescues. This supports adaptive retention as future work rather than a stronger fixed cutoff.
