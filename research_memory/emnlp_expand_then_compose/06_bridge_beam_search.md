# Bridge-Beam Search Decision Memo

Last updated: 2026-03-30

## Scope

This note records the first fixed-score search ablation for the non-oracle selector story.

Goal:
- hold the bridge-aware local score fixed
- compare `bridge_greedy` vs `bridge_beam`
- test whether the remaining bottleneck is search myopia rather than only local scoring

Canonical backbone:
- `causal_engine_version = v2`
- `causal_v2_base_retrieval_mode = legacy_fact_graph`
- `causal_enabled = false`

Implementation checkpoint:
- selector code commit: `53f1c8f` (`Add bridge beam setwise selector`)
- key file: `scripts/eval_causal_qwen3.py`
- validation file: `tests/test_setwise_selector.py`

## Main 2Wiki Search Ablation

Report files:
- `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_beam_pilot_100_legacy.json`
- `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_100_legacy.json`

Matched setup:
- dataset: `2wikimultihopqa`
- limit: `100`
- pool: `100`
- anchor count: `2`
- selector score: same bridge-aware score for both methods
- beam params: `beam_width=4`, `beam_expand_per_state=4`

Results:

| Method | EM | Delta EM | F1 | Delta F1 | Recall@5 | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.3800 | — | 0.4332 | — | 0.7800 | 0.8675 |
| Bridge-Greedy | 0.4300 | +0.0500 | 0.4726 | +0.0394 | 0.8150 | 0.8825 |
| Bridge-Beam | 0.4600 | +0.0800 | 0.4924 | +0.0592 | 0.8075 | 0.8825 |

Beam over greedy:
- `EM +0.0300`
- `F1 +0.0198`

Bucket results:

| Bucket | Greedy Delta EM | Beam Delta EM | Greedy Delta F1 | Beam Delta F1 |
|---|---:|---:|---:|---:|
| `2-doc` | +0.0909 | +0.0909 | +0.0812 | +0.0682 |
| `4-doc` | -0.0870 | +0.0435 | -0.1007 | +0.0291 |

## MuSiQue Beam Outcome

Report file:
- `outputs_step0_general_musique/eval_reports/setwise_bridge_beam_pilot_100_legacy.json`

Matched setup:
- dataset: `musique`
- limit: `100`
- pool: `100`
- anchor count: `2`
- selector score: same bridge-aware score family as `2Wiki`
- beam params: `beam_width=4`, `beam_expand_per_state=4`

Results:

| Method | EM | Delta EM | F1 | Delta F1 | Recall@5 | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.2600 | — | 0.3466 | — | 0.6150 | 0.7858 |
| Bridge-Beam | 0.1800 | -0.0800 | 0.2498 | -0.0968 | 0.4950 | 0.7933 |

Bucket results:

| Bucket | Beam Delta EM | Beam Delta F1 |
|---|---:|---:|
| `2-doc` | -0.1458 | -0.1410 |
| `3-doc` | -0.0667 | -0.1122 |
| `4-doc` | +0.0455 | +0.0205 |

What this means:
- The current bridge-aware score does recover some hard `4-doc` cases on `MuSiQue`.
- But it over-trades away easy and medium cases: `Recall@5` drops from `0.6150` to `0.4950`.
- So the current `beam` configuration is not yet a cross-dataset practical method.

Immediate interpretation:
- `beam` is still a useful mechanism probe because it helps exactly where deeper composition should matter.
- But the `MuSiQue` result says the local score is not calibrated well enough yet for broad deployment.
- The remaining question is now narrower:
  - is `beam` less bad than matched `greedy` on `MuSiQue`, which would support a "search helps under the same score" claim
  - or is the score itself the dominant problem on this dataset

## MuSiQue Matched Greedy-vs-Beam Check

Greedy report file:
- `outputs_step0_general_musique/eval_reports/setwise_bridge_greedy_pilot_100_legacy.json`

Matched comparison:

| Method | EM | Delta EM | F1 | Delta F1 | Recall@5 | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.2600 | — | 0.3466 | — | 0.6150 | 0.7858 |
| Bridge-Greedy | 0.1800 | -0.0800 | 0.2426 | -0.1040 | 0.4925 | 0.7933 |
| Bridge-Beam | 0.1800 | -0.0800 | 0.2498 | -0.0968 | 0.4950 | 0.7933 |

Bucket comparison:

| Bucket | Greedy Delta EM | Beam Delta EM | Greedy Delta F1 | Beam Delta F1 |
|---|---:|---:|---:|---:|
| `2-doc` | -0.1250 | -0.1458 | -0.1351 | -0.1410 |
| `3-doc` | -0.0667 | -0.0667 | -0.1122 | -0.1122 |
| `4-doc` | +0.0000 | +0.0455 | -0.0250 | +0.0205 |

Interpretation:
- On `MuSiQue`, `beam` is slightly better than matched `greedy`, but only marginally.
- The dominant failure is not search anymore; it is the local bridge-aware score, which knocks down `Recall@5` for easy and medium cases.
- This means the current evidence-set selector story is:
  - strong and paper-usable on `2Wiki`
  - mechanistically suggestive but not practically robust on `MuSiQue`

## MuSiQue Reserved-Prefix Rescue

Report file:
- `outputs_step0_general_musique/eval_reports/setwise_bridge_beam_pilot_100_legacy_reserve3_dedup.json`

Updated setup:
- same canonical backbone and bridge-aware score
- add `reserve_top_m=3`
- add `non_anchor_title_dedup=true`

Results:

| Method | EM | Delta EM | F1 | Delta F1 | Recall@5 | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.2600 | — | 0.3466 | — | 0.6150 | 0.7858 |
| Beam + reserve3 + dedup | 0.2700 | +0.0100 | 0.3382 | -0.0084 | 0.5508 | 0.7958 |

Bucket results:

| Bucket | Delta EM | Delta F1 |
|---|---:|---:|
| `2-doc` | -0.0208 | -0.0302 |
| `3-doc` | +0.0333 | -0.0022 |
| `4-doc` | +0.0455 | +0.0303 |

Interpretation:
- This rescues the catastrophic MuSiQue failure: overall `EM` is now positive instead of `-0.08`.
- The main gain comes from preserving shallow relevance while still letting beam help on harder buckets.
- `2-doc` damage is now small instead of destructive, while `3-doc` and `4-doc` are positive on `EM`.
- `F1` is still slightly below baseline, so the method is improved but not fully stable yet.

## Hotpot Shared-Config Boundary Check

Report file:
- `outputs_step0_general_hotpotqa/eval_reports/setwise_bridge_beam_pilot_100_legacy_reserve3_dedup.json`

Matched setup:
- dataset: `hotpotqa`
- limit: `100`
- pool: `100`
- anchor count: `2`
- reserve top `3`
- non-anchor title dedup: `true`
- same canonical `legacy_fact_graph` backbone as above

Results:

| Method | EM | Delta EM | F1 | Delta F1 | Recall@5 | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5900 | — | 0.7114 | — | 0.9150 | 0.9650 |
| Beam + reserve3 + dedup | 0.5500 | -0.0400 | 0.6610 | -0.0504 | 0.8600 | 0.9700 |

Bucket results:

| Bucket | Delta EM | Delta F1 |
|---|---:|---:|
| `2-doc` | -0.0400 | -0.0504 |

Interpretation:
- This shared configuration does not transfer cleanly to shallow `HotpotQA`.
- `Recall@20` improves slightly, but `Recall@5` drops by more than five points and QA quality falls with it.
- So even with a stronger reserved prefix, the current beam selector is still too willing to trade away top-ranked evidence when the dataset is already shallow.

## Interpretation

What this means:
- The bridge-aware selector is now clearly useful on the canonical `2Wiki` setting.
- Search matters on top of scoring: `beam` beats `greedy` under the same local score.
- The gain is not explained by shallow recall alone.
- On `MuSiQue`, stronger prefix preservation makes the method usable enough to keep as a hard-dataset stress-test result.
- On `HotpotQA`, the same shared config is negative, so the current selector is not a universal drop-in replacement.

Why the last point matters:
- `bridge_greedy` has slightly higher `Recall@5` than `bridge_beam` (`0.8150` vs `0.8075`)
- `Recall@20` is identical (`0.8825`)
- but `bridge_beam` still wins on QA, especially on `4-doc` queries

Conclusion:
- `bridge_beam` improves evidence composition, not just top-5 lexical coverage.
- This is exactly the kind of evidence needed for the `Expand-then-Compose` story.
- Today the strongest clean claim is still on canonical `2Wiki`.
- `MuSiQue` is now a partially rescued hard-case stress test rather than a total failure.
- `HotpotQA` is an explicit shallow boundary condition showing that the shared config still over-explores when top evidence is already concentrated.

## Paper Positioning

Recommended framing:
- present `bridge_beam` as the strongest current practical candidate on `2Wiki`
- present `bridge_greedy` as the search-ablation control
- present reserved-prefix `bridge_beam` on `MuSiQue` as a robustness-improvement stress test
- present `HotpotQA` as a boundary-condition check rather than a success case
- make the claim narrow and defensible:
  - widening the pool is necessary but not sufficient
  - under a fixed bridge-aware score, beam search recovers better evidence sets than greedy selection

What should not be claimed yet:
- do not claim broad cross-dataset robustness
- do not claim that the current shared score/config generalizes cleanly across datasets
- do not claim that beam fully solves hard multi-hop retrieval

## Relation To Oracle Headroom

Rough orientation only:
- `2Wiki-1000` oracle select at `K=100` gives `EM +0.124`
- this `2Wiki-100` beam pilot gives `EM +0.080`

Approximate fraction of full-dataset oracle headroom recovered:
- about `64.5%`

Important caveat:
- this is not a matched-slice comparison
- use it as intuition, not as a main-table statistic

## Decision

Current decision:
- keep `bridge_beam` as the main `2Wiki` non-oracle selector line
- keep `bridge_greedy` as the immediate search-ablation / control
- freeze the learned selector as pilot-only evidence unless later runs become much stronger
- treat `MuSiQue reserve3 + dedup` as improved but still not fully stable evidence
- treat `HotpotQA` as evidence that score calibration and prefix protection remain the next bottlenecks for shallow datasets

## Immediate Next Step

Run:
- retune the selector toward stronger top-prefix preservation or adaptive scoring on shallow datasets

Decision rule:
- keep the search claim narrow:
  - on `2Wiki`, beam search helps under a fixed bridge-aware score
  - on `MuSiQue`, stronger prefix preservation recovers a modest positive `EM`
  - on `HotpotQA`, the remaining failure is still over-exploration on shallow cases
- next method iteration should preserve anchors / shallow relevance more aggressively before adding bridge exploration
