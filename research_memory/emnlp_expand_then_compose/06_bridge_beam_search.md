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

## Interpretation

What this means:
- The bridge-aware selector is now clearly useful on the canonical `2Wiki` setting.
- Search matters on top of scoring: `beam` beats `greedy` under the same local score.
- The gain is not explained by shallow recall alone.

Why the last point matters:
- `bridge_greedy` has slightly higher `Recall@5` than `bridge_beam` (`0.8150` vs `0.8075`)
- `Recall@20` is identical (`0.8825`)
- but `bridge_beam` still wins on QA, especially on `4-doc` queries

Conclusion:
- `bridge_beam` improves evidence composition, not just top-5 lexical coverage.
- This is exactly the kind of evidence needed for the `Expand-then-Compose` story.
- But today this conclusion is solid only on canonical `2Wiki`; `MuSiQue` is currently a robustness failure case.

## Paper Positioning

Recommended framing:
- present `bridge_beam` as the strongest current practical candidate on `2Wiki`
- present `bridge_greedy` as the search-ablation control
- make the claim narrow and defensible:
  - widening the pool is necessary but not sufficient
  - under a fixed bridge-aware score, beam search recovers better evidence sets than greedy selection

What should not be claimed yet:
- do not claim broad cross-dataset robustness
- do not claim that the current score generalizes from `2Wiki` to `MuSiQue`
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
- treat `MuSiQue` as an active stress test, not as solved evidence

## Immediate Next Step

Run:
- `MuSiQue bridge_greedy@100` on the same canonical backbone and score weights

Decision rule:
- if `beam` still beats matched `greedy`, keep the search claim but narrow it to:
  - search helps under a fixed bridge-aware score on hard cases
  - cross-dataset score calibration remains open
- if matched `greedy` is better than `beam`, demote `beam` to a failed search variant and focus the paper on diagnosis plus a more conservative selector
