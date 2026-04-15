# CE Local Repair Smoke Summary (2026-04-13)

## Scope

- Method: `bridge_append_plus_ce_local_repair`
- Setting: 100-query smoke, `qa_top_k=5`, `append_max_docs=3`, `expand_base_k=10`, `ce_device=cuda:7`
- Datasets:
  - `musique`
  - `2wikimultihopqa`
  - `hotpotqa`

## Headline Results

| Dataset | baseline_top10_plus_ce | bridge_append_plus_ce | ce_local_repair | Delta vs baseline | Delta vs bridge |
|---|---:|---:|---:|---:|---:|
| musique | 0.3100 / 0.3698 | 0.3200 / 0.3810 | 0.3200 / 0.3749 | +0.0100 / +0.0051 | +0.0000 / -0.0061 |
| 2wikimultihopqa | 0.4200 / 0.4553 | 0.4300 / 0.4728 | 0.4100 / 0.4438 | -0.0100 / -0.0115 | -0.0200 / -0.0290 |
| hotpotqa | 0.6200 / 0.7173 | 0.6200 / 0.7173 | 0.6200 / 0.7173 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |

Numbers are `EM / F1`.

## Retrieval For `ce_local_repair`

| Dataset | R@5 | R@20 | R@100 |
|---|---:|---:|---:|
| musique | 0.6325 | 0.7858 | 0.8875 |
| 2wikimultihopqa | 0.7750 | 0.8675 | 0.8975 |
| hotpotqa | 0.9350 | 0.9650 | 1.0000 |

## Repair Diagnostics

| Dataset | Repair Applied Query Rate | Positive Req Swap Rate | Avg CE Drop vs Scaffold |
|---|---:|---:|---:|
| musique | 18.0% | 18.0% | 3.2454 |
| 2wikimultihopqa | 5.0% | 5.0% | -1.0876 |
| hotpotqa | 8.0% | 8.0% | 4.7931 |

## Dataset-Specific Notes

### MuSiQue

- Repair triggered on a nontrivial fraction of queries (`18%`).
- It preserved the EM gain over `baseline_top10_plus_ce`, but did not improve over plain `bridge_append_plus_ce`.
- Proxy had partial signal, but swaps still came with notable CE degradation on average.

### 2WikiMultiHopQA

- Repair almost never triggered (`5%`).
- When it did trigger, it was usually wrong:
  - `proxy_positive_and_oracle_positive = 0`
  - `applied_swap_mean_delta_em = -0.2`
  - `applied_swap_mean_delta_f1 = -0.2309`
- This side branch is harmful on current 2Wiki smoke.

### HotpotQA

- Repair triggered occasionally (`8%`), but produced zero QA gain.
- Final result is exactly tied with both existing controls.
- This is consistent with Hotpot already being saturated under the current top-10 + CE scaffold.

## Overall Takeaway

- `ce_local_repair` is **not** a generally winning side branch.
- Best case is MuSiQue: small signal, but still not above existing `bridge_append_plus_ce`.
- 2Wiki is negative.
- Hotpot is neutral.

## Related Analysis Files

- `run_logs/musique_ce_local_repair_proxy_20260413.md`
- `run_logs/2wiki_ce_local_repair_proxy_20260413.md`
- `run_logs/hotpotqa_ce_local_repair_proxy_20260413.md`
