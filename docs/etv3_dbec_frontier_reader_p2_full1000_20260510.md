# ETv3 + DBEC Frontier Reader P2 Full1000

Date: 2026-05-10

## Scope

This note records the reader-level follow-up to the selector-only preservation /
admission frontier in `docs/etv3_dbec_selector_frontier_full1000_20260510.md`.

P1 already showed that selector-level title completeness is not monotonic in the
preservation strength.  P2 only runs the two most informative reader points:

1. `2Wiki top3/max2`, because it is the only point where selector-level
   title-all@5 beats the robust `top4/max1` point.
2. `MuSiQue top2/max3`, because it tests whether a clearly looser preservation
   point also hurts the reader, rather than only increasing gold-out counts.

The goal is not to sweep reader QA over the whole frontier.  The goal is to test
whether selector-level set completeness is sufficient to choose the operating
point.

## Setup

Both runs materialize frozen selector outputs as external reader pools.  The
first five documents in each pool are exactly the P1 selector top-5 documents;
the remaining pool order is preserved for reader-run compatibility.

| Item | Setting |
| --- | --- |
| Base method | `evidence_transition_graphragv3_variable_flow` |
| Selector signal | DBEC / `daec_noisyor_safe_llm` traces from P1 |
| Reader | Qwen3-8B, no-thinking |
| Embedding | NV-Embed-v2 |
| QA top-k | `5` |
| Selector recomputation during reader QA | No |
| Causal graph reader path | Disabled |
| Export script | `scripts/export_selector_frontier_reader_pool.py` |

The exported reader pools were verified to match the P1 selector top-5 titles
for all `1000 / 1000` queries.

## Outputs

| Dataset | Variant | Reader pool | QA output |
| --- | --- | --- | --- |
| 2Wiki | `top3/max2` | `run_logs/etv3_dbec_frontier_reader_p2_full1000_20260510/pools/2wikimultihopqa_etv3_pool100_dbec_top3_max2_reader_pool_limit1000.json` | `run_logs/etv3_dbec_frontier_reader_p2_full1000_20260510/evals/2wikimultihopqa_etv3_pool100_dbec_top3_max2_reader_qa_limit1000.json` |
| MuSiQue | `top2/max3` | `run_logs/etv3_dbec_frontier_reader_p2_full1000_20260510/pools/musique_etv3_pool100_dbec_top2_max3_reader_pool_limit1000.json` | `run_logs/etv3_dbec_frontier_reader_p2_full1000_20260510/evals/musique_etv3_pool100_dbec_top2_max3_reader_qa_limit1000.json` |

## Reader Results

| Dataset | Variant | Selector title-all@5 | Reader R@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2Wiki | ETv3 baseline | 0.7060 | 0.9015 | 0.5860 | 0.6516 |
| 2Wiki | stable DBEC `top1/max2` | 0.8200 | 0.9307 | 0.6140 | 0.6867 |
| 2Wiki | `top4/max1` | 0.8560 | 0.9503 | 0.6240 | 0.7005 |
| 2Wiki | `top3/max2` | **0.8790** | **0.9557** | **0.6310** | **0.7081** |
| MuSiQue | ETv3 baseline | 0.4600 | 0.7391 | 0.3330 | 0.4332 |
| MuSiQue | stable DBEC `top1/max2` | 0.4520 | 0.7322 | 0.3000 | 0.3989 |
| MuSiQue | `top4/max1` | **0.4850** | **0.7551** | **0.3430** | **0.4468** |
| MuSiQue | `top3/max2` | 0.4810 | 0.7548 | 0.3340 | 0.4379 |
| MuSiQue | `top2/max3` | 0.4620 | 0.6972 | 0.3120 | 0.4037 |

## Immediate Read

P2 answers the critical reader question in two different directions.

For 2Wiki, selector-level and reader-level agree: relaxing preservation from
`top4/max1` to `top3/max2` improves both title-all@5 (`0.8560 -> 0.8790`) and
reader F1 (`0.7005 -> 0.7081`).  Therefore `top4/max1` is not a universal
reader-optimal point.  2Wiki can safely use a looser admission policy, consistent
with the earlier finding that DBEC has real value on title-identifiable deep
chains.

For MuSiQue, looser preservation damages the reader.  `top2/max3` has
selector-level title-all@5 near the ETv3 baseline (`0.4620` vs `0.4600`) but
reader F1 falls far below the robust `top4/max1` point (`0.4037` vs `0.4468`) and
approaches unrestricted stable DBEC (`0.3989`).  This means set membership alone
is not sufficient: ordering, context coherence, and preservation of
reader-useful evidence matter.

## Conclusion

The P1/P2 combination supports a stronger claim than a fixed `top4` rule:

> ETv3-pool composition has a dataset-dependent preservation/admission frontier.
> Selector-level set completeness is necessary but not sufficient for reader
> performance; reader-facing composition must preserve high-confidence context
> while admitting residual evidence.

Operationally, `top4/max1` remains the robust parameter-level floor because it
is positive on all three datasets and avoids MuSiQue collapse.  However, the
2Wiki `top3/max2` result shows that a future ETv4-composition method should not
hard-code `top4`; it should replace rank-cutoff preservation with an
evidence-level retention/admission rule if it wants to safely admit more than
one residual document.

This still does not prove a state-binding mechanism.  It supports
preservation-constrained residual evidence admission as the ETv4-composition
design prior.
