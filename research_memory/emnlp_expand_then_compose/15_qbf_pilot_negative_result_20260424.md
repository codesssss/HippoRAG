# QBF Pilot Negative Result - 2026-04-24

## Context

This note records the pilot implementation and result for the **QBF (Query-conditioned Backward Flow)** branch.

The branch tested whether HippoRAG-style graph retrieval could be improved by adding a terminal-schema backward potential:

- infer a terminal evidence schema from the question, for example `country`, `date`, `birthplace`
- score facts by terminal-schema compatibility `chi`
- propagate terminal potential backward over the fact-entity graph
- rank documents by query relevance times backward potential

The intended paper-level claim was:

> PPR is a degenerate case of QBF with uniform terminal potential and uniform edge compatibility.

The pilot was designed as a go/no-go test before investing in full-test or paper framing.

## Implementation Artifacts

Code:

- `src/hipporag/qbf.py`
- `scripts/eval_qbf_pilot.py`
- `tests/test_qbf.py`

Result files:

- `run_logs/qbf_pilot_20260424/2wikimultihopqa_pilot100.json`
- `run_logs/qbf_pilot_20260424/2wikimultihopqa_pilot100_ppr_qbf_rerank.json`
- `run_logs/qbf_pilot_20260424/musique_pilot100.json`
- `run_logs/qbf_pilot_20260424/musique_pilot100_ppr_qbf_rerank.json`

Validation:

- `python -m pytest tests/test_qbf.py`
- Result: `5 passed`

Important protocol note:

- The `ppr` baseline in `scripts/eval_qbf_pilot.py` is the no-LLM fact-graph retrieval route, traced as `fact_graph_no_llm_rerank`.
- This is intentional for operator isolation and speed.
- It is not the full QA reader pipeline and should not be compared directly to end-to-end QA numbers.

## Tested Variants

| Variant | Description |
|---|---|
| `ppr` | Current fact-graph PPR baseline, no LLM reranking |
| `ppr_chi_rerank` | Preserve PPR top-100 pool, rerank by terminal schema score `chi` |
| `ppr_qbf_rerank` | Preserve PPR top-100 pool, rerank by QBF document score |
| `qbf_schema_relational` | Raw QBF ranking over relation edges only |
| `qbf_schema_all_edge` | Raw QBF ranking with doc containment nodes included |
| `qbf_random_chi` | Raw QBF with deterministic random terminal signal |
| `qbf_oracle_relation` | Raw QBF with an oracle relation phrase derived from gold supporting facts |

## Main Results

### 2Wiki pilot100

| Variant | R@5 | R@20 | R@100 | SC@5 | SC@20 | SC@100 | Mean full-support depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ppr` | 0.8150 | 0.8950 | 0.9300 | 0.5600 | 0.7300 | 0.8100 | 8.75 |
| `ppr_chi_rerank` | 0.1800 | 0.3300 | 0.9300 | 0.1300 | 0.2000 | 0.8100 | 50.26 |
| `ppr_qbf_rerank` | 0.5450 | 0.6950 | 0.9300 | 0.2800 | 0.4200 | 0.8100 | 31.19 |
| `qbf_schema_relational` | 0.4675 | 0.5600 | 0.6550 | 0.2100 | 0.3000 | 0.3800 | 16.16 |
| `qbf_schema_all_edge` | 0.4600 | 0.5325 | 0.6225 | 0.2100 | 0.2800 | 0.3500 | 15.54 |
| `qbf_random_chi` | 0.5425 | 0.6525 | 0.7525 | 0.3000 | 0.3800 | 0.5000 | 15.68 |
| `qbf_oracle_relation` | 0.6800 | 0.7550 | 0.8450 | 0.3900 | 0.5000 | 0.6500 | 14.97 |

### MuSiQue pilot100

| Variant | R@5 | R@20 | R@100 | SC@5 | SC@20 | SC@100 | Mean full-support depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ppr` | 0.6517 | 0.8300 | 0.9475 | 0.3500 | 0.6000 | 0.8400 | 19.01 |
| `ppr_chi_rerank` | 0.1217 | 0.2450 | 0.9475 | 0.0200 | 0.0800 | 0.8400 | 63.99 |
| `ppr_qbf_rerank` | 0.4350 | 0.6100 | 0.9475 | 0.1300 | 0.3200 | 0.8400 | 36.32 |
| `qbf_schema_relational` | 0.3908 | 0.5408 | 0.7108 | 0.1100 | 0.2600 | 0.4100 | 21.29 |
| `qbf_schema_all_edge` | 0.3825 | 0.5350 | 0.7025 | 0.1100 | 0.2500 | 0.4100 | 23.78 |
| `qbf_random_chi` | 0.3983 | 0.5608 | 0.7650 | 0.0900 | 0.2400 | 0.5200 | 32.77 |
| `qbf_oracle_relation` | 0.5583 | 0.7350 | 0.8925 | 0.2600 | 0.4700 | 0.7100 | 22.00 |

## Interpretation

The result is a clear negative.

### 1. Terminal schema `chi` is actively harmful

`ppr_chi_rerank` preserves the same PPR top-100 pool, so R@100 is unchanged.

However, top-5 and top-20 quality collapse:

- 2Wiki SC@5: `0.5600 -> 0.1300`
- MuSiQue SC@5: `0.3500 -> 0.0200`

This means the terminal-schema sink is not a useful rank signal for reader-ready evidence selection.

### 2. Raw QBF destroys pool recall

Raw QBF does not just reorder PPR candidates. It creates a new graph ranking.

That ranking loses many support documents:

- 2Wiki R@100: `0.9300 -> 0.6550`
- MuSiQue R@100: `0.9475 -> 0.7108`

This invalidates QBF as a replacement retrieval operator in the current HippoRAG graph substrate.

### 3. PPR plus QBF rerank is safer but still much worse than PPR

The `ppr_qbf_rerank` control keeps PPR R@100 intact and only changes ordering.

It still substantially hurts top-5 support:

- 2Wiki SC@5: `0.5600 -> 0.2800`
- MuSiQue SC@5: `0.3500 -> 0.1300`

So the issue is not only raw-pool destruction. The QBF document score itself is not aligned with support-complete top-5 selection.

### 4. Random `chi` is competitive with or better than schema `chi`

On both datasets, `qbf_random_chi` is close to or better than `qbf_schema_relational`.

This is the strongest evidence against the current terminal-schema design:

- If schema `chi` encoded useful information, it should dominate random `chi`.
- It does not.

### 5. Oracle relation does not rescue QBF

`qbf_oracle_relation` improves over deterministic schema QBF but remains below PPR:

- 2Wiki SC@5: PPR `0.5600`, oracle-QBF `0.3900`
- MuSiQue SC@5: PPR `0.3500`, oracle-QBF `0.2600`

Therefore, the problem is not just a weak deterministic schema parser. The operator and graph propagation semantics are misaligned with the target retrieval objective.

## Decision

Stop QBF as a main method line.

Do not run QBF full1000.

Do not spend more time tuning:

- terminal schema vocabulary
- edge-type weighting
- alpha / horizon
- psi floor
- raw QBF vs all-edge QBF

The pilot failure is operator-level, not a small hyperparameter miss.

## Paper Implication

QBF should not be framed as the end-to-end graph-retrieval contribution.

At most, it can be used internally or in an appendix as a negative result:

> We tested a PPR generalization based on terminal-schema backward flow, but found that terminal-schema potentials were not aligned with reader-ready multi-hop evidence selection. This further supports treating multi-hop QA as a set composition problem rather than a single graph-ranking problem.

The main research line should return to:

1. fixed-pool composition;
2. PropRAG-pool + DAEC/DtC transfer;
3. oracle select@100 on strong pools;
4. failure taxonomy, especially wrong-entity binding and reader interference.

## Current Follow-up State

As of this note, the following full1000 jobs are still running:

- `PropRAG pool + DAEC/DtC` on 2Wiki
- `PropRAG pool + DAEC/DtC` on HotpotQA
- `PropRAG pool + DAEC/DtC` on MuSiQue

Logs:

- `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa.log`
- `run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa.log`
- `run_logs/layer1_proprag_pool_eval_fixed_20260424/musique.log`

Expected outputs:

- `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- `run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json`
- `run_logs/layer1_proprag_pool_eval_fixed_20260424/musique_proprag_pool_daec_oracle.json`
