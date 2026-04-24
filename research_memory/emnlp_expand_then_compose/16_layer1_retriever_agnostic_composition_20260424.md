# Layer-1 Retriever-Agnostic Composition Results

Date: 2026-04-24

## Purpose

This memo records the completed Layer-1 evidence for the fixed-pool evidence composition paper line.

The tested question was:

> Does demand-aware evidence composition improve final top-5 reader utility across different top-100 retrieval pools, including a strong proposition retriever and a non-graph dense retriever?

The answer is yes. `DtC/DAEC` improves all three datasets on both `PropRAG` top-100 pools and dense `NV-Embed` top-100 pools.

This is the key result that separates the current paper line from a HippoRAG-specific selector story.

## Protocol

Common protocol:

- Reader: frozen `qwen3-8b-train`, no-think evaluation protocol.
- Reader context budget: `qa_top_k=5`.
- Reader document truncation: `qa_doc_max_chars=2048`.
- Embedding model: `VLLM/nvidia/NV-Embed-v2`.
- Embedding endpoint: `http://localhost:8019/v1/embeddings`.
- Candidate pool budget: top-100 external pool.
- Selector: `dtc_embed`.
- Oracle upper bound: `--oracle_select_k 100`.
- External pool strict alignment enabled by default.

Current full `DtC/DAEC` configuration:

- `--setwise_selector dtc_embed`
- `--setwise_pool_k 100`
- `--setwise_reserve_top_m 0`
- `--dtc_rank_weight 0.2`
- `--dtc_include_satisfiable_by true`
- `--dtc_repairable_filter_enabled true`
- `--dtc_satisfiable_by_policy binding_override`
- `--dtc_enable_dependency_binding true`
- `--dtc_binding_max_candidates 4`
- `--dtc_binding_entity_hit_required true`

## Output Roots

PropRAG-pool full1000:

- `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`
- `run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json`
- `run_logs/layer1_proprag_pool_eval_fixed_20260424/musique_proprag_pool_daec_oracle.json`

Dense-pool full1000:

- `run_logs/layer1_dense_pool_eval_20260424/2wikimultihopqa_dense_pool_daec_oracle.json`
- `run_logs/layer1_dense_pool_eval_20260424/hotpotqa_dense_pool_daec_oracle.json`
- `run_logs/layer1_dense_pool_eval_20260424/musique_dense_pool_daec_oracle.json`

2Wiki PropRAG-pool internal ablations:

- `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/2wikimultihopqa_nobinding.json`
- `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/2wikimultihopqa_norepairtyping.json`
- `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/2wikimultihopqa_norank.json`

Summary command:

```bash
.venv-hipporag/bin/python scripts/summarize_layer1_results.py run_logs/layer1_proprag_pool_eval_fixed_20260424
.venv-hipporag/bin/python scripts/summarize_layer1_results.py run_logs/layer1_dense_pool_eval_20260424
.venv-hipporag/bin/python scripts/summarize_layer1_results.py run_logs/layer1_proprag_pool_ablation_2wiki_20260424
```

## Main Result 1: PropRAG Pool + DtC/DAEC

This is the most important substrate experiment. It tests whether a strong proposition-level retriever still leaves composition headroom.

| Dataset | Base EM | Base F1 | DAEC EM | DAEC F1 | Delta EM | Delta F1 | Oracle F1 | Oracle Gap | R@5 | R@20 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.5750 | 0.6457 | 0.6070 | 0.6810 | +0.0320 | +0.0353 | 0.7311 | 0.0854 | 0.9028 | 0.9607 | 0.9872 |
| HotpotQA | 0.5950 | 0.7227 | 0.6060 | 0.7348 | +0.0110 | +0.0121 | 0.7697 | 0.0470 | 0.9500 | 0.9925 | 0.9990 |
| MuSiQue | 0.3300 | 0.4266 | 0.3390 | 0.4404 | +0.0090 | +0.0138 | 0.6019 | 0.1753 | 0.7131 | 0.8942 | 0.9677 |

Approximate F1 oracle-gap recovery:

| Dataset | Delta F1 | Oracle Gap | Gap Recovery |
|---|---:|---:|---:|
| 2Wiki | +0.0353 | 0.0854 | 41.3% |
| HotpotQA | +0.0121 | 0.0470 | 25.7% |
| MuSiQue | +0.0138 | 0.1753 | 7.9% |

Interpretation:

- Composition headroom remains after PropRAG, especially on `2Wiki` and `MuSiQue`.
- `DtC/DAEC` improves all three datasets under the same no-think reader protocol.
- The `MuSiQue` pool has a large oracle gap but low recovery, which points to selector/scoring limits rather than pool absence.
- This result supports the paper claim that evidence composition is not merely a HippoRAG artifact.

## Main Result 2: Dense Pool + DtC/DAEC

This tests whether the composition gain transfers to a non-graph retrieval pool.

| Dataset | Base EM | Base F1 | DAEC EM | DAEC F1 | Delta EM | Delta F1 | Oracle F1 | Oracle Gap | R@5 | R@20 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.4550 | 0.4984 | 0.5060 | 0.5628 | +0.0510 | +0.0644 | 0.6396 | 0.1412 | 0.7238 | 0.7990 | 0.8770 |
| HotpotQA | 0.5950 | 0.7106 | 0.6140 | 0.7326 | +0.0190 | +0.0220 | 0.7669 | 0.0563 | 0.9305 | 0.9830 | 0.9925 |
| MuSiQue | 0.2980 | 0.3896 | 0.3160 | 0.4138 | +0.0180 | +0.0242 | 0.5521 | 0.1625 | 0.6628 | 0.8197 | 0.9055 |

Approximate F1 oracle-gap recovery:

| Dataset | Delta F1 | Oracle Gap | Gap Recovery |
|---|---:|---:|---:|
| 2Wiki | +0.0644 | 0.1412 | 45.6% |
| HotpotQA | +0.0220 | 0.0563 | 39.1% |
| MuSiQue | +0.0242 | 0.1625 | 14.9% |

Interpretation:

- Dense-only pools have weaker top-5 baselines and larger composition headroom.
- `DtC/DAEC` improves all three dense-pool runs, with the largest absolute gain on `2Wiki`.
- This is the strongest evidence that the method is retriever-agnostic: gains transfer from graph/proposition pools to a non-graph dense pool.

## Internal Ablation: 2Wiki PropRAG Pool

Full method reference:

- `run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json`

Ablations:

- `nobinding`: disables dependency binding while preserving repair typing and rank prior.
- `norepairtyping`: disables the repairable requirement filter while keeping binding and rank prior.
- `norank`: sets `dtc_rank_weight=0.0` while keeping binding and repair typing.

| Variant | Base EM | Base F1 | Selector EM | Selector F1 | Delta EM | Delta F1 | Selector R@5 | Selector R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.5750 | 0.6457 | 0.6070 | 0.6810 | +0.0320 | +0.0353 | 0.9355 | 0.9675 |
| `nobinding` | 0.5750 | 0.6457 | 0.5900 | 0.6599 | +0.0150 | +0.0142 | 0.9125 | 0.9613 |
| `norepairtyping` | 0.5750 | 0.6457 | 0.6110 | 0.6841 | +0.0360 | +0.0384 | 0.9357 | 0.9680 |
| `norank` | 0.5750 | 0.6457 | 0.6110 | 0.6846 | +0.0360 | +0.0389 | 0.9353 | 0.9673 |

Interpretation:

- Dependency binding is load-bearing on `2Wiki`.
- Removing binding reduces Delta F1 from `+0.0353` to `+0.0142`.
- The drop is `-0.0211 F1`, which is a meaningful component contribution.
- Repair typing is not load-bearing on `2Wiki` under the current PropRAG-pool setup.
- Removing the repairable filter slightly improves F1: `0.6810 -> 0.6841`.
- Rank prior is also not load-bearing on `2Wiki` under this setup.
- Removing rank regularization slightly improves F1: `0.6810 -> 0.6846`.

## Current Research Claim

The Layer-1 claim is now strong enough to support a fixed-pool composition paper:

> Evidence composition improves reader utility across both strong graph/proposition retrieval pools and dense-only pools. The main remaining gap is not whether composition helps, but which method components are actually necessary and why composition still fails on residual cases.

This is a stronger claim than the earlier HippoRAG-only story.

The paper-safe wording should avoid saying that the current method is a complete end-to-end retriever. It is a retriever-agnostic fixed-pool compositor.

## Method Implications

What is supported:

- `DtC/DAEC` as a fixed-pool compositor.
- Retriever-agnostic transfer across `PropRAG` and dense `NV-Embed` pools.
- Dependency binding as an important component, at least on `2Wiki`.
- Oracle select@100 as a real upper bound showing remaining composition headroom.

What is not yet supported:

- Rank regularization as a necessary core component.
- Repair typing as a necessary core component.
- End-to-end `DAPG` as the main method.
- `QBF` as a viable retrieval operator.

## Remaining Gaps

Priority order:

- Run failure taxonomy over the full1000 outputs: binding failure, wrong-entity distractor, gold pushed out, reader interference with gold still present.
- Decide the paper-facing main configuration. The current full config is conservative and positive across all pools, but the 2Wiki ablation says `rank_weight=0.2` and repair typing should not be overclaimed.
- Run cross-dataset component ablations if component-level claims are needed. At minimum, run `nobinding`, `norepairtyping`, and `norank` on `MuSiQue` pilot/full.
- Consolidate an oracle-on-pools table: HippoRAG/NV baseline pools from earlier reports, PropRAG top-100 pool, and dense top-100 pool.
- Keep Layer-2/DAPG as optional future work unless proposition-level local graph shows a clear incremental gain over document-level DtC.

## Decision

Use Layer-1 as the current paper floor.

Do not spend more time on QBF or end-to-end graph retrieval until the failure taxonomy and component ablations are documented. The current strongest contribution is:

- a diagnosis that multi-hop QA has a fixed-pool evidence composition bottleneck;
- a retriever-agnostic compositor that improves multiple top-100 pool sources;
- mechanism analysis showing why generic diversity and some hard gates fail.
