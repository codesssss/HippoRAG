# Experiment Board

Last updated: 2026-04-22

## Canonical Outputs

- `2Wiki-1000 oracle sweep + reorder control`: `outputs_step0_general_2wikimultihopqa/eval_reports/oracle_select_sweep_1000_reorder20.json`
- `HotpotQA-1000 oracle sweep + reorder control`: `outputs_step0_general_hotpotqa/eval_reports/oracle_select_sweep_1000_reorder20.json`
- `MuSiQue-1000 oracle sweep + reorder control`: `outputs_step0_general_musique/eval_reports/oracle_select_sweep_1000.json`
- `2Wiki bridge analysis script`: `scripts/analyze_2wiki_oracle_ceiling.py`
- `Cross-dataset summary script`: `scripts/compile_oracle_select_summary.py`
- `Cross-dataset summary output`: `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.md`
- `Non-oracle pilot note`: `research_memory/emnlp_expand_then_compose/05_non_oracle_bridge_greedy.md`
- `Bridge-beam search memo`: `research_memory/emnlp_expand_then_compose/06_bridge_beam_search.md`
- `PCRS-RAG V1 branch memo`: `research_memory/emnlp_expand_then_compose/07_pcrs_rag_v1.md`
- `PCRS-RAG V2 parser/compiler spec`: `research_memory/emnlp_expand_then_compose/08_pcrs_v2_parser_compiler_spec.md`
- `Requirement cache builder`: `scripts/build_requirement_cache.py`
- `Requirement matcher trainer`: `scripts/train_requirement_setwise.py`
- `Requirement beam diagnostics script`: `scripts/analyze_requirement_beam_report.py`
- `MuSiQue requirement beam diagnostic note`: `research_memory/emnlp_expand_then_compose/musique_requirement_beam_reserve_diagnostics_20260402.md`
- `2Wiki strongest smoke note`: `outputs_step0_general_2wikimultihopqa/eval_reports/strongest_bridge_append_none_applypool_smoke40_20260416.md`
- `HotpotQA strongest GBC audit`: `outputs_step0_general_hotpotqa/eval_reports/strongest_gbc_e2e_audit_hotpotqa20_20260416.md`
- `Requirement-Aware Strongest backup memo`: `research_memory/emnlp_expand_then_compose/10_requirement_aware_strongest_backup.md`
- `Diversity selector baseline memo`: `research_memory/emnlp_expand_then_compose/11_diversity_selector_baselines_20260421.md`
- `DtC supplemental experiment memo`: `research_memory/emnlp_expand_then_compose/12_dtc_supplemental_experiments_20260421.md`
- `MuSiQue 4-doc DtC regression memo`: `research_memory/emnlp_expand_then_compose/13_musique_4doc_regression_analysis_20260421.md`
- `DtC NV full1000 + rank-prior direction memo`: `research_memory/emnlp_expand_then_compose/14_dtc_nv_full1000_rank_prior_20260422.md`

## Done

- Added `--oracle_select_k` support to `scripts/eval_causal_qwen3.py`
- Added oracle-select sweep support for multiple `K` values
- Added minimal full-support depth reporting
- Ran `2Wiki-100` oracle-select exploratory sweep
- Ran `2Wiki-1000` oracle-select sweep for `K=20,30,50,100`
- Ran `2Wiki-1000` reorder@20 control rerun
- Ran `HotpotQA-1000` oracle-select sweep for `K=20,30,50,100`
- Ran `HotpotQA-1000` reorder@20 control rerun
- Ran `MuSiQue-1000` oracle-select sweep for `K=20,30,50,100`
- Extended `scripts/analyze_2wiki_oracle_ceiling.py` to support anchor/bridge split metrics
- Ran `2Wiki` bridge vs anchor analysis and saved report files
- Created `scripts/compile_oracle_select_summary.py`
- Compiled the cross-dataset summary markdown/json
- Implemented a minimal non-oracle setwise selector: `bridge_greedy`
- Fixed missing structure-object preparation in the current `v2 + general_relation_graph` path
- Validated selector logic with targeted tests
- Ran `2Wiki` non-oracle pilot (`limit=20`, `pool_k=100`) with positive QA and recall gains
- Ran `2Wiki` non-oracle pilot on the canonical `legacy_fact_graph` backbone
- Ran `MuSiQue` non-oracle pilot on the canonical `legacy_fact_graph` backbone
- Tried one cheap selector tweak (`seed union + title dedup`) and reverted it after it hurt `2Wiki`
- Implemented a parallel `PCRS-RAG V1` branch with a new selector:
  - `requirement_beam`
- Added offline requirement-cache construction:
  - `scripts/build_requirement_cache.py`
- Added a lightweight requirement matcher training path:
  - `scripts/train_requirement_setwise.py`
- Added selector/test coverage for requirement support and counterfactual leakage scoring
- Added `setwise_selector_query_traces` export and reserve-policy ablation support for `requirement_beam`
- Ran `requirement_beam` Phase A oracle smoke on `2Wiki` and `MuSiQue`
- Ran `MuSiQue` reserve ablation for `requirement_beam` (`reserve3` vs `reserve1`)
- Added `scripts/analyze_requirement_beam_report.py` for offline requirement-beam diagnostics
- Diagnosed the current `MuSiQue` failure mode as upstream signal quality:
  - weak positive requirement construction
  - weak counterfactual construction
  - poor positive-vs-negative separation
  - near-collapsed leakage axis
- Added a full implementation spec for the next V2 parser/compiler rebuild:
  - `research_memory/emnlp_expand_then_compose/08_pcrs_v2_parser_compiler_spec.md`
- Migrated a clean strongest sidecar plus `GBC` guardrail into the current codebase
- Ran a positive `2Wiki` strongest apply-to-pool smoke and saved the summary note
- Ran a `HotpotQA` end-to-end strongest/GBC audit and recorded the query-level failure pattern
- Preserved `Requirement-Aware Strongest` as a documented backup method line
- Ran fixed-pool MMR/DPP diversity baselines at 1000-query scale on `2Wiki`, `HotpotQA`, and `MuSiQue`
- Implemented and tested `DtC-Embed` as a demand-aware fixed-pool evidence composition selector
- Ran aligned NV-Embed non-instruction DtC full1000 on all three datasets
- Ran hard-crossing pilot100 ablation and rejected it as the main fix
- Diagnosed deep-pool false positives as the next DtC scoring problem

## Running

- PropRAG aligned same-protocol comparison is running in `/mnt/nvme/code/PropRAG`:
  - script: `run_logs/run_proprag_aligned_qwenopenie_top100_8042_20260422.sh`
  - log: `run_logs/proprag_aligned_qwenopenie_top100_maxtok2048_nothinkopenie_propcap12_8042_20260422.log`
  - output root: `outputs_aligned_qwenopenie_top100_maxtok2048_nothinkopenie_propcap12`

## Next Wave

1. Implement rank-regularized DtC scoring:
   - add `--dtc_rank_weight`
   - use `normalized_rank = pool_position / max(pool_k - 1, 1)`
   - subtract `rank_weight * normalized_rank` in the greedy gain
2. Sweep `rank_weight in {0.1, 0.2, 0.3, 0.6, 1.0}` on pilot100 for all three datasets
3. Promote to full1000 only if the sweep preserves most wins while reducing losses
4. Keep hard-crossing as a negative ablation, not a main method
5. Finish the PropRAG aligned comparison after its 8B OpenIE assets finish building

## Method Priority

Method implementation order should be:

1. Active paper method candidate:
   - `DtC-Embed` under fixed-pool evidence composition
   - next variant: rank-regularized demand coverage
2. Baselines and controls:
   - baseline top-5
   - MMR/DPP structure-blind diversity
   - oracle select@100
   - PropRAG aligned comparison when assets complete
3. Backup line:
   - keep `Requirement-Aware Strongest` documented and runnable as a reserve option
   - do not promote it without at least `HotpotQA` baseline parity plus retained `2Wiki` gains
4. Do not jump to open iterative retrieval, online judges, or planner-style control before the fixed-pool DtC objective is fully tested

## Blockers

- Full1000 DtC is positive but still leaves most oracle headroom unused
- `2Wiki` full1000 gain is weak because wins and losses are nearly balanced
- Hard-crossing cuts wins more than losses, so the next fix must target scoring false positives rather than acceptance thresholds
- The strongest/GBC backup line is coherent, but it still lacks a positive shallow-dataset end-to-end result

## Do Not Drift

- Do not re-open candidate injection
- Do not re-open reader-side graph context
- Do not re-open causal-only framing
- Do not pitch this as a generic reranking paper
- Do not pitch DtC as open iterative retrieval; the protocol is fixed-pool post-retrieval composition
- Do not let the old unvalidated `PCRS-RAG V1` branch overwrite the active DtC line by default
- Do not let the current strongest/GBC backup line overwrite the paper mainline based on smoke-scale evidence
