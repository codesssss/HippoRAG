# Experiment Board

Last updated: 2026-04-25

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
- `DtC satisfiable_by prompt sweep memo`: `research_memory/emnlp_expand_then_compose/satisfiable_by_prompt_sweep_20260424.md`
- `QBF pilot negative result memo`: `research_memory/emnlp_expand_then_compose/15_qbf_pilot_negative_result_20260424.md`
- `Layer-1 retriever-agnostic composition memo`: `research_memory/emnlp_expand_then_compose/16_layer1_retriever_agnostic_composition_20260424.md`
- `Layer-1 follow-up taxonomy/significance memo`: `research_memory/emnlp_expand_then_compose/17_layer1_followup_taxonomy_significance_20260424.md`
- `QBF pilot implementation`: `src/hipporag/qbf.py`
- `QBF pilot evaluator`: `scripts/eval_qbf_pilot.py`
- `QBF pilot outputs`: `run_logs/qbf_pilot_20260424/`
- `PropRAG-pool + DAEC/DtC full1000 outputs`: `run_logs/layer1_proprag_pool_eval_fixed_20260424/`
- `Dense-pool + DAEC/DtC full1000 outputs`: `run_logs/layer1_dense_pool_eval_20260424/`
- `2Wiki PropRAG-pool DAEC/DtC ablation outputs`: `run_logs/layer1_proprag_pool_ablation_2wiki_20260424/`
- `Layer-1 failure taxonomy outputs`: `run_logs/failure_taxonomy_layer1_20260424/`
- `Layer-1 paired significance outputs`: `run_logs/layer1_significance_20260424/`
- `Layer-1 targeted nobinding ablation outputs`: `run_logs/layer1_targeted_nobinding_20260424/`
- `BSGS Week0/Week1 execution plan`: `docs/bsgs_week0_week1_plan_20260425.md`
- `BSGS Week0/Week1 research memo`: `research_memory/emnlp_expand_then_compose/18_bsgs_week0_week1_plan_20260425.md`

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
- Ran the `satisfiable_by` prompt/policy sweep and kept the field optional:
  - default main experiments keep the old prompt via `--dtc_include_satisfiable_by=false`
  - `satisfiable_by` remains available for diagnostics/appendix runs
- Implemented and tested the QBF retrieval-only pilot:
  - `src/hipporag/qbf.py`
  - `scripts/eval_qbf_pilot.py`
  - `tests/test_qbf.py`
- Ran QBF pilot100 on `2Wiki` and `MuSiQue`
- Rejected QBF as a main method line: raw QBF destroys pool recall, PPR+QBF rerank still hurts top-5 support, random-chi is competitive with schema-chi, and oracle-relation QBF remains below PPR
- Ran PropRAG-pool + DAEC/DtC full1000 on `2Wiki`, `HotpotQA`, and `MuSiQue`
- Consolidated PropRAG-pool oracle select@100 in the same full1000 reports
- Ran dense `NV-Embed` top-100 pool + DAEC/DtC full1000 on `2Wiki`, `HotpotQA`, and `MuSiQue`
- Ran 2Wiki PropRAG-pool internal ablations:
  - `nobinding`
  - `norepairtyping`
  - `norank`
- Recorded the Layer-1 retriever-agnostic composition memo:
  - `research_memory/emnlp_expand_then_compose/16_layer1_retriever_agnostic_composition_20260424.md`
- Fixed and reran `scripts/analyze_failure_taxonomy.py` for the six completed Layer-1 full1000 reports
- Added and ran paired bootstrap/sign-flip significance analysis:
  - `scripts/analyze_layer1_significance.py`
  - `run_logs/layer1_significance_20260424/`
- Ran targeted cross-pool `nobinding` ablations:
  - launcher: `run_logs/run_layer1_targeted_nobinding_20260424.sh`
  - output root: `run_logs/layer1_targeted_nobinding_20260424/`
  - completed pairs:
    - `HotpotQA x PropRAG x nobinding`
    - `MuSiQue x PropRAG x nobinding`
    - `2Wiki x Dense x nobinding`
    - `MuSiQue x Dense x nobinding`
- Recorded the Layer-1 follow-up memo:
  - `research_memory/emnlp_expand_then_compose/17_layer1_followup_taxonomy_significance_20260424.md`
- Froze the DAEC Layer-1 code/docs checkpoint:
  - `2555018 Add DAEC layer1 external-pool evaluation`
- Created the BSGS exploration branch:
  - `bsgs-week0-week1`
- Recorded the BSGS Week0/Week1 execution boundary:
  - `docs/bsgs_week0_week1_plan_20260425.md`
  - `research_memory/emnlp_expand_then_compose/18_bsgs_week0_week1_plan_20260425.md`

## Running

- No active Layer-1 DtC/DAEC evaluation jobs are expected to be running.
- No BSGS jobs have been launched yet on this branch.

## Next Wave

1. Implement and run the BSGS split audit:
   - `scripts/bsgs_split_audit.py`
   - first required outputs: `reports/week0/split_audit.md`, `reports/week0/split_audit.json`
2. Run the rest of Week 0 only after the split audit is clear:
   - IRCoT baseline
   - DeBERTa-NLI calibration and throughput
   - oracle slot pipeline
   - Qwen slot quality evaluation
3. Generate the Week 0 decision report:
   - `reports/week0/decision.md`
4. Run Week 1 oracle-slot BSGS only if Week 0 is usable:
   - `scripts/bsgs_run_oracle.py`
   - `reports/week1/operator_validation.md`
5. Keep the DAEC paper floor frozen at `2555018`; BSGS must not rewrite the DAEC main path.
6. Keep the DAEC paper-writing line available:
   - manually audit `MuSiQue` loss cases from Layer-1 taxonomy outputs
   - consolidate oracle-on-pools tables
   - start paper skeleton around retriever-agnostic fixed-pool composition

## Method Priority

Method implementation order should be:

1. Frozen paper-floor method:
   - `DtC/DAEC` as a retriever-agnostic fixed-pool evidence compositor
   - strongest supported component so far: dependency binding across tested PropRAG and dense pool settings
2. New exploratory method branch:
   - `BSGS` as multi-step node-marginal belief-state filtering
   - Week 0 is risk audit only
   - Week 1 is oracle-slot operator validation only
   - do not promote BSGS to a full method until the mechanism gate passes
3. Baselines and controls:
   - baseline top-5
   - MMR/DPP structure-blind diversity
   - oracle select@100
   - PropRAG aligned comparison
   - dense-pool comparison
   - IRCoT-style iterative baseline for BSGS
4. Backup line:
   - keep `Requirement-Aware Strongest` documented and runnable as a reserve option
   - do not promote it without at least `HotpotQA` baseline parity plus retained `2Wiki` gains
5. Do not jump to latent-slot, soft-binding, DPP, or full proposition-graph extraction before Week 1 BSGS oracle-slot validation passes
6. Do not continue QBF unless a new graph substrate or a fundamentally different supervision signal is introduced

## Blockers

- Full1000 DtC is positive but still leaves most oracle headroom unused
- `MuSiQue` has large oracle headroom but low DAEC/DtC gap recovery, so selector/scoring remains the main weakness there
- Targeted cross-pool `nobinding` supports dependency binding as a main component, but rank prior and repair typing are not yet supported as necessary components
- Hard-crossing cuts wins more than losses, so hard per-candidate gates remain rejected
- The strongest/GBC backup line is coherent, but it still lacks a positive shallow-dataset end-to-end result
- QBF terminal-schema backward flow fails as a retrieval operator on 2Wiki and MuSiQue pilot100; the issue appears operator-level rather than hyperparameter-level
- BSGS is blocked until MuSiQue split audit confirms whether current evaluation data is `MuSiQue-Ans` or `MuSiQue-Full`
- BSGS is blocked until IRCoT, NLI calibration, oracle slots, and Qwen slot quality are audited in Week 0
- BSGS absorbing transition is blocked unless DeBERTa-NLI calibration reaches the ECE threshold in the Week 0 report

## Do Not Drift

- Do not re-open candidate injection
- Do not re-open reader-side graph context
- Do not re-open causal-only framing
- Do not pitch this as a generic reranking paper
- Do not pitch DtC as open iterative retrieval; the protocol is fixed-pool post-retrieval composition
- Do not let the old unvalidated `PCRS-RAG V1` branch overwrite the active DtC line by default
- Do not let the current strongest/GBC backup line overwrite the paper mainline based on smoke-scale evidence
- Do not implement BSGS latent slots, soft binding, or DPP in Week 1
- Do not use an LLM verifier for BSGS Week 0/1
- Do not mix `MuSiQue-Full` and `MuSiQue-Ans` baselines
- Do not write oracle-slot BSGS as a non-oracle main result
- Do not frame DAEC and BSGS as the same probability space
