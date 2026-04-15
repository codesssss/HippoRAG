# Experiment Board

Last updated: 2026-04-14

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
- `Baseline qa_top_k sweep summary`: `run_logs/baseline_qa_topk_sweep_20260406.summary.md`
- `Bridge-append matrix summary`: `run_logs/bridge_append_matrix_20260406.summary.md`
- `Baseline+same CE attribution summary`: `run_logs/bridge_append_ce_control_20260407.summary.md`
- `Selector@10 expand probe summary`: `run_logs/selector_top10_expand_probe_20260406.summary.md`
- `MuSiQue top-7 width-matched control`: `run_logs/width_matched_control_musique_top7_20260407.summary.md`
- `Unified top-5 width-matched control`: `run_logs/width_matched_control_top5_20260407.summary.md`
- `Coverage/control status memo`: `research_memory/emnlp_expand_then_compose/10_coverage_probe_and_control_status_20260409.md`
- `NQ/PopQA bridge-threshold diagnosis memo`: `research_memory/emnlp_expand_then_compose/11_nq_popqa_bridge_threshold_diagnosis_20260410.md`
- `Actionized interface repair status memo`: `research_memory/emnlp_expand_then_compose/12_actionized_interface_repair_status_20260414.md`
- `Real full-scale width-matched control queue`: `run_logs/fullscale_width_matched_control_20260409.sh`
- `MuSiQue top-5 remaining optimized queue`: `run_logs/fullscale_width_matched_control_musique_top5_remaining_20260409.sh`
- `Actionized NoisyOR smoke summary`: `run_logs/action_swap_noisyor_smoke_20260414.summary.md`
- `MuSiQue actionized NoisyOR audit`: `run_logs/musique_action_swap_noisyor_audit_20260414.md`
- `2Wiki actionized NoisyOR audit`: `run_logs/2wiki_action_swap_noisyor_audit_20260414.md`

## Current Run Notes

- Do not use `run_logs/fullscale_width_matched_control_20260409.summary.md` for paper claims. That run was accidentally limited to 20 queries.
- Use the `20260409fullfix` tag for the real full-scale control family.
- When rerunning width-matched controls on the same dataset and `qa_top_k`, reuse:
  - `--baseline_report_json` to skip the baseline reader pass
  - `--retrieval_cache_json` to skip duplicated retrieval
  - `--save_retrieval_cache_json` when creating a fresh baseline

## Width-Matched Control Definitions

- `baseline_top5_plus_ce`:
  - baseline-only CE rerank
  - uses `expand_base_k = qa_top_k`
  - uses `append_max_docs = 0`
  - this is not the same-pool append control
- `baseline_top10_plus_ce`:
  - width-matched CE rerank over the top-10 baseline prefix
  - uses `expand_base_k = 10`
  - uses `append_max_docs = 0`
- `bridge_append_plus_ce`:
  - same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`
  - uses `setwise_selector = bridge_append`
  - uses `expand_base_k = 10`
  - uses `append_max_docs = 3`
  - uses `append_policy = bridge`
  - uses `assemble_mode = cross_encoder`
- `random3_deep_plus_ce`:
  - matched random-append control with the same CE rerank stage
  - uses `append_policy = random_deep`
  - uses `append_max_docs = 3`

The `limit=100` width-matched controls and the valid `20260409fullfix` full-scale controls are method-matched.
- They use the same selector, append width, and CE assemble definition.
- The intended difference is only evaluation size and cache reuse.

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
- Ran `baseline qa_top_k` sweep on `MuSiQue / HotpotQA / 2Wiki` over `K in {3,5,7,10}`
- Implemented and locked the `bridge_append` expand path with `assemble_mode in {none, base_score, embedding_similarity, cross_encoder}`
- Ran the `bridge_append` matrix on `MuSiQue / HotpotQA / 2Wiki` over `qa_top_k in {5,7,10}`
- Ran `baseline + same CE` attribution controls for `MuSiQue / HotpotQA / 2Wiki` over `qa_top_k in {5,7,10}`
- Ran the ungated `selector@10` expand probe and confirmed the beam-fill failure mode
- Ran the width-matched attribution control on `MuSiQue top-7`
- Ran the unified width-matched attribution control on `top-5` for `MuSiQue / HotpotQA / 2Wiki`
- Ran the focused `nq/popqa` bridge failure diagnosis at `limit=100`:
  - varied `setwise_query_entity_source in {seed, question}`
  - varied `expand_min_structure_score in {0.35, 0.0}`
  - reused baseline QA and retrieval cache to isolate bridge proposal effects
- Implemented `action_swap_v0_dryrun` and `action_swap_v0_judge` inside the existing `bridge_append` assemble path
- Ran current and relaxed one-swap oracle ceilings on `MuSiQue` and `2Wiki`
- Added the zero-shot controller family:
  - `action_swap_noisyor_flat`
  - `action_swap_noisyor_dep`
- Ran the `limit=100` `action_swap_noisyor` smoke on `MuSiQue` and `2Wiki`
- Added `scripts/audit_action_swap_noisyor.py` for offline action-level postmortem
- Ran action-level NoisyOR audits on `MuSiQue` and `2Wiki`
- Froze the conclusion that the current `noisyor` line is a failed zero-shot controller instance, not the next controller to iterate

## Running

- None

## Next Wave

1. Keep `bridge-aware Expand + CE-centered Assemble` as the frozen paper-facing mainline
2. Use `action_swap_v0_*`, current/relaxed oracle decomposition, and the `noisyor` audit as analysis evidence for interface mismatch
3. Do not expand the `noisyor` controller family with more thresholds, more gates, or more zero-shot variants
4. Add one compact paper table for the actionized line:
  - `bridge_append_plus_ce`
  - `action_swap_v0_dryrun`
  - `action_swap_v0_judge`
  - oracle current
  - oracle relaxed
5. If controller work is revisited later, treat it as a future scaffold-conditional utility project rather than a continuation of the current `noisyor` branch
6. Use the locked `bridge_append` line as the frozen expand baseline for new expand exploration
7. If expand exploration continues, compare against the locked width-matched controls rather than rerunning ad hoc probes
8. For `nq/popqa` specifically, remember the new diagnosis:
  - default bridge failure is threshold-gated
  - `query_entity_source` is not the primary fix lever
  - threshold-unlocked bridge has no QA gain on `nq` and only a small gain on `popqa`

## Method Priority

Method implementation order should be:

1. Paper mainline:
   - keep `bridge_beam + set_closure + pathcore_guard + reserve3 + dedup` frozen as the current simple story
2. Parallel branch:
   - treat `feature/pcrs-rag-v1` as the default development branch
   - fix upstream requirement and counterfactual construction first
   - only revisit matcher training after the oracle signal stops collapsing
3. Only after those gates clear should the new branch be considered for promotion
4. Do not jump to more complex repair loops, online judges, or planner-style control before the branch clears the oracle gate

## Blockers

- The current non-oracle heuristic is not robust on the canonical backbone
- It helps some easy `2-doc` cases but does not yet improve hard `4-doc` composition
- The new `PCRS-RAG V1` branch now has implementation and initial smoke evidence, but not promotion-quality cross-dataset wins
- The biggest technical risk on the new branch is upstream signal quality, not beam search mechanics
- Current `MuSiQue` failure appears to come from requirement and counterfactual construction rather than reserve tuning
- On the `action_swap` line, the current zero-shot support-to-utility surrogate fails to assign positive value to most oracle-positive swaps
- The current dependency-aware NoisyOR controller should be treated as a stopped branch, not an active optimization target

## Do Not Drift

- Do not re-open candidate injection
- Do not re-open reader-side graph context
- Do not re-open causal-only framing
- Do not pitch this as a generic reranking paper
- Do not let the unvalidated `PCRS-RAG V1` branch overwrite the current simple paper line by default
