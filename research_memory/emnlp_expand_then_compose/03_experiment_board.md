# Experiment Board

Last updated: 2026-04-02

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
- `Requirement cache builder`: `scripts/build_requirement_cache.py`
- `Requirement matcher trainer`: `scripts/train_requirement_setwise.py`
- `Requirement beam diagnostics script`: `scripts/analyze_requirement_beam_report.py`
- `MuSiQue requirement beam diagnostic note`: `research_memory/emnlp_expand_then_compose/musique_requirement_beam_reserve_diagnostics_20260402.md`

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

## Running

- None

## Next Wave

1. Patch positive requirement construction on `feature/pcrs-rag-v1`:
   - stop emitting wh-word anchors
   - keep only entity-like or title-grounded anchors
   - do not force-fill anchor count
2. Patch counterfactual construction on `feature/pcrs-rag-v1`:
   - keep `entity_swap`
   - add minimal `role_swap`
   - add minimal `temporal_shift`
3. Rebuild a small `MuSiQue` requirement cache and rerun offline diagnostics before paying reader cost
4. Only rerun QA if the upstream diagnostics improve materially:
   - suspicious anchor rate drops
   - positive-vs-negative separation rises above random
   - finalist leakage range stops collapsing
5. Keep the current simple `pathcore_guard` line as the paper-facing default unless the new branch clearly wins
6. Add one compact paper table: baseline vs CE rerank vs oracle reorder vs oracle select vs current simple selector

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

## Do Not Drift

- Do not re-open candidate injection
- Do not re-open reader-side graph context
- Do not re-open causal-only framing
- Do not pitch this as a generic reranking paper
- Do not let the unvalidated `PCRS-RAG V1` branch overwrite the current simple paper line by default
