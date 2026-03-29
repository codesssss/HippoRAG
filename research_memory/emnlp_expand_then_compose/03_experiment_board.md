# Experiment Board

Last updated: 2026-03-29

## Canonical Outputs

- `2Wiki-1000 oracle sweep + reorder control`: `outputs_step0_general_2wikimultihopqa/eval_reports/oracle_select_sweep_1000_reorder20.json`
- `HotpotQA-1000 oracle sweep + reorder control`: `outputs_step0_general_hotpotqa/eval_reports/oracle_select_sweep_1000_reorder20.json`
- `MuSiQue-1000 oracle sweep + reorder control`: `outputs_step0_general_musique/eval_reports/oracle_select_sweep_1000.json`
- `2Wiki bridge analysis script`: `scripts/analyze_2wiki_oracle_ceiling.py`
- `Cross-dataset summary script`: `scripts/compile_oracle_select_summary.py`
- `Cross-dataset summary output`: `research_memory/emnlp_expand_then_compose/oracle_select_cross_dataset_summary_20260329_resume.md`

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

## Running

- None

## Next Wave

1. Build one simple non-oracle heuristic baseline for `Expand-then-Compose`
2. Add one compact paper table: baseline vs CE rerank vs oracle reorder vs oracle select
3. Turn support-depth and ceiling curves into paper figures
4. Optional: run a lighter bridge-style analysis on `MuSiQue` if reviewer risk feels high

## Method Priority

Method implementation order should be:

1. Cheapest heuristic first:
   - widen candidate pool
   - setwise select top-5
   - no reader prompt changes
2. Only then consider learned setwise scoring
3. Only after that consider more complex planning or slot-filling variants

## Blockers

- The paper story is still ceiling-heavy until a non-oracle baseline lands

## Do Not Drift

- Do not re-open candidate injection
- Do not re-open reader-side graph context
- Do not re-open causal-only framing
- Do not pitch this as a generic reranking paper

