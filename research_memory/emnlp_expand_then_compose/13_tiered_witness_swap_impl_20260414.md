# Tiered Witness Swap Implementation

Last updated: 2026-04-14

Result note:

- The first offline audit result for this branch is recorded in:
  - `research_memory/emnlp_expand_then_compose/14_tiered_witness_audit_results_20260414.md`

## Scope

This note records the successor controller branch implemented after the `noisyor` line was stopped.

The new branch is:

- `assemble_mode=action_swap_tiered_witness`

Its purpose is narrow:

- keep the validated `keep / swap` execution skeleton
- avoid global zero-shot set utility
- avoid online LLM judge control
- replace the failed NoisyOR controller with a local bottleneck-witness scorer

This note is implementation-facing.
It records the frozen design choices, code entry points, validation status, and the intended next experiment order.

## Design choices that were explicitly fixed

### 1. Tiering

Main path:

- heuristic tiers only
- at most two tiers
- tier-aware when the query looks bridge-like

Fallback:

- forced flat fallback when the query is fragmented, too wide, or weakly structured

This keeps tiering deterministic and removes an online LLM failure mode.

### 2. Witness units

Each document is evaluated through a witness envelope:

- `full_doc`
- `title_plus_1sent`
- `title_plus_2sent`

The scorer takes the best unit per facet instead of averaging across units.

### 3. Dual-channel witness score

For each facet `f` and unit `u`:

- `r_f(u)` is the existing CE relevance score
- `a_f(u)` is an ASRank-like answer-scent score
- witness score is `s_f(u) = min(r_f(u), a_f(u))`

This is the concrete answer to the earlier failure mode where CE-only witness signals were too easy to fool lexically.

### 4. Action rule

The controller remains:

- single-swap
- slot-preserving
- `keep ∪ {d ↔ i}`

Current engineering scope:

- candidates are appended bridge docs
- replacees are drawn from bottom-2 incumbents
- legality is relaxed to dedup-only inside the controller

The scoring logic is tier-aware:

- find the earliest tier with a positive candidate witness gain
- choose the strongest facet gain within that tier
- prefer replacees that minimize loss on earlier tiers and same-tier collateral damage
- only execute when witness gain beats witness loss

This intentionally avoids a global `delta` threshold heuristic.

## Code entry points

### Main implementation

- `scripts/eval_causal_qwen3.py`

New helper surfaces added there:

- `build_query_tiers(...)`
- `build_witness_units(...)`
- `compute_answer_scent_score(...)`
- `compute_doc_facet_witness_matrix(...)`
- `summarize_query_tier_supports(...)`
- `score_action_swap_tiered_witness_jobs(...)`
- `select_action_swap_tiered_witness(...)`

### Patch point

The controller is wired into the existing `bridge_append` branch in:

- `scripts/eval_causal_qwen3.py`

The patch location remains the same actionized interface-repair hook:

- after CE `ranking_rows` and action job construction
- before `materialize_reader_top_positions(...)`

### Offline audit

New local audit entry:

- `scripts/audit_action_swap_tiered_witness.py`

It is meant to answer the same type of question as the earlier NoisyOR audit, but for the tiered witness scorer:

- can it rank oracle-positive swaps above negatives offline?
- does it lift dryrun/judge positive actions?
- does heuristic tiering change ranking relative to flat fallback?

## Operational note

The tiered witness audit is local-only, but importing the main eval module can transitively initialize HTTP clients through optional dependencies.

To avoid workstation proxy settings breaking this offline script, the audit runner now clears:

- `ALL_PROXY`
- `HTTP_PROXY`
- `HTTPS_PROXY`

and their lowercase variants before importing `eval_causal_qwen3`.

This is a runtime isolation fix for the audit tool only.
It does not change the online selector path.

## Validation status

### Passed

Syntax:

- `.venv-hipporag/bin/python -m py_compile scripts/eval_causal_qwen3.py scripts/audit_action_swap_tiered_witness.py tests/test_setwise_selector.py`

Selector tests with clean env:

- `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy .venv-hipporag/bin/pytest tests/test_setwise_selector.py -q`

Observed result:

- `152 passed`

Audit CLI boot:

- `.venv-hipporag/bin/python scripts/audit_action_swap_tiered_witness.py --help`

### Added selector tests

The selector test file now covers:

- witness unit construction includes full-doc fallback
- heuristic two-tier decomposition
- forced flat fallback on fragmented queries
- a tier-aware replacement case that protects earlier-tier support and swaps a weaker suffix doc

## What is not yet established

The following is still unknown:

- whether `action_swap_tiered_witness` has offline action-ranking separation on oracle-positive swaps
- whether it executes meaningful swaps online on MuSiQue / 2Wiki smoke
- whether heuristic tiers actually change action ordering in a useful way

So this branch is currently:

- implemented
- unit-tested
- not yet experimentally accepted

## Recommended experiment order

Do not skip directly to broad reader runs.

### Step 1

Run the offline audit first:

- MuSiQue relaxed oracle report + dryrun/judge reports
- 2Wiki relaxed oracle report + dryrun/judge reports

Primary questions:

- do oracle-positive swaps get higher witness gain than negatives?
- do dryrun/judge positive swaps receive positive gain?
- does heuristic tiering change ranking versus flat fallback?

Reference commands:

- MuSiQue:
  - `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy .venv-hipporag/bin/python scripts/audit_action_swap_tiered_witness.py --dataset musique --oracle_relaxed_report run_logs/musique_action_swap_oracle_relaxed_20260413.json --query_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_noisyor_dep_qatopk5_20260414noisyor.json --dryrun_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json --judge_report outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json --output_json run_logs/musique_action_swap_tiered_witness_audit_20260414.json --output_md run_logs/musique_action_swap_tiered_witness_audit_20260414.md --output_csv run_logs/musique_action_swap_tiered_witness_audit_20260414.csv --ce_device cuda:0 --negative_sample_size 40`
- 2Wiki:
  - `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy .venv-hipporag/bin/python scripts/audit_action_swap_tiered_witness.py --dataset 2wikimultihopqa --oracle_relaxed_report run_logs/2wiki_action_swap_oracle_relaxed_20260413.json --query_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_action_swap_noisyor_dep_qatopk5_20260414noisyor.json --dryrun_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_action_swap_v0_dryrun_qatopk5_20260413impl.json --judge_report outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_action_swap_v0_judge_qatopk5_20260413impl.json --output_json run_logs/2wiki_action_swap_tiered_witness_audit_20260414.json --output_md run_logs/2wiki_action_swap_tiered_witness_audit_20260414.md --output_csv run_logs/2wiki_action_swap_tiered_witness_audit_20260414.csv --ce_device cuda:0 --negative_sample_size 40`

### Step 2

Only if Step 1 shows separation, run the smallest online smoke:

- MuSiQue 100
- 2Wiki 100

Against:

- `bridge_append_plus_ce`
- `action_swap_v0_dryrun`
- `action_swap_v0_judge`
- `action_swap_tiered_witness`

Reference launch shape:

- `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy .venv-hipporag/bin/python scripts/eval_causal_qwen3.py --dataset musique --limit 100 --setwise_selector bridge_append --assemble_mode action_swap_tiered_witness --qa_top_k 5`

### Step 3

If Step 1 fails to separate positives from negatives, stop this controller line early.
Do not add new utility surrogates, new thresholds, or more online experiments before that evidence exists.

## Canonical interpretation rule

This branch should be interpreted as:

- a reviewer-driven successor to the failed NoisyOR controller
- still an analysis / repair branch
- not a replacement for the frozen paper mainline

The paper mainline remains:

- `bridge-aware Expand + CE-centered Assemble`
