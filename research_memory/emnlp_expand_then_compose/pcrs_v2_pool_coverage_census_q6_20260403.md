# PCRS-RAG V2 Pool Coverage Census and Q6 Selector Probe

Date: 2026-04-03

## Scope

This note follows the selector-ceiling probe.

Goals:

1. build a small `pool coverage census` over the current smoke10 selector setting
2. stop using q7 as the main selector mechanism case
3. check whether q6 is mainly blocked at `pool -> source` or at `source -> shortlist`

All analysis below uses the same V2 substrate:

- parser / compiler unchanged
- conditional third-hop unchanged
- widened live frontier substrate unchanged
- no retrieval redesign
- no final-stage objective change

## Inputs

Reports used:

- widened frontier baseline
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_frontier_widened_20260403.json`
- pool-gold final ceiling
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_force_pool_gold_final_20260403.json`
- q6 forced-source
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_q6_forced_source_20260403.json`
- positive-only shortlist
  - `outputs_step0_general_musique/eval_reports/requirement_beam_needunit_oracle_smoke10_conditional_bridge_seed_hybrid_probe_positive_only_shortlist_20260403.json`

## Smoke10 Census

Heuristic reading rule:

- `pool_coverage_limited`
  - at least one critical gold title is missing from pool and pool-gold forced-final still does not recover the answer
- `pool_to_source_failed`
  - key in-pool gold stays `pool_only`
- `source_to_shortlist_failed`
  - key in-pool gold reaches `source` but not `shortlist`
- `reader_or_reasoning_limited`
  - pool-gold forced-final materially increases gold evidence in final, but answer still does not improve

This is a rough diagnosis, not a formal label.

| Query | Gold in pool | Widened best gold stages | Ceiling delta | Heuristic bottleneck |
| --- | ---: | --- | --- | --- |
| Messi / Barcelona | 2/2 | selected, selected | no need | already healthy |
| Tripartite / Warsaw Pact | 3/3 | Warsaw Pact selected, Molotov selected, Szlachta source | no gain | source-to-shortlist or reader/reasoning limited |
| Erik Hort | 1/2 | Erik Hort selected, Montebello missing | no gain | pool coverage limited |
| Labyrinth publisher | 2/2 | Labyrinth selected, Acornsoft pool_only | `F1 0 -> 1` | pool-to-source failed |
| Lady Godiva | 2/2 | Spalding Priory selected, Mercia source | no gain | source-to-shortlist or reader/reasoning limited |
| Southeast Library | 4/4 | Southeast Library selected, other 3 all pool_only | `F1 0.4 -> 0.8` | pool-to-source failed |
| Vilaiyaadu Mankatha chain | 3/4 | Vilaiyaadu selected, Right Stuff source, Sony pool_only, Santa Monica missing | no gain | mixed: pool coverage limited + source-to-shortlist |
| III / birthplace battle | 2/3 | III shortlist, Battle of New Orleans pool_only, Flyin' the Koop missing | no gain | mixed: pool coverage limited + downstream use limited |
| Till dom ensamma | 2/2 | selected, selected | no need | already healthy |

## Aggregate Read

The pool-gold final ceiling is real:

- widened baseline:
  - EM `0.2`
  - F1 `0.2733`
  - Recall@5 `0.4083`
- pool-gold final ceiling:
  - EM `0.2`
  - F1 `0.38`
  - Recall@5 `0.6667`

Interpretation:

- selector still has meaningful headroom inside the fixed pool
- but that headroom is mostly showing up as evidence quality / partial answer improvement, not EM recovery
- this is exactly why q7 should not keep driving selector design

## Why Q7 Should Be Downgraded

q7 remains:

- in-pool gold:
  - `Vilaiyaadu Mankatha`
  - `The Right Stuff Records`
  - `Sony Music`
- missing from pool:
  - `Santa Monica, California`

Even after forcing all pool-in gold titles into final:

- final top titles become:
  - `Vilaiyaadu Mankatha`
  - `The Right Stuff Records`
  - `Sony Music`
  - `Look What I Almost Stepped In...`
  - `News World India`
- answer still stays wrong

Conclusion:

> q7 is now best treated as an extreme long-chain, pool-coverage-limited case, not as the main selector tuning target.

## Q6 Stage Probe

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

### Baseline widened trace

Gold titles:

- `Southeast Library`
- `Riverside Plaza`
- `Minneapolis`
- `Mississippi River`

Widened stages:

- `Southeast Library`: `selected`
- `Riverside Plaza`: `pool_only`
- `Minneapolis`: `pool_only`
- `Mississippi River`: `pool_only`

So before any probe, q6 is clearly blocked before source admission.

### Positive-only shortlist

Positive-only does not help q6 at all:

- all three bridge gold docs still stay `pool_only`
- answer degrades to `Gulf of Mexico.`

This means:

> q6 is not mainly blocked by margin-first shortlist sorting, because the key bridge docs never even reach source.

### Forced-source on q6

Probe titles:

- `Riverside Plaza`
- `Minneapolis`
- `Mississippi River`

Overall report:

- EM `0.2`
- F1 `0.2333`

q6-specific trace:

- `Riverside Plaza`
  - forced into source
  - stage becomes `source`
  - still never reaches shortlist
- `Minneapolis`
  - forced into source
  - stage becomes `source`
  - still never reaches shortlist
- `Mississippi River`
  - forced into source
  - then reaches shortlist
  - then reaches selected/final
- final top titles become:
  - `Southeast Library`
  - `Colorado River (Texas)`
  - `Gulf of Mexico`
  - `Mexico City`
  - `Mississippi River`
- answer becomes:
  - `Gulf of Mexico.`

Interpretation:

> q6 now looks like a serial bottleneck, not a single bottleneck.

More specifically:

- primary failure:
  - `pool -> source`
  - because without forcing, all three bridge gold docs stay `pool_only`
- secondary failure:
  - `source -> shortlist`
  - because even after forced-source, `Riverside Plaza` and `Minneapolis` still do not get promoted

This is the cleanest selector mechanism case in the current smoke10 slice.

## Bottom Line

The evidence now supports the following working split:

- q7:
  - downgrade to retrieval / pool-coverage-limited hard case
  - do not let it dominate selector design
- q6:
  - keep as the main selector mechanism case
  - it cleanly exposes early-stage selector bottlenecks with all gold docs already in pool

So the next sensible selector work should be:

1. use a slightly larger `pool coverage census` to measure how common q6-like vs q7-like failures are
2. keep stage probes focused on q6-like queries
3. prioritize `pool -> source` and then `source -> shortlist`, not final-stage objective tuning
