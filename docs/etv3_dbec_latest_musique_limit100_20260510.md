# ETv3 + DBEC Latest MuSiQue Limit100

Update: this first run used the full-rebuild DBEC selector
(`daec_noisyor_llm`).  A follow-up stability fix changed the default combination
line to baseline-stable local edit (`daec_noisyor_safe_llm` with strict positive
swap gain).  See `docs/etv3_dbec_2hop_stability_fix_20260510.md`.

## Setup

| Item | Setting |
| --- | --- |
| Base retriever | `evidence_transition_graphragv3_variable_flow` |
| Candidate pool | ETv3 top5 prefix + ETv3 candidate universe, pool@100 |
| Selector | `daec_noisyor_llm` |
| Gate | off |
| Graph prior | off |
| Reader | Qwen3-8B, no-thinking |
| Embedding | NV-Embed-v2 |
| Output | `run_logs/etv3_dbec_latest_limit100_20260510/evals/musique_etv3_pool100_dbec_latest_limit100.json` |

The external-pool baseline reproduced frozen ETv3 reader QA:

| metric | value |
| --- | ---: |
| baseline EM | 0.3600 |
| baseline F1 | 0.4282 |
| baseline R@5 | 0.7092 |

So the pool export and reader alignment are valid.

## Result

| subset | n | ETv3 EM | ETv3 F1 | ETv3+DBEC EM | ETv3+DBEC F1 | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 100 | 0.3600 | 0.4282 | 0.3800 | 0.4382 | +0.0100 |
| 2-doc | 48 | 0.5000 | 0.5837 | 0.4792 | 0.5489 | -0.0347 |
| 3-doc | 30 | 0.3000 | 0.3800 | 0.3667 | 0.4356 | +0.0556 |
| 4-doc | 22 | 0.1364 | 0.1545 | 0.1818 | 0.2000 | +0.0455 |

Selector retrieval metrics moved in the opposite direction:

| metric | ETv3 | ETv3+DBEC |
| --- | ---: | ---: |
| R@5 | 0.7092 | 0.6850 |

This means the small QA gain is not because DBEC simply increases gold-document
recall.  It changes the reader-facing composition: some deeper queries become
more answerable, while some short queries lose a useful top5 arrangement.

## Flip Audit

| subset | F1 wins | F1 losses | F1 ties | changed top5 |
| --- | ---: | ---: | ---: | ---: |
| 2-doc | 3 | 4 | 41 | 40/48 |
| 3-doc | 6 | 3 | 21 | 30/30 |
| 4-doc | 2 | 0 | 20 | 19/22 |

The 4-doc gain is real but sparse: only two 4-doc queries improve.  Most 4-hop
queries remain unchanged because the selector does not get enough reliable
binding signal from the pool/query decomposition to build the whole chain.

## Cost / Binding Signal

| binding stat | value |
| --- | ---: |
| LLM binding attempts | 765 |
| LLM binding calls | 731 |
| cache hits | 34 |
| empty entity responses | 386 |
| prompt tokens | 138,930 |
| completion tokens | 5,286 |
| total tokens | 144,216 |
| total binding latency | 57.24 s |
| avg calls/query | 7.31 |
| median calls/query | 5 |

This is the main cost issue.  The selector spends many small LLM calls on
binding extraction, but about half of the binding attempts return no entity.
That makes ETv3+DBEC expensive for the amount of 4-hop gain it gets here.

## Interpretation

1. ETv3's 4-hop problem is not just reader budget.  ETv4 reader-context@10
   raises 4-doc F1 from 0.1545 to 0.2273, but hurts shorter queries and does
   not fix the global method.
2. ETv3+DBEC shows the complementary direction: fixed top5 budget, better
   recomposition.  It improves 3-doc and 4-doc subsets but hurts 2-doc.
3. The 4-hop bottleneck remains because the current DBEC binding layer is too
   weak/empty on MuSiQue.  It often cannot turn the question into a reliable
   chain of intermediate entities, so the selector behaves like noisy demand
   coverage rather than true branch-bound composition.
4. The next clean research move is not another gate or graph-score patch.  It
   is to verify and improve the binding substrate itself: whether each
   decomposed demand has an identifiable upstream entity and whether candidate
   documents expose that entity in a usable form.

## Status

`evidence_transition_graphragv4_reader_context` remains the reader/budget
diagnostic line.  `evidence_transition_graphragv3_dbec_latest` is now the clean
dependency-recomposition diagnostic line over frozen ETv3 retrieval.
