# ETv5 Diagnostic Notes

Date: 2026-05-13

## Scope

This note records the current ETv5 diagnostic state for the
`evidence_transition_graphragv4_fact_witnessed_sto` GraphRAG pipeline.

ETv5 is not a new paper-ready method yet. All results below are diagnostics for
deciding whether a chain-closure-aware readout can safely improve ETv4.

## Mainline Boundary

ETv4 clean Qwen32B remains the current mainline:

```text
/mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/
```

The experimental `chain_closure` code path has been frozen and removed from the
paper-facing runner entry points. It must not be reported as the final method.

## ETv5 Fix Attempts

The first ETv5 chain-closure run was a no-op because the chain-closure guard
effectively treated the full admitted universe as source prior.

After fixing that, several deterministic tail-protection variants were tested
on MuSiQue:

```text
Variant  Scope             Effect
-------  ----------------  ------------------------------------------------
fix1     limit100          0/100 top5 changed, no effect
fix2c    limit100/full1000 5/100 and 35/1000 top5 changed, slightly worse
fix3     limit100          18/100 changed, worse
fix4     limit100          13/100 changed, same bad direction as fix3
fix5     limit100          much worse, R@5 collapsed on the pilot
```

Full1000 fix2c on MuSiQue:

```text
Metric         ETv4 clean  fix2c    Delta
------------   ---------   ------   -------
Exact R@5      0.718333    0.718000 -0.000333
Exact All@5    0.422000    0.421000 -0.001000
Title R@5      0.744250    0.743583 -0.000667
Title All@5    0.472000    0.470000 -0.002000
```

Conclusion: simple chain-support/tail-protection rules are not sufficient.
They sometimes remove graph-recovered gold documents.

## Clean Qwen32B Chain Diagnostic

Diagnostic output:

```text
/mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/reports_chain_diagnostic_gainloss/
```

The clean Qwen32B run reproduces the earlier 3-hop failure mode:

```text
Hop  Bucket  Rows  Inserted  Evi-supported%  Weak-graph%  Tail gold drop%  Mean drop rank
---  ------  ----  --------  --------------  -----------  ---------------  --------------
2    gain    69    74        56.76           43.24        0.0              0.0
2    loss    15    22        36.36           63.64        100.0            4.6
3    gain    35    49        57.14           38.78        0.0              0.0
3    loss    30    43        39.53           60.47        93.55            4.5484
4    gain    32    49        57.14           42.86        0.0              0.0
4    loss    8     10        0.0             90.0         100.0            4.625
```

Interpretation:

- The 3-hop issue is real: losses usually drop dense tail gold at ranks 4/5.
- Weak graph connectivity is enriched in loss buckets, but not exclusive to
  losses.
- Many useful graph-recovered gold documents also look weak under the current
  evidence-class features.

## Counterfactual Separability Check

Script:

```text
/mnt/nvme/code/HippoRAG/run_logs/analyze_chain_signal_separability_20260513.py
```

Scope: gain/loss rows only, using gold only for offline scoring.

Counterfactual rule: replace selected graph inserted docs matching a gold-free
feature predicate with the corresponding removed dense-tail docs.

```text
Rule                                  Changed rows  Hit delta  All-gold delta
------------------------------------  ------------  ---------  --------------
replace weak_graph_or_isolated        99            -22        -13
replace weak_graph_no_new_coverage    71            -9         -11
replace no_evidence_tier              99            -22        -13
replace weak_low_fact_degree          81            -19        -11
replace supported_leaf_no_later       131           -56        -45
```

Conclusion: the obvious weak-edge predicates are not separable enough. They
recover some 3-hop losses but hurt more 2-hop/4-hop gains and graph-recovered
gold cases.

## Inserted-Document Origin Check

Script:

```text
/mnt/nvme/code/HippoRAG/run_logs/analyze_chain_inserted_origin_20260513.py
```

This checks whether harmful insertions are mostly symbolic roots, textual tails,
or graph expansions.

```text
Hop  Bucket  Inserted  SymOnly%  Sym+Text%  TextTail%  Expanded%  MeanCandPos  MeanAdmDist
---  ------  --------  --------  ---------  ---------  ---------  -----------  -----------
2    gain    74        0.00      10.81      37.84      51.35      17.15        0.51
2    loss    22        4.55      50.00      18.18      27.27      26.18        0.27
3    gain    49        4.08      20.41      22.45      53.06      28.61        0.53
3    loss    43        0.00      25.58      18.60      55.81      37.63        0.56
4    gain    49        0.00      18.37      20.41      61.22      34.84        0.61
4    loss    10        0.00      20.00      0.00       80.00      57.80        0.80
```

Conclusion: 3-hop losses are not simply symbolic-root preemption. Their inserted
documents have a similar origin profile to 3-hop gains, with most insertions
coming from graph-expanded documents in both buckets.

## Decision

Do not promote the current ETv5 chain-closure variants to the method.

The paper-safe interpretation is:

```text
ETv4's 3-hop failure mode is weakly-connected topical substitution, but the
current graph features do not cleanly separate harmful weak neighbors from
useful graph-recovered gold documents. A future ETv5 needs a stronger
evidence-chain objective, not a simple dense-tail fallback or weak-edge filter.
```

## HotpotQA Reader Diagnostic

HotpotQA retrieval is already near-saturated, so the remaining gap is likely in
reader extraction, context ordering, or prompt sensitivity rather than R@5.

Oracle reader diagnostics:

```text
/mnt/nvme/code/HippoRAG/run_logs/hotpotqa_reader_diagnostics_20260513/outputs/
```

Policies:

```text
gold_first  oracle gold docs first, then ETv4 top5 fill
gold_only   oracle gold docs only
```

These are diagnostics only and must not be mixed into method comparison tables.

Results:

```text
Context      R@5     All-gold@5  Answer-hit@5  EM     F1
-----------  ------  ----------  ------------  -----  ------
ETv4 top5    0.9505  0.9050      0.9100        0.610  0.7413
gold_first   1.0000  1.0000      0.9660        0.642  0.7781
gold_only    1.0000  1.0000      0.9620        0.665  0.7961
```

Interpretation:

- Perfect HotpotQA gold-doc context improves EM by only +5.5 points over ETv4
  top5.
- Removing distractors improves EM by +2.3 points over gold-first.
- Even with gold-only context, GPT-4o-mini still misses 33.5% exact-match
  answers, so the dominant HotpotQA bottleneck is reader extraction/prompt
  behavior, not graph readout.
- Reader-prompt changes are not a retrieval-method contribution unless rerun
  for all baselines under the same reader protocol.

## Local-Optimum Risk

The current evidence says continuing to tune ETv5 by adding weak-edge,
candidate-position, or dense-tail fallback rules is likely local optimization:

```text
Potential rule family             Status
--------------------------------  ------------------------------------------
Weak-edge filter                  Hurts overall; removes graph-recovered gold
Dense/source tail preservation    Hurts when ETv4 graph tail is actual gold
Candidate-position threshold      Positive on gain/loss rows, but ad hoc and
                                  not yet validated on changed-tie rows
Reader ordering on HotpotQA       Limited upside; oracle context still low EM
```

Recommended stop condition:

```text
Unless a principled graph objective separates harmful topical substitutions
from useful graph-recovered gold without dense-tail fallback, freeze ETv4 as
the paper method and report the 3-hop substitution as an honest limitation.
```
