# ETv4 STO Readout Claim and Chain Diagnostic

Date: 2026-05-12

## Scope

This note consolidates the current paper-facing interpretation of
`evidence_transition_graphragv4_fact_witnessed_sto` after the full1000
same-entry dense ablation, MuSiQue per-hop analysis, and chain-consistency
diagnostic.

It is a documentation artifact. It does not define a new method and does not
modify the ETv4 retrieval pipeline.

## Current Method Boundary

```text
Item                         Status
---------------------------- -------------------------------------------------------------
Method folder                evidence_transition_graphragv4_fact_witnessed_sto
Default runner               query_grounded_sto_clean_mainline_v4
Paper-facing behavior        dense/textual entry -> query-local STO graph -> graph readout
Retrieval object             document top-5
Dense role                   graph entry / candidate universe, not final answer default
Graph role                   document-document transition substrate
OpenIE role                  evidence units for STO edge construction / witnessing
Source-prior guard           disabled in paper-facing mainline
Source-text closure          diagnostic only, disabled for selection
Weighted score fusion        not used
LLM query schema             not used
```

ETv4 should be described as a baseline-aligned GraphRAG method, not as a pure
graph-from-scratch retriever. This is consistent with PropRAG/HippoRAGv2-style
systems that use non-graph entry signals, while ETv4's final readout is a
query-local graph readout over document transitions.

## Main Claim

The main claim should be narrow:

```text
Query-local STO document graph readout improves multi-hop document selection
over the same dense entry.
```

Do not overclaim that fact-witness filtering alone is the core contribution.
The raw STO adjacency ablation did not show a retrieval-side drop, so the
supported claim is graph readout over a query-local STO document graph, with
evidence-confirmed promotion as an auxiliary stability constraint.

## Full1000 Same-Entry Dense-Only Ablation

This is the strongest evidence that ETv4 is not just dense retrieval renamed.
Dense top5 is derived from the exact same ETv4 run using saved
`source_prior_prefix_doc_indices`.

```text
Dataset          Rows  Dense R@5  ETv4 R@5  Delta R@5  Dense all-gold@5  ETv4 all-gold@5  Delta all-gold@5  Top5 changed  Gold gains  Gold losses
---------------  ----  ---------  --------  ---------  ----------------  ---------------  ----------------  ------------  ----------  -----------
2wikimultihopqa  1000  0.758500   0.920000  +0.161500  0.493000          0.797000         +0.304000         811           375         5
musique          1000  0.682500   0.724917  +0.042417  0.372000          0.432000         +0.060000         793           149         49
hotpotqa         1000  0.933500   0.950500  +0.017000  0.871000          0.906000         +0.035000         693           52          17
```

Interpretation:

```text
Finding                         Meaning
------------------------------- ---------------------------------------------------------------
2Wiki gain is large              Graph readout strongly improves structured multi-hop selection.
MuSiQue gain is positive/noisy   Graph readout helps, but also introduces substitution losses.
HotpotQA gain is small/positive  Baseline is already high; remaining headroom is limited.
Same-entry protocol              The gain is not from a stronger dense entry or different pool.
```

## MuSiQue Per-Hop Breakdown

Hop is defined as `len(gold_doc_indices)`.

```text
Hop  Rows  Dense R@5  ETv4 R@5  Delta R@5  Dense all-gold@5  ETv4 all-gold@5  Delta all-gold@5  Top5 changed  Mean overlap  Gold gains  Gold losses  Gain/loss
---  ----  ---------  --------  ---------  ----------------  ---------------  ----------------  ------------  ------------  ----------  -----------  ---------
2    518   0.768340   0.827220  +0.058880  0.555985          0.673745         +0.117761         405           0.791598      70          11           6.36
3    316   0.683544   0.688819  +0.005274  0.265823          0.243671         -0.022152         248           0.751733      33          29           1.14
4    166   0.412651   0.474398  +0.061747  0.000000          0.036145         +0.036145         140           0.672476      46          9            5.11
all  1000  0.682500   0.724917  +0.042417  0.372000          0.432000         +0.060000         793           0.759226      149         49           3.04
```

Key correction:

```text
Old intuition                         Corrected interpretation
------------------------------------- ------------------------------------------------------------
4-hop is the main bottleneck           Not supported by full1000 retrieval.
Performance degrades monotonically     Not supported; MuSiQue shows a non-monotonic pattern.
3-hop is just mildly harder            3-hop is the actual fault line for all-gold stability.
```

3-hop is the only MuSiQue bucket where all-gold@5 decreases against the same
dense entry. This is the important limitation to explain.

## Chain-Consistency Diagnostic

Diagnostic script:

```text
evidence_transition_graphragv4_fact_witnessed_sto/diagnose_chain_consistency.py
```

Output artifacts:

```text
run_logs/etv4_full1000_optimized_equivalence_20260511/reports/etv4_musique_chain_consistency_diagnostic.md
run_logs/etv4_full1000_optimized_equivalence_20260511/reports/etv4_musique_chain_consistency_diagnostic.json
```

The diagnostic reconstructs each query-local STO graph from the saved ETv4
retrieval report and compares ETv4 top5 against the same-entry dense prefix.
Gold labels are used only to assign gain/loss buckets. The diagnostic is not
used by the retriever.

Definitions:

```text
Class                 Meaning
--------------------- --------------------------------------------------------------------------
evidence_consumed     Inserted doc has later evidence-transition neighbor or >=2 evidence-transition neighbors.
evidence_leaf         Inserted doc has one evidence-transition neighbor but is not consumed later.
weak_graph_connected  Inserted doc is graph-connected / fact-witnessed but lacks evidence-transition support.
isolated_fill         Inserted doc has no internal selected-top5 edge in the reconstructed local graph.
```

Full diagnostic summary:

```text
Hop  Bucket  Rows  Inserted  Evi-consumed%  Evi-supported%  Weak-graph%  Tail gold drop%  Mean drop rank
---  ------  ----  --------  -------------  --------------  -----------  ---------------  --------------
2    gain    70    74        16.22          59.46           40.54        0.0              0.0
2    loss    11    15        33.33          33.33           66.67        100.0            4.7273
3    gain    33    42        23.81          64.29           30.95        0.0              0.0
3    loss    29    39        23.08          38.46           61.54        96.55            4.5517
4    gain    46    68        14.71          38.24           60.29        0.0              0.0
4    loss    9     13        15.38          23.08           76.92        88.89            4.4444
all  gain    149   184       17.39          52.72           45.65        0.0              0.0
all  loss    49    67        23.88          34.33           65.67        95.92            4.5714
```

Most important comparison:

```text
Bucket      Rows  Inserted  Evi-consumed  Evi-leaf  Weak-graph  Tail gold drop%
----------  ----  --------  ------------  --------  ----------  ---------------
3-hop gain  33    42        10            17        13          0.0
3-hop loss  29    39        9             6         24          96.55
```

## Corrected Failure Mechanism

The earlier phrase "dangling topical branch" is too broad. The diagnostic shows
that most harmful insertions are not isolated. They are graph-connected, often
fact-witnessed, but the connection is frequently weak STO connectivity rather
than evidence-transition support.

Use this wording:

```text
ETv4's 3-hop failure mode is weakly-connected topical substitution:
the graph readout promotes documents that are connected in the STO graph but
not strongly evidence-transition-supported, and these documents often displace
answer-tail gold documents already present at dense ranks 4 or 5.
```

Avoid this wording:

```text
Bad wording                         Reason
----------------------------------- -------------------------------------------------------------
3-hop fails because branches dangle  Many inserted docs are connected; the issue is weak evidence strength.
MuSiQue 3-hop is annotation noise    Case study and diagnostic do not support metric artifact as the main cause.
4-hop is the bottleneck              Full1000 per-hop retrieval shows 4-hop gains are positive and large.
Dense fallback fixes the issue       This would reintroduce source-prior guard style engineering.
```

## Paper-Facing Claim Ledger

```text
Claim type       Recommended statement
---------------- ------------------------------------------------------------------------------------------------
Main claim       Query-local STO document graph readout improves multi-hop document selection over the same dense entry.
Evidence claim   Same-entry dense-only ablation shows gains are from graph readout, not a stronger dense entry.
Auxiliary claim  Evidence-confirmed promotion stabilizes symbolic frontier promotion, but is not the main contributor.
Limitation       MuSiQue 3-hop exposes weakly-connected topical substitution and answer-tail gold displacement.
Future work      Evidence-strength-aware chain readout should distinguish strong evidence transitions from weak STO connectivity.
```

## What Not To Do Next

```text
Temptation                         Decision
---------------------------------- ---------------------------------------------------------------
Add source-prior fallback           Do not. It blurs the clean GraphRAG claim.
Add dense-top5 preservation rule     Do not make it the main method; it looks like a guard.
Add hop-specific routing             Do not. It would be dataset/task-specific engineering.
Use answer-position/gold signals     Forbidden for method; only allowed in offline diagnostics.
Rename weak graph fixes as v5        Only after a clean objective and full ablations exist.
```

## If We Build V5 Later

V5 should not be "ETv4 plus fallback". A clean V5 hypothesis would be:

```text
Evidence-strength-aware chain readout:
read out top5 from the same query-local STO document graph, but distinguish
evidence-transition support from weak STO connectivity when deciding which
graph insertions can occupy the fixed reader budget.
```

The key is to change the graph readout objective, not to preserve dense top5 by
rule. A future V5 must pass the same standards:

```text
Requirement                         Reason
----------------------------------- -------------------------------------------------------------
Same-entry dense comparison          Proves gains are not from entry changes.
No source-prior guard                Keeps the method claim clean.
No hop-specific routing              Avoids dataset-specific tricks.
No answer/gold access                Preserves retrieval fairness.
Full1000 retrieval before QA          Prevents limit100 overfitting.
Ablation against ETv4                Shows the new objective, not added complexity, is responsible.
```

## Current Decision

Freeze ETv4 as the paper-facing method line. Use the chain-consistency
diagnostic as analysis and limitation evidence, not as a new method. The next
work item should be paper planning / claim writing unless a separate V5 research
cycle is explicitly opened.
