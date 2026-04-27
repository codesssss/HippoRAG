# Full Attempt History Synthesis - 2026-04-27

## Purpose

This note is a neutral research-memory consolidation of the recorded `Expand-then-Compose` history.

It is not a paper-writing plan. It summarizes all recorded routes, including positive results, stopped branches, backup branches, partial diagnostics, and planned-but-not-promoted directions.

Scope:
- records under `research_memory/emnlp_expand_then_compose/`;
- canonical reports referenced by `03_experiment_board.md` and `04_result_registry.md`;
- recent diagnostic reports for BSGS, D-PathRAG/CEE/C-CEE, CAPS, CPAG, and Answer-Contrastive Verifier.

## One-Line Accumulated Diagnosis

The durable positive finding is that multi-hop QA benefits from widened-pool evidence composition, and the strongest validated implementation so far is `DtC/DAEC` with dependency binding. Most failed routes share the same pattern:

```text
local signal exists, but it does not become a safe global top-5 evidence or answer decision.
```

Across routes, the recurring gap is:

```text
support presence != evidence coherence != reader consumability != answer correctness
```

## Timeline of Major Pivots

| Date | Pivot | Reason | Current Status |
|---|---|---|---|
| 2026-03-26 | Freeze retrieval backbone on aligned `legacy_fact_graph` | General-graph and causal-retrieval variants underperformed | Fixed historical baseline decision |
| 2026-03-27 | Stop causal-only framing | Multi-hop QA relations are broader than causal relations | Stopped framing |
| 2026-03-28 | Pivot from pointwise reranking to setwise evidence composition | Oracle select ceiling exceeded simple reorder; bridge docs were deep | Foundational framing retained |
| 2026-04-02 | Open `PCRS-RAG V1` while freezing simple bridge line | Requirement-aware objective might fix MuSiQue objective mismatch | Branch later diagnosed as signal-quality limited |
| 2026-04-17 | Preserve `Requirement-Aware Strongest` | Graph-side backup had a positive 2Wiki smoke but mixed Hotpot behavior | Backup only |
| 2026-04-22 | Promote `DtC-Embed` as active fixed-pool composition line | Full1000 NV non-instruction runs positive on all three datasets | Folded into DAEC positive line |
| 2026-04-23 | Demote hard demand gate | Gates reduced losses but over-abstained and lost wins | Negative ablation |
| 2026-04-24 | Promote retriever-agnostic DAEC evidence | DAEC positive on PropRAG and Dense pools, six F1 gains significant | Strongest positive floor |
| 2026-04-25 | Stop BSGS | Oracle-slot graph belief operator failed at set/path coherence | Negative diagnostic |
| 2026-04-26 | De-risk SetR framing and start D-PathRAG pilot | SetR-windowed made generic set-selection framing unsafe | D-PathRAG later stopped |
| 2026-04-27 | Stop D-PathRAG/CEE/C-CEE, CAPS, CPAG; record Answer-Contrastive partial | Residual routes failed to solve hard-negative/global decision gap | Diagnostic-only |
| 2026-04-27 | Stop DAEC-ALR Step-1 single-edit | Reader consistency signal did not transfer into safe edit admission | Negative diagnostic |
| 2026-04-27 | Audit NREV Day-0 | Null-dominated reader likelihood did not pass; same-title replacement bug fixed but first-30 rerun stayed marginal | Diagnostic-only; no full prototype |

## Attempt Ledger

| Route | Goal | Main Result | Success / Failure Cause | Status |
|---|---|---|---|---|
| Oracle select / support-depth analysis | Prove widened pools contain headroom | Strongly positive across 2Wiki, HotpotQA, MuSiQue | Correctly targeted missing support depth and bridge asymmetry | Foundational positive |
| Causal / retrieval-graph variants | Improve retrieval graph itself | Stopped before mainline | Relation space too broad; causal-only operator mismatched QA | Stopped |
| Bridge-greedy | Simple non-oracle setwise selector | Small 2Wiki positives, MuSiQue destructive | Bridge closure helps some hard cases but over-expands on shallow/noisy cases | Superseded |
| Bridge-beam / set-closure stack | Improve search over bridge evidence | 2Wiki positive, shared config safe but weak across datasets | Beam/search helped; reserve/dedup guarded reader but reduced hard-case lift | Frozen old simple line |
| PCRS-RAG V1 / requirement beam | Move from path closure to requirement support + counterfactual leakage | Implementation landed; signal quality weak | Positive requirements noisy, counterfactual axis sparse, leakage collapsed | Stopped as mainline |
| PCRS V2 need units | Replace lexical requirements with semantic need units | MuSiQue-40 oracle F1 `-0.0349` | Schema changed, but parser/scorer remained lexical; counterfactuals sparse | Stopped / diagnostic |
| MuSiQue coverage + alias audit | Patch predicate/alias coverage gaps | Patch helped only `1/5` top-5 gold cases | Failures split across predicate, alias, reachability, relation composition | No broad patch |
| G1 precision audit | Explain bridge-beam residual precision failures | Local structure scores overlapped positives and negatives | High-structure topical expansion mimics true bridges | Diagnostic |
| Requirement-Aware Strongest | Graph-side backup with requirement alignment | 2Wiki smoke positive; Hotpot still below baseline | Graph-side strengthening needs reader-aware and requirement-aware guarding | Backup only |
| MMR/DPP diversity baselines | Test generic structure-blind diversity | Mostly flat; small MuSiQue help | Diversity alone does not assemble dependency-bound support | Baseline only |
| DtC / DAEC fixed-pool composition | Compose top-5 from widened top-100 using demand coverage and binding | Positive on all datasets and both PropRAG/Dense pools | LLM decomposition + dependency binding targets composition rather than generic diversity | Strongest positive floor |
| Hard-crossing DtC | Prevent soft-gain false positives | Weaker than soft DtC | Hard rule removed wins more than losses | Negative ablation |
| Demand gate | Preserve baseline when demands already covered | Over-abstained on all datasets | Hard compose-or-preserve decision too conservative | Negative ablation |
| `satisfiable_by` prompt sweep | Make repairability taxonomy paper-safe | All schema variants below old prompt on average | Prompt/schema perturbation changed LLM decomposition itself | Default off |
| QBF | Generalize PPR with terminal-schema backward flow | Clear negative; QBF below PPR, random chi competitive | Terminal-schema potentials and graph propagation misaligned with support-complete retrieval | Stopped |
| SetR Path A adapter | Compare DAEC to SetR-style widened-pool selection | Adapter implemented and tested; framing later made unsafe | SetR-windowed strong enough that "SetR cannot handle widened pools" is not safe | Available baseline, not current claim |
| BSGS | Graph-native multi-step belief-state filtering | Oracle-slot MuSiQue full support only `54/1000` | Node marginals lose joint set/path coherence; graph edges are not reasoning transitions | Stopped |
| D-PathRAG | Autoregressive evidence-path selector over PropRAG pool | Support complete up, reader F1 down | Selector imported too many hard negatives; support gain decoupled from reader utility | Stopped |
| CEE-pairwise | Learn conservative edit admission | Oracle Edit@1 large headroom; learned beneficial recall very low | Pairwise preferences did not calibrate absolute edit admission | Stopped |
| C-CEE | Use frozen reader likelihood for edit admission | Semantic AUC high, non-oracle edit AUC near random | Reader likelihood needs reliable answer hypothesis; lexical hard negatives dominate | Stopped |
| CAPS Day 0-2 | Candidate-answer proof search with NLI proof scoring | NLI local AUC positive, candidate/proof ranker failed | Local entailment did not aggregate into answer-level proof ranking | Stopped |
| CPAG / RRF | Training-free LLM extraction + agreement graph operator | Cross-pool AUC positive, support/F1 far below PropRAG | Agreement over-rewarded shared hard negatives and hubs | Stopped |
| Answer-Contrastive Verifier v0 | Learn scalar answer-level verifier from CAPS cache | Top-k placement improved, AUC still weak | Scalar features recover ranking hints but not clean gold-vs-best-wrong separability | Partial diagnostic |
| DAEC-ALR Day-0 consistency probe | Test reader self-consistency under DAEC evidence perturbations | Qwen consistency AUC vs F1>=0.5 = `0.7572` | Reader perturbation stability is a real Qwen-side signal | Signal passed |
| DAEC-ALR Step-1 single-edit | Use consistency plus demand/binding proxy for in-pool repair | Main F1 `0.5881 -> 0.5757`, support_complete `0.8550 -> 0.8150` | Consistency diagnoses unstable contexts but fails as edit-admission utility; hard-negative import remains | Stopped |
| NREV Day-0 | Test null-dominated reader-likelihood falsification over gold vs plausible wrong answer-evidence pairs | Full100: `l_plus` AUC `0.7085`, original REV AUC `0.4266`, full NREV AUC `0.6042`; audit fixed same-title replacement and first-30 full NREV became `0.6644` | Evidence-world likelihood has marginal signal; current null construction is not safe enough, and fixed T-minus is still below keep-alive threshold | No full prototype; at most one bounded T-minus redesign |
| LCPS | Typed claim/program execution idea | Not executed; rejected by review | Assumed brittle LLM typed extraction, query compilation, predicate canonicalization | Do not prioritize without Day-0 executability proof |
| SC-SER | Sufficiency-calibrated selective evidence repair | Planned after demand gate | Intended to replace hard gate with continuous preservation/repair objective | Planned, no recorded full result |

## Foundational Analysis: Oracle Select and Support Depth

### What was tested

The first core question was whether the limitation is pointwise ranking or evidence composition from a widened pool.

Recorded in:
- `00_north_star.md`
- `01_claim_ledger.md`
- `oracle_select_cross_dataset_summary_20260329_resume.md`
- `04_result_registry.md`

### Results

2Wiki:
- baseline EM/F1: `0.4360 / 0.4916`
- oracle@100 EM/F1: `0.5600 / 0.6318`
- F1 delta: `+0.1402`
- 2-doc median support depth: `3`
- 4-doc median support depth: `64`
- anchor Recall@20: `0.9995`
- bridge Recall@20: `0.5997`
- partial top-5 failures bridgeable: `0.7115`

HotpotQA:
- baseline F1: `0.6760`
- oracle@100 F1: `0.7707`
- F1 delta: `+0.0947`
- shallower 2-doc contrast dataset.

MuSiQue:
- baseline F1: `0.3427`
- oracle@100 F1: `0.5401`
- F1 delta: `+0.1974`
- support depth grows with complexity: 2-doc median `5`, 3-doc `18`, 4-doc `55`.

### Why this succeeded

The analysis targeted the correct structural bottleneck:

```text
the needed evidence often exists in the expanded pool, but not in the top-5 reader context.
```

This justified `Expand-then-Compose` and killed the idea that the problem is only better pointwise reranking.

### Core insight

Widening the pool creates real oracle headroom, but the hard part is selecting a reader-usable, coherent top-5 support set.

## Early Non-Oracle Bridge Selectors

### Bridge-greedy

Recorded in:
- `05_non_oracle_bridge_greedy.md`

Early `general_relation_graph` pilot:
- baseline F1: `0.5000`
- selector F1: `0.6375`
- delta F1: `+0.1375`

Canonical `legacy_fact_graph` behavior:
- 2Wiki pilot stayed weakly positive.
- MuSiQue pilot was destructive: F1 `0.3250 -> 0.1559`, delta `-0.1691`.

Failure reason:
- It behaved too much like structure/diversity expansion, not a reliable bridge completer.
- It over-traded easy/shallow reader-compatible contexts for noisy graph-near documents.

### Bridge-beam / set-closure stack

Recorded in:
- `06_bridge_beam_search.md`
- `simple_g1_precision_failure_audit_20260405.md`

2Wiki:
- baseline F1 `0.4332`
- greedy F1 `0.4726`
- beam F1 `0.4924`

MuSiQue aggressive beam:
- baseline F1 `0.3466`
- bridge-beam F1 `0.2498`
- 4-doc bucket still positive while 2-doc/3-doc were harmed.

Reserve3 + dedup:
- MuSiQue EM became slightly positive, F1 only slightly negative.

Shared closure-only config:
- 2Wiki delta F1 `+0.0387`
- Hotpot delta F1 `-0.0050`
- MuSiQue delta F1 `-0.0014`

G1 precision audit:
- positives and negatives overlapped on structure, novelty, closure, support mean, and state score.
- high-structure topical expansion can look like a valid bridge.

Success reason:
- Beam search and reserve/dedup make evidence assembly less brittle than greedy append.

Failure reason:
- Local graph structure cannot reliably distinguish true bridge utility from topical high-structure noise.

Core insight:

```text
search matters, but local bridge scores need incremental chain utility and reader-aware guarding.
```

## PCRS / Requirement-Beam / Need-Unit Route

### PCRS-RAG V1

Recorded in:
- `07_pcrs_rag_v1.md`
- `musique_requirement_beam_reserve_diagnostics_20260402.md`

Goal:
- Move from path-centric bridge closure to:
  - positive requirement support;
  - counterfactual leakage suppression;
  - Pareto-style set selection.

Diagnosis:
- positive requirement construction was noisy;
- counterfactual construction was too weak;
- positive-vs-negative separation was near random;
- leakage axis collapsed within finalists.

### PCRS V2 need units

Recorded in:
- `pcrs_v2_needunit_review_20260403.md`
- `pcrs_v2_needunit_smoke40_analysis_20260403.md`
- `pcrs_v2_shortlist_trace_diagnosis_20260403.md`
- `pcrs_v2_atomic_hybrid_smoke10_20260403.md`
- `pcrs_v2_selector_ceiling_smoke10_20260403.md`
- `pcrs_v2_stage_isolation_probes_smoke10_20260403.md`
- `pcrs_v2_live_frontier_ablation_smoke10_20260403.md`
- `pcrs_v2_bridge_*`
- `pcrs_v2_q6_*`
- `pcrs_v2_varfix_*`
- `pcrs_v2_failure_family_scan_smoke40_20260404.md`
- `pcrs_v2_family_aware_gated_smoke40_20260404.md`

Key MuSiQue-40 need-unit oracle result:
- baseline F1: `0.4016`
- selector F1: `0.3667`
- delta F1: `-0.0349`

Cache/schema diagnostics:
- lexical-looking relation-hop rate: `0.8049`
- counterfactual sets present in only `8/40` queries
- average counterfactual sets per query: `0.375`

Bottom line from the recorded review:

```text
V2 is a structural migration, but not yet a semantic migration.
It is V1-style signals inside a better schema.
```

### PCRS micro-probe failure families

The Q6/Q7 and bridge-seed probes repeatedly isolated the same families:
- pool coverage limits;
- source admission errors;
- variable binding breaks;
- predicate/relation coverage gaps;
- alias and canonicalization failures;
- relation composition not represented by local overlap scores;
- beam state saturation without true evidence utility.

Why the route failed:
- The selector interface was moving in the right conceptual direction, but upstream semantic units were not reliable enough.
- The LLM/parser/scorer stack produced structured-looking artifacts without stable support/contradiction semantics.

Core insight:

```text
Requirement-aware selection only works if requirement units are semantic enough.
A better schema does not fix lexical/noisy signals by itself.
```

## MuSiQue Coverage / Alias Audit

Recorded in:
- `09_musique_coverage_alias_audit_20260406.md`

Gate-boundary split:
- beneficial_withheld: `8`
- harmful_blocked: `11`
- beneficial_allowed: `1`
- harmful_allowed: `0`

Patch audit:
- 5 A-class `structure_score=0` cases had raw triples.
- `general_factual_v2` made structure nonzero in `2/5`.
- gate unblocked `2/5`.
- top-5 gold improved `1/5`.
- go decision: false.

Alias audit:
- Q65: pure title alias closure.
- Q69: alias already present, reachability still missing.
- Q85: descriptor canonicalization plus relation composition.

Why this branch did not continue:
- The residual failures split across different layers.
- A broad predicate/alias patch would be low precision.

Core insight:

```text
Residual MuSiQue errors are not one bug family.
They distribute across predicate coverage, aliasing, reachability, and relation composition.
```

## Requirement-Aware Strongest Backup

Recorded in:
- `10_requirement_aware_strongest_backup.md`

2Wiki strongest smoke:
- control F1: `0.4170`
- strongest F1: `0.4824`
- delta F1: `+0.0654`

HotpotQA strongest/GBC audit:
- baseline F1: `0.6568`
- standard strongest F1: `0.5568`
- clean GBC F1: `0.6068`

Interpretation:
- Graph-side strengthening can help on 2Wiki.
- Raw graph strongest is too aggressive.
- Clean GBC stabilizes but remains below Hotpot baseline.

Status:
- keep as a backup line only.
- do not promote without Hotpot parity, retained 2Wiki gains, and non-destructive MuSiQue trend.

Core insight:

```text
Graph proximity is useful but must be aligned to unmet query requirements and reader compatibility.
```

## Diversity Baselines

Recorded in:
- `11_diversity_selector_baselines_20260421.md`
- `04_result_registry.md`

Structure-blind MMR/DPP @100 -> 5:

| Dataset | Baseline F1 | MMR F1 | DPP F1 | Oracle@100 F1 |
|---|---:|---:|---:|---:|
| 2Wiki | 0.4868 | 0.4805 | 0.4863 | 0.6318 |
| HotpotQA | 0.6839 | 0.6830 | 0.6829 | 0.7707 |
| MuSiQue | 0.3439 | 0.3545 | 0.3569 | 0.5401 |

Why they failed:
- Generic embedding diversity does not encode dependency, bridge, answer-type, or requirement satisfaction.
- It can help deep MuSiQue cases slightly, but closes only a small fraction of the oracle gap.

Core insight:

```text
Assemble is not the same as diversify.
```

## DtC / DAEC Positive Line

Recorded in:
- `12_dtc_supplemental_experiments_20260421.md`
- `13_musique_4doc_regression_analysis_20260421.md`
- `14_dtc_nv_full1000_rank_prior_20260422.md`
- `15_demand_gate_ablation_20260423.md`
- `satisfiable_by_prompt_sweep_20260424.md`
- `16_layer1_retriever_agnostic_composition_20260424.md`
- `17_layer1_followup_taxonomy_significance_20260424.md`
- `04_result_registry.md`

### What succeeded

Aligned NV-Embed full1000 positive:
- 2Wiki delta F1: `+0.0082`
- HotpotQA delta F1: `+0.0175`
- MuSiQue delta F1: `+0.0245`

Layer-1 PropRAG top100 DAEC:
- 2Wiki: `0.6457 -> 0.6810`, delta `+0.0353`
- HotpotQA: `0.7227 -> 0.7348`, delta `+0.0121`
- MuSiQue: `0.4266 -> 0.4404`, delta `+0.0138`

Layer-1 Dense top100 DAEC:
- 2Wiki: `0.4984 -> 0.5628`, delta `+0.0644`
- HotpotQA: `0.7106 -> 0.7326`, delta `+0.0220`
- MuSiQue: `0.3896 -> 0.4138`, delta `+0.0242`

Significance:
- all six F1 gains have bootstrap 95% confidence intervals above zero and paired sign-flip p-values below `0.05`.

Dependency binding ablation:
- all five tested `(dataset, pool)` pairs lose F1 when binding is disabled.
- binding contribution examples:
  - 2Wiki Dense: `+0.0544`
  - MuSiQue Dense: `+0.0245`
  - 2Wiki PropRAG: `+0.0211`

### Why DAEC succeeded

DAEC is the closest match to the validated bottleneck:
- it keeps the widened pool fixed;
- it composes the final reader budget instead of replacing retrieval;
- it uses decomposed demand coverage rather than generic diversity;
- dependency binding preserves multi-hop variable continuity;
- it remains conservative enough to avoid the worst graph/operator drift.

### What did not work inside DAEC

Hard-crossing:
- 2Wiki soft delta `+0.0256`, hard `+0.0157`;
- Hotpot soft `+0.0310`, hard `+0.0010`;
- MuSiQue soft `+0.0497`, hard `+0.0023`.

Demand gate:
- no-gate beats every hard gate variant:
  - 2Wiki no-gate `+0.0456`, best gate `+0.0382`;
  - Hotpot no-gate `+0.0310`, best gate `+0.0050`;
  - MuSiQue no-gate `+0.0397`, best gate `+0.0034`.

`satisfiable_by` prompt:
- old prompt average selector F1: `0.5672`;
- all `satisfiable_by` variants lower;
- `regex_only` also dropped, proving the prompt/schema perturbation itself hurt decomposition.

MuSiQue 4-doc regression:
- pilot100 overall positive `+0.0589`;
- 4-doc bucket negative `-0.0508`;
- losses dominated by false/soft-gain insertions that evicted answer-bearing baseline docs.

Core insight:

```text
Continuous demand-aware composition works better than hard gates.
Dependency binding is load-bearing.
Prompt/schema changes can hurt by perturbing decomposition, even if downstream policy is unchanged.
```

## QBF

Recorded in:
- `15_qbf_pilot_negative_result_20260424.md`

Goal:
- Generalize PPR with query-conditioned terminal-schema backward flow.

Key results:
- 2Wiki PPR SC@5: `0.5600`
- 2Wiki `ppr_qbf_rerank` SC@5: `0.2800`
- 2Wiki oracle-relation QBF SC@5: `0.3900`
- MuSiQue PPR SC@5: `0.3500`
- MuSiQue `ppr_qbf_rerank` SC@5: `0.1300`
- MuSiQue oracle-relation QBF SC@5: `0.2600`

Critical diagnostics:
- terminal-schema `chi` actively harmed ranking;
- raw QBF destroyed top-100 pool recall;
- random `chi` was competitive with schema `chi`;
- oracle relation did not rescue QBF above PPR.

Failure reason:
- This was operator-level mismatch, not just a parser or hyperparameter miss.
- The current graph substrate does not support terminal-flow ranking as a reader-ready evidence objective.

Core insight:

```text
Graph-native directional flow is not automatically a better intervention point than fixed-pool composition.
```

## SetR Path A

Recorded in:
- `20_pathA_setr_widened_pool_plan_20260426.md`
- `21_dpathrag_direction_20260426.md`

What was implemented:
- `scripts/export_setr_inputs.py`
- `scripts/run_setr_style_selector.py`
- `scripts/apply_setr_selection_to_pool.py`
- `scripts/analyze_pool_support_depth.py`
- `tests/test_setr_adapter.py`

Validation:
- SetR adapter tests passed.
- Mock smoke passed.
- External-pool alignment exact in smoke.

Why this did not become the current main framing:
- SetR-windowed was strong enough that "SetR cannot handle widened pools" became unsafe.
- Official SetR code is top-20/eval-incomplete locally, but the conceptual baseline is close enough that generic set-selection novelty is risky.

Status:
- Adapter remains useful as a baseline tool.
- Not a negative method result by itself.

Core insight:

```text
DAEC should be framed around dependency-bound widened-pool composition, not generic set selection.
```

## BSGS

Recorded in:
- `18_bsgs_week0_week1_plan_20260425.md`
- `19_bsgs_negative_diagnostic_20260425.md`

Goal:
- Replace fixed-pool composition with graph-native multi-step belief filtering over proposition nodes.

Week-0 slot diagnostics:
- Qwen slot F1: `0.6552`
- slot count within one: `0.9428`
- dependency presence accuracy: `0.6908`
- position-chain accuracy: `0.5575`

Week-1 oracle-slot MuSiQue-1000:
- supporting paragraph recall: `0.2268`
- full support covered: `54/1000`
- no support covered: `546/1000`

MuSiQue-200 oracle diagnostic:
- base paragraph recall: `0.1608`
- oracle_pool: `0.2113`
- oracle_pool_binding: `0.4758`
- oracle_pool_binding_likelihood: `0.5746`
- full support still only `43/200` under strongest oracle variant.
- duplicate-title rate rose to `0.7170`.

Failure reason:
- Node marginal belief drops the joint set/path information needed for multi-hop QA.
- HippoRAG graph edges are association edges, not reliable reasoning transitions.
- Oracle slots and oracle likelihoods improve local pieces but do not recover coherent evidence sets.

Core insight:

```text
The validated object is the evidence set, not node marginal mass.
Binding and set coherence cannot be recovered by graph diffusion alone.
```

## D-PathRAG / CEE / C-CEE

Recorded in:
- `21_dpathrag_direction_20260426.md`
- `20_dpathrag_cee_ccee_negative_diagnostic_20260427.md`

### D-PathRAG v1

Goal:
- Learn/free-form select an evidence path over PropRAG top candidates.

Result:
- PropRAG support_complete: `0.7720`
- D-PathRAG support_complete: `0.8050`
- PropRAG F1: `0.4824`
- D-PathRAG F1: `0.4634`
- added gold docs: `76`
- added non-gold docs: `1205`
- non-gold/gold import ratio: `15.8553`

Failure reason:
- Mechanism-positive for support, but reader-negative.
- Hard-negative import overwhelmed recovered support.
- Gold support presence did not imply reader consumability.

### CEE-pairwise

Goal:
- Start from PropRAG top-5 and apply at most one conservative edit.

Oracle headroom:
- rank top-5 support_complete: `0.7720`
- oracle Edit@1 top20 support_complete: `0.8950`

Learned admission:
- CEE-pairwise beneficial recall: `0.0221` / `0.0588`
- support_complete stayed around `0.7710`.

Failure reason:
- Pairwise preference did not become calibrated absolute admission.
- Conservative models avoided bad edits but missed good edits.

### C-CEE

Goal:
- Use frozen reader likelihood as test-time edit admission.

Result:
- reader semantic AUC: `0.988332`
- oracle-answer beneficial-vs-lexical-HN AUC: `0.713375`
- non-oracle beneficial-vs-lexical-HN AUC: `0.536875`
- top1 beneficial rate: `0`

Failure reason:
- Reader likelihood has semantic capacity when answer/evidence are oracle-controlled.
- Under non-oracle answer uncertainty, lexical hard negatives dominate.

Core insight:

```text
Evidence selection gains can decouple from answer F1.
Edit admission is a separate hard problem from edit action-space headroom.
```

## CAPS

Recorded in:
- `21_caps_day0_nli_sanity_20260427.md`
- `22_caps_day1_candidate_recall_failure_20260427.md`
- `23_caps_day1_5_candidate_v2_failure_20260427.md`
- `24_caps_day2_proof_separability_20260427.md`

Goal:
- Make answer candidate explicit and score each candidate by proof obligations using NLI.

Day-0:
- NLI verifier AUC: `0.903`
- local obligation-level signal was strong.

Day-1:
- candidate recall@5/10/20: `0.670 / 0.740 / 0.750`
- gold answer string in union top60 docs: `0.900`
- diagnosis: evidence often exists, but candidate extraction/ranking misses it.

Day-1.5:
- LLM-only candidate recall@20: `0.485`
- LLM + v1 + string extraction recall@5/10/20: `0.635 / 0.665 / 0.840`
- diagnosis moved from pure generation to candidate ranking.

Day-2:
- candidate recall@20: `0.840`
- all-query top1/top3: `0.285 / 0.490`
- conditional top1/top3 given gold present: `0.339286 / 0.583333`
- gold-vs-best-wrong AUC: `0.398136`
- mean gold proof score: `0.383771`
- mean best wrong proof score: `0.530366`

Failure reason:
- Local NLI obligation scoring did not aggregate into answer-level proof ranking.
- Plausible wrong candidates satisfy enough local obligations to beat gold under noisy-OR style aggregation.
- Even oracle/template obligations did not rescue the proof ranker.

Core insight:

```text
Obligation-level entailment is not answer-level proof discrimination.
Local verifier strength is not sufficient without a proof object that preserves variable binding and specificity.
```

## CPAG / RRF

Recorded in:
- `25_cpag_day1_agreement_failure_20260427.md`
- `reports/cpag/audit200/cpag_implementation_audit.md`

Goal:
- Use training-free LLM local extraction plus robust query-local agreement graph closure.

Setup:
- 2Wiki dev fold 0, 200 queries.
- PropRAG top20 + Dense top20.
- BM25 unavailable in local cache.
- cached Qwen OpenIE artifacts.

Local positive signal:
- gold-vs-non-gold cross-pool count AUC: `0.772391`.

Selection results:
- PropRAG support_complete: `0.7050`
- RRF support_complete: `0.4900`
- CPAG anchor-first support_complete: `0.4700`
- PropRAG F1: `0.4720`
- RRF F1: `0.4162`
- CPAG anchor-first F1: `0.3897`
- CPAG added `3` gold docs and `87` non-gold docs.

Audit:
- avg PropRAG/Dense overlap@20: `8.84`
- all implementation checks passed.
- PropRAG top1 pushed below RRF rank 3 in `12/200` queries.

Failure reason:
- Cross-pool agreement is locally discriminative but not specific enough.
- Shared distractors get reinforced by both retrievers.
- Agreement closure chases high-degree wrong hubs.

Core insight:

```text
Redundancy is not specificity.
RRF and agreement can over-reward shared hard negatives on a strong substrate.
```

## Answer-Contrastive Verifier v0

Recorded in:
- `26_answer_contrastive_verifier_20260427.md`
- `reports/contrastive_verifier/day1_gate.md`

Goal:
- Learn a lightweight answer-level verifier from existing CAPS Day-2 candidate/proof/source scalar features.

Setup:
- 200 dev queries.
- 4000 candidate rows.
- 168 gold-present rows for conditional ranking.
- no new LLM calls, no DeBERTa/Qwen fine-tuning.

Baseline:
- proof-score gold-vs-best-wrong AUC: `0.398136`
- conditional top1/top3: `0.339286 / 0.583333`

Best scalar model:
- `hist_gradient_boosting`
- gold-vs-best-wrong AUC: `0.570826`
- 95% CI: `[0.514952, 0.624433]`
- conditional top1/top3: `0.595238 / 0.827381`

Interpretation:
- It recovers meaningful top-k placement.
- It still fails clean same-query gold-vs-best-wrong separability.
- It is not enough to reopen CAPS as a mainline in this form.

Core insight:

```text
Supervised scalar features can correct some ranking order, but the current feature space still lacks the semantic discrimination needed for robust answer choice.
```

## DAEC-ALR Day-0 Reader Consistency Probe

Recorded in:
- `29_daec_alr_consistency_probe_20260427.md`
- `reports/daec_alr/consistency_probe.md`

Goal:
- Test the previously untested reader-in-the-loop signal: whether answer stability under evidence perturbation separates correct and wrong DAEC contexts.

Setup:
- 2Wiki first 200 queries from PropRAG top100 DAEC.
- Perturbations: `original`, `swap01`, `reverse`, `rotate_left`, `drop_last`.
- Primary reader: local `qwen3-8b-train` through the HippoRAG one-shot QA prompt.
- 1000 total reader outputs.

Primary result:
- original reconstructed DAEC EM/F1: `0.4900 / 0.5882`
- majority-consistency AUC vs original `F1>=0.5`: `0.7572`, CI `[0.6875, 0.8202]`
- inverse-entropy AUC vs original `F1>=0.5`: `0.7598`, CI `[0.6902, 0.8226]`
- EM-label majority-consistency AUC: `0.7683`

Control:
- Fine-tuned Flan-T5 reader did not show the same separation: majority-consistency AUC vs `F1>=0.5` was `0.5666`.

Interpretation:
- This is a positive reader-side signal, but it is reader-dependent.
- It does not yet prove edit/retrieval improvement.
- It justifies a bounded single-edit-in-pool probe using DAEC binding improvement plus consistency improvement as joint admission.

Core insight:

```text
Qwen answer self-consistency under evidence perturbation is the first new residual signal after DAEC that clears a signal-only probe.
```

## DAEC-ALR Step-1 Single-Edit Gate

Recorded in:
- `30_daec_alr_step1_plan.md`
- `31_daec_alr_step1_single_edit_failure_20260427.md`
- `reports/daec_alr/step1_single_edit_gate.md`

Goal:
- Test whether the Day-0 Qwen consistency signal can safely drive at most one replacement inside the existing PropRAG top100 pool.

Important implementation boundary:
- The exported DAEC report does not include enough embedding state to exactly recompute arbitrary candidate DAEC scores.
- Step-1 therefore used a fixed trace-local demand/binding proxy: protect covered requirement positions, replace a low-utility filler, and rank candidates by requirement anchors, binding-candidate title hits, subquery overlap, answer-type cues, and rank prior.

Validity audit:
- An initial `max_new_tokens=32` run was invalid because it truncated Qwen answers and collapsed the no-edit baseline to F1 `0.2034`.
- The cache key was fixed to include generation limits.
- The valid run used `max_new_tokens=64` and matched Day-0 no-edit: EM/F1 `0.4900 / 0.5881`.

Primary 2Wiki eval200 result:
- no-edit F1/support_complete: `0.5881 / 0.8550`
- main `double_gate_skip` F1/support_complete: `0.5757 / 0.8150`
- F1 delta: `-0.0125`, CI `[-0.0371, 0.0110]`
- accepted edit rate: `0.230`
- edited-subset F1: `0.2910 -> 0.2343`
- flips: wrong-to-correct `3`, correct-to-wrong `5`
- hard-negative import: added gold/non-gold `4/42`, ratio `10.50`
- decision: `STOP_DAEC_ALR_STEP1_SINGLE_EDIT`

Core insight:

```text
Reader self-consistency is a useful context-stability diagnostic, but not a reliable edit-admission utility for local evidence repair.
```

This closes the DAEC-ALR route in its current form. Reopening requires either reconstructing the exact DAEC embedding scorer for arbitrary candidate edits or replacing the admission object materially.

## Reviewed But Not Executed as Experiments

### LCPS

Context:
- Discussed as a training-free LLM + symbolic program route.
- CPAG memo records it as rejected before execution.

Reason for rejection:
- It assumes high-fidelity typed claim extraction.
- It assumes accurate question-to-program compilation.
- It assumes reliable predicate canonicalization.
- These assumptions were contradicted by PCRS/CAPS evidence: Qwen-style extraction and multi-doc candidate generation were not stable enough.

Status:
- Do not prioritize without a strict Day-0 gold-doc executability proof.

### SC-SER

Context:
- Proposed after demand-gate ablation.

Goal:
- Replace hard compose-or-preserve gate with continuous sufficiency-calibrated repair.

Status:
- Planned direction only.
- No recorded full implementation/evaluation in the memory docs.

## What Has Actually Succeeded

1. Oracle select and support-depth analysis succeeded because they measured the correct structural bottleneck: missing support is often deep, especially bridge/final-hop support.

2. Bridge-beam succeeded partially because it introduced setwise search and preserved some baseline context, but it was not robust enough across datasets.

3. DAEC succeeded because it targets the same object as the oracle: a final evidence set under reader budget, using decomposed demand coverage and dependency binding rather than generic diversity.

4. Dependency binding is the most consistently supported DAEC component; all tested no-binding settings lost F1.

5. Local diagnostic signals can be strong:
- NLI obligation AUC in CAPS Day-0: `0.903`.
- reader semantic capacity in C-CEE: `0.988332`.
- cross-pool count AUC in CPAG: `0.772391`.
- oracle edit headroom in CEE: support_complete `0.772 -> 0.895`.
- Qwen reader perturbation consistency in DAEC-ALR Day-0: AUC vs `F1>=0.5` = `0.7572`.
- DAEC-ALR Step-1 consistency-gated edits: accepted edit rate `0.230`, but F1/support both dropped.

But these local successes usually failed at global decision time.

## Why Most Routes Failed

### 1. Local-to-global mismatch

Repeated pattern:

```text
local score separates something useful
global top-5 or top-1 decision still fails
```

Examples:
- CAPS: NLI AUC `0.903`, proof ranker AUC `0.398`.
- CPAG: cross-pool count AUC `0.772`, CPAG F1 `-8.23pp`.
- C-CEE: reader semantic AUC `0.988`, non-oracle edit AUC `0.537`.

### 2. Hard-negative import

D-PathRAG added real gold supports, but also imported many more non-gold supports:

```text
76 added gold vs 1205 added non-gold
```

CPAG and RRF repeated the same failure in agreement space.

### 3. Reader sensitivity

The reader does not simply reward support completeness. It is sensitive to distractors and ordering.

Key evidence:
- D-PathRAG support_complete improved but F1 dropped.
- gold_support_only F1 exceeded gold_plus_selector_distractors by `11.74pp`.
- HotpotQA DAEC losses are mostly reader interference.

### 4. Binding is hard and load-bearing

BSGS oracle diagnostics show the largest jump from oracle binding, but even oracle binding + oracle likelihood does not solve full support.

DAEC no-binding ablations show binding contributes across pools/datasets.

### 5. Hard gates over-abstain

Hard-crossing and demand gate both removed too many wins.

Result:

```text
hard filters reduce some losses, but they usually shrink useful intervention capacity more.
```

### 6. LLM use is only safe when the task is native and the operator is robust

LLM decomposition/proposition extraction can be useful, but:
- adding `satisfiable_by` perturbed decomposition;
- LLM-only CAPS candidate generation was worse than reader/string union;
- LCPS-style typed program compilation is too brittle without a hard executability gate.

### 7. Graph operators need matching substrates

QBF and BSGS both failed because the graph edges encode association, containment, alias, and noisy OpenIE relations, not clean reasoning transitions.

Core condition:

```text
operator semantics must match substrate semantics.
```

## Dataset-Specific Lessons

### 2Wiki

Most useful for:
- bridge-depth and anchor/bridge asymmetry;
- dependency-binding validation;
- evidence composition gains;
- hard-negative diagnostics in PropRAG residuals.

Failure mode:
- bridge entities and lexical neighbors are close enough that local selection easily imports hard negatives.

### HotpotQA

Most useful as:
- shallow contrast dataset;
- reader-interference stress test.

Failure mode:
- baseline support is often already strong, so edits have limited headroom and can hurt reader context.

### MuSiQue

Most useful as:
- deep multi-hop generalization dataset;
- residual failure stress test.

Failure modes:
- larger oracle headroom but weaker recovery;
- relation composition, alias/canonicalization, binding, and reader-noise failures overlap;
- 3-doc/4-doc cases expose objective mismatch and false-positive coverage.

## Current Status Map

### Keep as positive floor

`DtC/DAEC`:
- retriever-agnostic fixed-pool evidence composition;
- positive on PropRAG and Dense top100 pools;
- dependency binding retained as load-bearing.

### Keep as backup only

`Requirement-Aware Strongest`:
- runnable reserve option;
- not promoted without Hotpot parity and MuSiQue non-destructive evidence.

### Keep as baselines / controls

- baseline top-5 rank;
- oracle select@K;
- MMR/DPP;
- SetR-style adapter;
- RRF, now also a cautionary multi-pool fusion control.

### Treat as stopped diagnostics

- causal-only framing;
- bridge-greedy as mainline;
- PCRS V1/V2 in current form;
- QBF;
- BSGS;
- D-PathRAG;
- CEE-pairwise;
- C-CEE;
- CAPS current proof object;
- CPAG current agreement object;
- Answer-Contrastive scalar verifier as mainline.

### Do not continue DAEC-ALR Step-1 as currently formulated

`DAEC-ALR Step-1`:
- Day-0 context-stability signal passed;
- single-edit admission failed on F1, support_complete, edited-subset utility, flip balance, and hard-negative import;
- do not tune thresholds or start active retrieval on top of this admission rule;
- reopen only if the exact DAEC scorer is reconstructed for candidate edits or the admission object changes materially.

### Only reopen if the object changes

CAPS:
- reopen only with a different proof object, not by tuning thresholds/priors.

CPAG:
- reopen only with typed relation-level agreement, contradiction/specificity filtering, or a new agreement object.

BSGS/QBF:
- reopen only with a graph substrate whose edges are closer to reasoning transitions, or with stronger supervision.

D-PathRAG/CEE:
- reopen only with reliable answer hypotheses or direct edit supervision that handles hard-negative admission.

PCRS:
- reopen only if semantic need-unit construction and support/contradiction scoring are rebuilt, not merely reskinned.

## Core Research Lessons

1. The main validated bottleneck is evidence composition under a small reader budget, not raw retrieval reachability alone.

2. The strongest positive path is fixed-pool, demand-aware, dependency-bound composition.

3. Generic diversity, generic graph flow, generic agreement, and generic local verification are all too weak for the residual regime after strong retrieval.

4. Hard negatives are not random distractors; they often share entities, retriever consensus, local NLI support, or bridge-like structure.

5. Good local measurements are necessary but not sufficient. The final object must preserve joint set coherence, variable binding, and reader consumability.

6. LLMs help when used for local decomposition/extraction under a robust operator. They are risky when asked to perform brittle symbolic compilation, exhaustive candidate generation, or exact predicate canonicalization.

7. Negative results converged toward the same boundary: passage-level or candidate-level signals often fail to distinguish gold evidence from plausible wrong evidence once the base substrate is already strong.

8. If a future route is attempted, it should not be another local reranker, hard gate, or threshold sweep. It needs a new object that directly represents coherent evidence sets, typed bindings, or reader-robust answer discrimination.

## Record Inventory Covered

High-level coordination:
- `00_north_star.md`
- `01_claim_ledger.md`
- `02_decision_log.md`
- `03_experiment_board.md`
- `04_result_registry.md`

Oracle and early selectors:
- `oracle_select_cross_dataset_summary_20260329_resume.md`
- `05_non_oracle_bridge_greedy.md`
- `06_bridge_beam_search.md`
- `simple_g1_precision_failure_audit_20260405.md`

PCRS / requirement route:
- `07_pcrs_rag_v1.md`
- `musique_requirement_beam_reserve_diagnostics_20260402.md`
- `pcrs_v2_needunit_review_20260403.md`
- `pcrs_v2_needunit_smoke40_analysis_20260403.md`
- `pcrs_v2_shortlist_trace_diagnosis_20260403.md`
- `pcrs_v2_atomic_hybrid_smoke10_20260403.md`
- `pcrs_v2_selector_ceiling_smoke10_20260403.md`
- `pcrs_v2_q7_forced_final_positive_only_20260403.md`
- `pcrs_v2_stage_isolation_probes_smoke10_20260403.md`
- `pcrs_v2_live_frontier_ablation_smoke10_20260403.md`
- `pcrs_v2_bridge_seed_smoke10_gate_20260403.md`
- `pcrs_v2_bridge_probe_seed_round1_20260403.md`
- `pcrs_v2_bridge_label_spec_20260403.md`
- `pcrs_v2_bridge_supervision_plan_20260403.md`
- `pcrs_v2_pool_coverage_census_q6_20260403.md`
- `pcrs_v2_q6_variable_binding_bonus_probe_20260404.md`
- `pcrs_v2_q6_variable_preservation_probe_20260404.md`
- `pcrs_v2_q6_continuity_probe_20260404.md`
- `pcrs_v2_q6_canonicalization_audit_20260404.md`
- `pcrs_v2_relation_coverage_probe_q6_20260404.md`
- `pcrs_v2_q6_graph_connectivity_audit_20260404.md`
- `pcrs_v2_q6_source_admission_probes_20260404.md`
- `pcrs_v2_q6_e_combo_probe_20260404.md`
- `pcrs_v2_q6_seed_target_bridge_probe_20260404.md`
- `pcrs_v2_q6_beam_state_audit_20260404.md`
- `pcrs_v2_varfix_offlinehybrid_q6_matrix_20260404.md`
- `pcrs_v2_varfix_clean_baseline_20260404.md`
- `pcrs_v2_two_query_q6_style_validation_20260404.md`
- `pcrs_v2_failure_family_scan_smoke40_20260404.md`
- `pcrs_v2_family_aware_gated_smoke40_20260404.md`

Coverage / alias / backup:
- `09_musique_coverage_alias_audit_20260406.md`
- `10_requirement_aware_strongest_backup.md`

DAEC / DtC:
- `11_diversity_selector_baselines_20260421.md`
- `12_dtc_supplemental_experiments_20260421.md`
- `13_musique_4doc_regression_analysis_20260421.md`
- `14_dtc_nv_full1000_rank_prior_20260422.md`
- `15_demand_gate_ablation_20260423.md`
- `satisfiable_by_prompt_sweep_20260424.md`
- `16_layer1_retriever_agnostic_composition_20260424.md`
- `17_layer1_followup_taxonomy_significance_20260424.md`

Alternative operators and later residual routes:
- `15_qbf_pilot_negative_result_20260424.md`
- `18_bsgs_week0_week1_plan_20260425.md`
- `19_bsgs_negative_diagnostic_20260425.md`
- `20_pathA_setr_widened_pool_plan_20260426.md`
- `21_dpathrag_direction_20260426.md`
- `20_dpathrag_cee_ccee_negative_diagnostic_20260427.md`

CAPS / CPAG / Answer-Contrastive:
- `21_caps_day0_nli_sanity_20260427.md`
- `22_caps_day1_candidate_recall_failure_20260427.md`
- `23_caps_day1_5_candidate_v2_failure_20260427.md`
- `24_caps_day2_proof_separability_20260427.md`
- `25_cpag_day1_agreement_failure_20260427.md`
- `26_answer_contrastive_verifier_20260427.md`
- `27_residual_route_lessons_20260427.md`
- `29_daec_alr_consistency_probe_20260427.md`
- `30_daec_alr_step1_plan.md`
- `31_daec_alr_step1_single_edit_failure_20260427.md`

Notes:
- `08_pcrs_v2_parser_compiler_spec.md` is referenced in `03_experiment_board.md`, but the file was not present in the current working tree at synthesis time.
- Large raw JSON/JSONL reports are not exhaustively re-listed here; this synthesis follows the markdown memory records and their canonical report references.
