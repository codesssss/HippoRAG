# Proposal C: AREC-RAG as Training-Free Answer-Residual Evidence Closure

Date: 2026-04-28

Status: smoke red-light after MuSiQue limit=100 controls. This proposal is
retained as an exploratory record, not an active execution plan.

This document proposes a train-free alternative after the DAEC-DAPG negative
gates. It does not supersede `docs/daec_l1_findings_decision_20260428.md`.
The initial AREC smoke gates did not pass: silver-oracle obligations show a
closure/projection signal, but generated obligations are negative and the
residual retrieval contribution is small after fixed-pool controls.

Current source-of-truth relationship:

- `docs/daec_l1_findings_decision_20260428.md` remains the current execution
  decision.
- `docs/daec_dapg_proposal_b_demand_lifted_absorption_20260428.md` remains an
  archived negative proposal.
- This file is Proposal C: a train-free closed-loop retrieval hypothesis that
  currently failed the generated-obligation smoke gate.

## 2026-04-28 Smoke Update: Red-Light Decision

MuSiQue limit=100 smoke runs produced a clear split between a silver-oracle
ceiling and the train-free generated-obligation version.

Main residual retrieval smoke:

| Stage | AREC Missing Hit | Raw Missing Hit | CoT Missing Hit | Initial SC@5 | Final SC@5 | Initial Recall@5 | Final Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Day 0 silver oracle | 0.0320 | 0.0080 | 0.0220 | 0.2900 | 0.5000 | 0.6458 | 0.7675 |
| Day 2 generated | 0.0160 | 0.0080 | 0.0220 | 0.2900 | 0.2400 | 0.6458 | 0.4892 |

Closure signal:

| Stage | Generated Closure | Silver Oracle Closure | Initial SC@5 | Active Obligations |
| --- | ---: | ---: | ---: | ---: |
| Day 1 generated | 0.8342 | 2.1344 | 0.2900 | 2.7600 |

Fixed-pool projection control, using cached NLI support matrices and no new
residual retrieval:

| Source | Projection Pool | Recall@5 | SC@5 | Initial Recall@5 | Initial SC@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| generated | top-5 | 0.6458 | 0.2900 | 0.6458 | 0.2900 |
| generated | top-20 | 0.4933 | 0.2300 | 0.6458 | 0.2900 |
| generated | top-100 | 0.3042 | 0.1300 | 0.6458 | 0.2900 |
| silver oracle | top-5 | 0.6458 | 0.2900 | 0.6458 | 0.2900 |
| silver oracle | top-20 | 0.7417 | 0.4500 | 0.6458 | 0.2900 |
| silver oracle | top-100 | 0.6767 | 0.3700 | 0.6458 | 0.2900 |

Interpretation:

- Silver-oracle AREC improves SC@5 from 0.29 to 0.50, but a top-20 fixed-pool
  closure projection already reaches 0.45. The residual retrieval component
  contributes at most about +0.05 SC@5 beyond the fixed-pool selector control
  in this smoke.
- Generated obligations are not merely weak. They are harmful under projection:
  top-20 generated closure drops SC@5 from 0.29 to 0.23, and top-100 drops it
  to 0.13.
- Generated residual retrieval also loses to the local CoT-style query control:
  0.0160 missing-hit rate vs 0.0220.
- The current evidence does not support AREC as a retrieval paper. At most, it
  suggests that high-quality obligations can be a useful selector/projection
  signal, which folds back into the DAEC-L1 fixed-pool story.

Generated-obligation audit packet:

- `reports/dpathrag/arec_rag_obligation_failure_audit_musique_limit100_20260428.md`
- `reports/dpathrag/arec_rag_obligation_failure_audit_musique_limit100_rows_20260428.jsonl`

The audit packet samples 30 high-risk generated top-20 projection cases. It is
not a manual label set yet, but heuristic flags already show the dominant risk:
27 sampled obligations contain the candidate-answer string and 8 are
duplicate/paraphrase candidates. The sample cases include severe
wrong-answer-amplification examples where an initially support-complete top-5
context becomes support-incomplete after generated-obligation projection.

Decision:

```text
Do not expand AREC to full1000.
Do not run K=3 multi-hypothesis yet.
Do not tune prompts blindly.
Do not claim answer-residual retrieval as the main mechanism.
Keep AREC as a red-light exploratory record unless manual obligation audit
shows a prompt-fixable failure mode and a rerun beats CoT and fixed-pool controls.
```

If this line is revived, the minimum restart gates are:

1. Manual labels on the 30-case audit packet show that most failures are
   prompt-schema errors rather than inherent wrong-answer amplification.
2. A revised shared prompt improves generated top-20 projection above the
   initial 0.29 SC@5 baseline instead of below it.
3. Generated residual retrieval beats both raw-question second retrieval and a
   strong IRCoT-style CoT query under the same pool/budget.
4. Full residual retrieval clearly beats the corresponding fixed-pool closure
   projection; otherwise the contribution is selector-only.

## Executive Position

AREC-RAG is a training-free closed-loop evidence assembly method for multi-hop
RAG.

It should not be framed as:

```text
end-to-end trainable RAG
joint retriever-reader training
another graph retrieval method
another fixed-pool selector
```

It should be framed as:

```text
training-free closed-loop evidence closure
inference-time reader-feedback retrieval
answer-conditioned residual evidence acquisition
proof-obligation-guided multi-hop retrieval
```

One-sentence thesis:

```text
AREC-RAG retrieves not what is merely similar to the question, but what is still
missing for the current answer to be evidentially closed.
```

The method is train-free:

- frozen embedding retriever;
- frozen reader;
- frozen obligation generator;
- frozen verifier / NLI scorer;
- no retriever training;
- no reranker training;
- no reader fine-tuning;
- no RL;
- no dataset-specific learned routing.

It is "closed-loop" at inference time:

```text
question
  -> retrieve
  -> read
  -> generate answer-conditioned proof obligations
  -> verify obligation support
  -> retrieve residual evidence
  -> project evidence under reader budget
  -> read again
```

Avoid calling it "true E2E" in the paper title or abstract. If "end-to-end" is
used at all, qualify it as inference-time closed-loop rather than training
end-to-end.

## Why This Proposal Exists

The 2026-04-28 DAEC-DAPG diagnostics imply:

1. Strong dense query embeddings dominate local graph absorption in the current
   substrate.
2. Demand split / multi-channel propagation is weaker than single-query dense
   retrieval.
3. Delayed noisy-OR aggregation does not separate from channel-sum in raw
   settings.
4. DAEC-L1 still shows a real fixed-pool composition signal on 2Wiki, meaning
   final reader context matters.
5. MuSiQue failure taxonomy still shows retrieval / composition headroom, so a
   better train-free evidence acquisition loop may still be useful.

The lesson is not to add more graph structure at the retrieval operator. The
next plausible train-free route is to let the frozen reader expose what is
missing from its current answer, then retrieve evidence for that missing support.

DAEC-DAPG was question-demand-first:

```text
q -> requirements / bindings -> phi[i,b,d]
```

AREC-RAG is answer-residual-first:

```text
q, S_t -> candidate answer a_t -> proof obligations O(a_t)
       -> unsupported obligations -> residual retrieval
```

The key question becomes:

```text
Does answer-conditioned residual retrieval find missing support documents better
than raw-question second retrieval or CoT-step retrieval?
```

## Relation To Prior Work

AREC-RAG is close to dynamic and corrective RAG methods. The paper must treat
these as direct pressure, not peripheral related work.

| Method line | Overlap | Required distinction |
| --- | --- | --- |
| IRCoT | Alternates reasoning and retrieval using generated CoT steps | AREC retrieves from unsupported answer proof obligations, not next reasoning text |
| FLARE | Uses low-confidence predicted future text to trigger retrieval | AREC uses proof-obligation closure gaps, not token-confidence heuristics |
| DRAGIN | Dynamic retrieval from LLM information needs | AREC concretizes information need as answer-conditioned obligations with verifier support scores |
| Self-RAG | Retrieval / critique / use decisions via trained reflection tokens | AREC uses frozen modules and no reflection-token training |
| RARR | Generate content, find attribution, revise unsupported content | AREC turns attribution gaps into the next retrieval action for multi-hop QA |
| CRAG / corrective RAG | Evaluates and corrects retrieved documents | AREC scores support at the obligation-document level and optimizes closure |
| Claim-level RAG verification | Decomposes outputs into atomic claims and checks support | AREC uses claim verification to drive retrieval acquisition, not only evaluation |
| DAEC-L1 | Fixed-pool demand-aware projection | AREC goes beyond fixed-pool projection if residual retrieval adds missing support outside the initial pool |

References to verify during writing:

- IRCoT: `https://arxiv.org/abs/2212.10509`
- FLARE: `https://arxiv.org/abs/2305.06983`
- DRAGIN: `https://arxiv.org/abs/2403.10081`
- Self-RAG: `https://arxiv.org/abs/2310.11511`
- RARR: `https://arxiv.org/abs/2210.08726`
- CRAG: `https://arxiv.org/abs/2401.15884`

The strongest novelty claim is not the noisy-OR formula. The strongest claim is:

```text
answer-conditioned proof obligations define a residual evidence need, and that
residual need retrieves missing multi-hop support better than question-only or
CoT-text retrieval under the same inference budget.
```

## Core Hypothesis

Multi-hop RAG failures often follow this pattern:

1. The initial retriever finds partial evidence.
2. The reader proposes a plausible answer.
3. One bridge claim, entity grounding claim, comparison claim, or attribution
   claim remains unsupported.
4. The final answer is wrong or unsupported because the evidence chain is not
   closed.

AREC-RAG assumes:

```text
The next retrieval query should target the unsupported proof obligations of the
current answer, not the original question again.
```

This differs from question decomposition because the obligations are conditioned
on the candidate answer. It differs from CoT retrieval because each residual
query corresponds to a verifier-checkable support claim.

## Formal Setup

Let:

```text
q       = question
D       = corpus documents
S_t     = reader-facing evidence set at round t, S_t subset D
k       = final reader document budget
T       = residual retrieval rounds
Reader  = frozen answer generator
ObGen   = frozen proof-obligation generator
Ver     = frozen verifier / NLI scorer
Ret     = frozen candidate retriever
```

At round `t`, the reader proposes:

```text
a_t = Reader(q, S_t)
```

The obligation generator maps the question and candidate answer to a small set
of proof obligations:

```text
O_t(a_t) = {o_1, ..., o_m}
```

Each obligation should be an answer-specific, atomic, factual claim. Example:

```text
q: Which country is the director of film X from?
a: Italy

o1: Film X was directed by person Y.
o2: Person Y is from Italy.
o3: Therefore the director's country is Italy.
```

Retrieval should target factual obligations. Operator obligations such as
comparison, sorting, or aggregation may be retained as control claims, but they
should not directly trigger document retrieval unless they require external
facts.

## Obligation Schema

The prompt schema must prevent vague claims.

Each generated obligation must satisfy:

```text
1. It is a factual claim, not a generic reasoning statement.
2. It contains named entities or explicit descriptors from the question/answer.
3. It is checkable by one document, or by a small pair of documents if the task
   is inherently bridging.
4. It is answer-conditioned: changing the candidate answer should change at
   least one obligation when the support requirements differ.
5. It is not allowed to say "the answer is supported by the documents" or any
   similarly vacuous claim.
```

Recommended obligation generator output format:

```json
{
  "answer": "...",
  "obligations": [
    {
      "id": "o1",
      "claim": "...",
      "type": "factual|bridge|entity_grounding|comparison_control|aggregation_control",
      "retrieval_active": true,
      "query": "Question: ... Candidate answer: ... Evidence needed: ..."
    }
  ]
}
```

The `retrieval_active=false` obligations are checked for final reasoning but do
not create retrieval queries.

Additional constraints before smoke:

```text
1. Deduplicate obligations by normalized entity pair and predicate. Repeated
   paraphrases of the same fact must not be counted multiple times in closure.
2. Do not make an answer-restatement obligation retrieval-active. Claims like
   "therefore the answer is Italy" are control claims, not evidence queries.
3. Require each retrieval-active obligation to name the bridge entity or missing
   relation it needs evidence for.
4. Use one shared obligation-generation prompt across 2Wiki, HotpotQA, and
   MuSiQue. Dataset-specific prompt changes are not allowed in the main method.
```

Prompt-boundary rule:

```text
Prompt engineering is allowed only before the smoke is run. After the shared
prompt is frozen, all datasets use the same prompt. If a dataset-specific prompt
is later tested, it must be labeled as an ablation and cannot support the main
train-free claim.
```

## Frozen Verifier Support Matrix

For each retrieval-active obligation `o` and candidate document `d`, the frozen
verifier estimates:

```text
v(o,d) = P_Ver(SUPPORTS | o, d)
```

The verifier should use three labels:

```text
SUPPORTS
REFUTES
NOT_ENOUGH_INFO
```

Definition of SUPPORTS:

```text
The document must explicitly entail the obligation. Semantic relatedness is not
enough. If the document only mentions the same entities but does not establish
the claim, output NOT_ENOUGH_INFO.
```

If label logprobs are available, use the SUPPORTS label probability as `v(o,d)`.
If logprobs are unavailable, use hard labels:

```text
v(o,d) = 1 if label == SUPPORTS else 0
```

The paper must not overclaim that LLM label probabilities are calibrated
probabilities. It should report verifier calibration diagnostics separately.

### Verifier Independence

The verifier must be specified before implementation. This is not an engineering
detail; it affects the validity of the closure signal.

Preferred main option:

```text
Use a frozen NLI / entailment model as Ver, separate from the reader and
obligation generator.
```

Examples:

```text
DeBERTa-large-MNLI or another frozen entailment model with SUPPORTS / REFUTES /
NOT_ENOUGH_INFO mapping.
```

Why this is preferred:

- avoids the same LLM proposing an answer, decomposing it into obligations, and
  judging its own evidence;
- reduces self-confirmation risk;
- makes verifier calibration easier to audit;
- gives a cleaner ACL defense than a fully self-verifying pipeline.

Allowed fallback:

```text
Use the same frozen LLM as Ver only if the paper includes a self-verifier risk
section and reports disagreement with an independent NLI verifier on a sampled
set.
```

Required verifier metadata in every report:

```text
reader_model
obligation_generator_model
verifier_model
verifier_prompt_or_label_mapping
same_model_reader_and_verifier: true|false
```

## Evidence Closure Objective

For an evidence set `S`, obligation-level closure is:

```text
C_o(S) = 1 - prod_{d in S} (1 - v(o,d))
```

Interpretation:

```text
Obligation o remains open only if every document in S fails to support it.
```

Residual mass is:

```text
R_o(S) = 1 - C_o(S)
       = prod_{d in S} (1 - v(o,d))
```

The answer-level closure objective is:

```text
F_a(S) = sum_{o in O(a), retrieval_active(o)} C_o(S)
```

For a new document `d`, the marginal closure gain is:

```text
Delta(d | S)
  = F_a(S union {d}) - F_a(S)
  = sum_o v(o,d) * prod_{e in S} (1 - v(o,e))
  = sum_o v(o,d) * R_o(S)
```

This is the retrieval primitive:

```text
The value of a document equals the support it provides for obligations that are
still open.
```

Theory note:

If `0 <= v(o,d) <= 1` and `O(a)` is fixed, then each `C_o(S)` is monotone
submodular, and `F_a(S)` is monotone submodular. Greedy selection under a
cardinality budget has the standard `(1 - 1/e)` approximation guarantee for the
fixed-answer, fixed-obligation projection objective.

This theorem is a support point, not the main novelty.

## Residual Evidence Acquisition

AREC uses a two-stage retrieval loop because full-corpus verifier scoring is not
feasible.

### Candidate Generation

For each open or partially open retrieval-active obligation `o`, construct a
query:

```text
Question: {q}
Candidate answer: {a_t}
Evidence needed: {o}
```

Retrieve a candidate frontier:

```text
C_t(o) = Ret(query(q, a_t, o), top-M)
```

The round-level frontier is:

```text
C_t = union_o C_t(o) union S_t
```

Allowed candidate generators:

- frozen dense retriever such as NV-Embed;
- BM25;
- existing dense top-N pools;
- PropRAG / legacy fact graph PPR as a recall substrate.

The candidate generator is not the main contribution. The main contribution is
how residual obligations choose and project evidence from the frontier.

### Residual Gain Scoring

For each `d in C_t`, compute:

```text
Delta(d | S_t) = sum_o v(o,d) * R_o(S_t)
```

The top residual documents form an expanded workspace:

```text
W_{t+1} = S_t union TopM_{d in C_t} Delta(d | S_t)
```

### Reader-Budget Projection

The reader can only consume `k` documents, so the final evidence for the next
round is selected by greedy maximization of the same closure objective:

```text
S_{t+1} = argmax_{S subset W_{t+1}, |S| <= k} F_{a_t}(S)
```

Greedy version:

```text
S = []
repeat k times:
    add d with largest Delta(d | S)
```

This unifies residual retrieval and final projection under one closure objective.

## Multi-Hypothesis Branching

Single-answer AREC can amplify an incorrect initial answer. The main method
should use a small answer hypothesis set:

```text
A_t = {a_t^(1), ..., a_t^(K)}
```

For each hypothesis:

```text
O_t(a_t^(h))
residual queries
candidate frontier
closure scores
```

Then merge candidate documents into a shared workspace:

```text
W_{t+1} = S_t union union_h TopM_h residual_docs(a_t^(h))
```

Final projection can use either:

1. best-hypothesis closure:

```text
S_{t+1} = argmax_h greedy_topk(F_{a_t^(h)}, W_{t+1})
```

2. mixture closure:

```text
F_mix(S) = sum_h w_h F_{a_t^(h)}(S)
```

Recommended main option:

```text
K = 3 answer hypotheses
mixture closure with uniform w_h
```

Rationale:

- avoids overcommitting to the first wrong answer;
- stays train-free;
- keeps implementation simple;
- lets the final reader re-answer from the merged evidence set.

Ablate:

```text
K = 1, 3, 5
best-hypothesis closure vs mixture closure
```

Report:

```text
wrong initial answer -> correct final answer
correct initial answer -> wrong final answer
```

## Full Algorithm

```text
Algorithm: AREC-RAG

Input:
  question q
  corpus D
  frozen retriever Ret
  frozen reader Reader
  frozen verifier Ver
  frozen obligation generator ObGen
  evidence budget k
  residual rounds T
  answer hypotheses K
  per-obligation candidate budget M

1. S_0 = Ret(q, top-k)

2. for t = 0 ... T-1:

    2.1 Generate answer hypotheses:
        A_t = Reader(q, S_t, num_hypotheses=K)

    2.2 For each a in A_t:
        O_t(a) = ObGen(q, a)

    2.3 Verify current evidence:
        for each a, o in O_t(a), d in S_t:
            v(o,d) = Ver(o,d)
            R_o(S_t) = prod_{e in S_t} (1 - v(o,e))

    2.4 Residual candidate retrieval:
        for each retrieval-active o:
            C_t(o) = Ret(query(q, a, o), top-M)

    2.5 Verify candidate evidence:
        for each candidate d in union_o C_t(o):
            compute v(o,d)

    2.6 Score candidates:
        Delta(d | S_t) = sum_o v(o,d) R_o(S_t)

    2.7 Build workspace:
        W_{t+1} = S_t union top residual candidates

    2.8 Project to reader budget:
        S_{t+1} = greedy top-k under F_mix(S)

3. Final answer:
    a_star = Reader(q, S_T)

Output:
  answer a_star
  final evidence S_T
  obligations and closure diagnostics
```

Recommended defaults for smoke only:

```text
T = 1
K = 1 initially, then K = 3 if single-answer smoke is promising
k = 5
M = 20 per active obligation
max active obligations = 5
```

Recommended defaults for paper experiments if smoke passes:

```text
T = 2
K = 3
k = 5
M = 20 or 50
max active obligations = 5
```

## Why This Is Not Just A Selector

A fixed-pool selector does:

```text
top-100 docs -> top-k docs
```

AREC-RAG does:

```text
top-k docs
  -> reader answer hypotheses
  -> proof obligations
  -> unsupported obligations
  -> new retrieval queries
  -> new documents
  -> closure projection
  -> final reader
```

The paper must prove this difference empirically:

```text
residual retrieval must recover missing support documents outside the initial
top-k, and ideally outside the initial top-100.
```

If AREC only improves by selecting better documents inside the initial pool, the
claim should be downgraded to:

```text
LLM-verifier-assisted DAEC-L1 / fixed-pool evidence projection
```

## Required Baselines

Main train-free baselines:

| Method | Train-free | Feedback loop | Residual retrieval | Purpose |
| --- | ---: | ---: | --- | --- |
| Dense RAG | yes | no | none | one-shot retrieval baseline |
| Legacy fact graph PPR + NV-Embed | yes | no | none | current strong non-feedback baseline |
| DAEC-L1 | yes | no | fixed-pool projection | existing positive fixed-pool line |
| raw-question second retrieval | yes | yes | original question | controls for one more retrieval call |
| answer-only second retrieval | yes | yes | answer text | controls for answer-conditioned retrieval without obligations |
| CoT-step retrieval / IRCoT-style | yes | yes | generated reasoning step | strongest train-free dynamic retrieval comparator |
| FLARE-style next-sentence retrieval | yes | yes | predicted next sentence / low confidence | active retrieval comparator if practical |
| AREC fixed-pool | yes | yes | none, top-100 only | tests whether gains are only selector gains |
| AREC full | yes | yes | proof-obligation residual | main method |

Self-RAG should be discussed as related work because it trains reflection tokens
and is not under the same strict train-free constraint. Include it as a baseline
only if a frozen-prompt reproduction is meaningful and clearly labeled.

IRCoT baseline requirement:

```text
The CoT-step retrieval baseline must use the official released IRCoT prompt /
configuration when available, not an ad hoc weak prompt. Record the exact prompt
file or config name in the report.
```

If the official prompt cannot be used directly, the fallback prompt must be
documented and reviewed before smoke. Otherwise an AREC win over a weak CoT
baseline is not paper evidence.

AREC fixed-pool requirement:

```text
AREC fixed-pool must appear next to AREC full in the main mechanism table.
```

Interpretation:

- if fixed-pool AREC and full AREC are close, the contribution is mainly
  verifier-assisted projection / selector behavior;
- if full AREC clearly beats fixed-pool AREC by retrieving new missing support,
  the contribution can be framed as residual evidence acquisition.

## Required Ablations

| Variant | Purpose |
| --- | --- |
| AREC full | main method |
| no residual retrieval | initial retrieval + final reader only |
| raw-question second retrieval | controls for extra retrieval budget |
| answer-only query | controls for candidate answer without proof obligations |
| CoT-query retrieval | direct IRCoT-style comparator |
| obligation query without verifier residual | tests query benefit without closure weighting |
| verifier residual with fixed pool | tests whether full retrieval contributes beyond reranking |
| single hypothesis K=1 | tests error amplification risk |
| multi hypothesis K=3/5 | tests wrong-answer robustness |
| hard verifier labels | tests whether logprob scoring is necessary |
| oracle obligations | estimates obligation generator bottleneck |
| oracle verifier | estimates verifier bottleneck |

The two most important comparisons:

```text
raw-question second retrieval
vs
proof-obligation residual retrieval
```

and:

```text
fixed-pool residual selection
vs
full-corpus residual retrieval
```

## Metrics

Do not evaluate only answer EM/F1.

Answer metrics:

```text
Answer EM
Answer F1
Hop-bucket EM/F1 for 2-hop / 3-hop / 4-hop
```

Evidence metrics:

```text
Support Recall@k
Support-Complete@k
Noise Rate@k
```

Residual retrieval metrics:

```text
Residual Hit Rate_t = |Z_t intersect G_missing| / |Z_t|
New Gold Outside Initial Top-k
New Gold Outside Initial Top-100
```

Closure diagnostics:

```text
Closure score F_a(S_t)
Evidence closure curve: Support-Complete@k(t)
Open obligation count by round
Closure delta from t to t+1
```

Verifier diagnostics:

```text
Verifier Support AUPRC against gold support docs
False SUPPORTS rate on distractors
SUPPORTS / REFUTES / NEI confusion matrix on sampled obligations
Closure score correlation with answer correctness
Closure score correlation with support completeness
```

Failure transition diagnostics:

```text
wrong initial answer -> correct final answer
correct initial answer -> wrong final answer
support-incomplete -> support-complete
support-complete -> support-incomplete
```

Cost metrics:

```text
reader calls / query
verifier calls / query
retriever calls / query
tokens / query
latency / query
```

## Compute Budget Estimation

AREC can be expensive because verifier scoring is obligation-by-document. The
budget must be estimated before running Day 2 / Day 3.

Approximate per-query verifier calls:

```text
K = answer hypotheses
O = active obligations per hypothesis
k = current evidence budget
M = retrieved documents per obligation

candidate docs per hypothesis ~= k + O * M
verifier calls per query ~= K * O * (k + O * M)
```

Default paper-oriented setting:

```text
K = 3
O = 5
k = 5
M = 20

verifier calls / query ~= 3 * 5 * (5 + 5 * 20)
                           = 1575
```

For 1000 examples:

```text
~1.6M verifier calls for one method run
```

This is feasible only with batching and careful candidate pruning. It is not a
small add-on to dense RAG.

Smoke-safe defaults:

```text
Day 0 / Day 1:
  K = 1
  O <= 5
  M = 0 for closure-only scoring

Day 2:
  K = 1
  O <= 5
  M = 10
  verifier only over workspace_top_30 when possible

After positive smoke:
  test K = 3 and M = 20
```

Cost fallback:

```text
If verifier throughput is too low, keep K=1 as the main smoke setting and move
K=3 to an ablation. Do not silently drop the multi-hypothesis branch from the
proposal; report the cost constraint explicitly.
```

Reports must include:

```text
K
O average / max
M
candidate_count_before_verification
candidate_count_after_pruning
verifier_batch_size
verifier_examples_per_second
total_verifier_calls
```

## Minimum Smoke Plan

Run the smoke before building full AREC.

### Day 0: Baseline And Oracle Setup

Day 0 is a hard setup gate. Do not run AREC smoke against weak baselines.

Tasks:

1. Reproduce the IRCoT-style CoT-step retrieval baseline with the official
   released IRCoT prompt / configuration where available.
2. Save the exact prompt file, config name, and retrieval query text used for
   each example.
3. Create a small oracle-obligation ceiling set.

Oracle obligation ceiling:

```text
Sample size: 50 to 100 queries
Priority: MuSiQue 3-hop and 4-hop, plus a small 2Wiki bridge slice
Annotation: manually write 2 to 5 verifier-checkable obligations per query
            from the gold support chain
```

Purpose:

```text
Measure whether the AREC closure machinery can work when obligations are good.
If oracle obligations do not produce useful closure / residual retrieval
signals, generated obligations will not save the method.
```

Day 0 outputs:

```text
official_ircot_prompt_path_or_fallback_reason
cot_step_queries
oracle_obligations
oracle_closure_scores
oracle_residual_hit_rate
```

Day 0 go signal:

```text
Oracle obligations produce closure scores that separate support-complete from
support-incomplete cases, and oracle residual queries can retrieve missing gold
support at a rate above raw-question second retrieval.
```

Day 0 no-go signal:

```text
Even oracle obligations fail to predict missing support or retrieve missing
gold. In that case, stop AREC before generated-obligation experiments.
```

### Day 1: Does Generated Closure Predict Failure?

Dataset:

```text
MuSiQue limit=100
optionally 2Wiki limit=100 for a cleaner bridge-style contrast
```

Procedure:

1. Use dense / current strong baseline top-5 evidence.
2. Run frozen reader to generate answer.
3. Generate 2 to 5 proof obligations for the answer.
4. Run verifier over current top-5 documents.
5. Compute closure score:

```text
F_a(S_0) = sum_o C_o(S_0)
```

6. Bucket examples by high / medium / low closure.
7. Compare buckets against:

```text
answer correct vs wrong
support-complete vs incomplete
```

Go signal:

```text
low closure strongly predicts wrong answers and support-incomplete cases
```

No-go signal:

```text
closure is uncorrelated with correctness and support completeness
```

Additional required comparison:

```text
generated-obligation closure
vs
oracle-obligation closure on the Day 0 annotated subset
```

This identifies whether the bottleneck is the obligation generator or the
closure/verifier mechanism.

### Day 2: Does Residual Retrieval Find Missing Support?

Dataset:

```text
support-incomplete cases from MuSiQue limit=100
```

Procedure:

1. For each open obligation, build residual query:

```text
Question + Candidate answer + Evidence needed
```

2. Retrieve top-M documents.
3. Compute missing-support hits.
4. Compare against raw-question second retrieval and official-prompt
   IRCoT-style CoT-step retrieval.

IRCoT-style baseline constraint:

```text
Use the official released IRCoT prompt / configuration if available. If a local
fallback is necessary, freeze it before seeing AREC results and include the full
prompt in the report.
```

Primary smoke metric:

```text
Residual Missing-Support Hit Rate
```

Go signal:

```text
proof-obligation residual retrieval hits missing gold support more often than
raw-question second retrieval and CoT-step retrieval, especially on MuSiQue
3/4-hop.
```

No-go signal:

```text
obligation residual retrieval mostly retrieves paraphrases, answer-string
distractors, or semantically related but non-supporting documents.
```

### Day 3 Optional: Single-Round Full Closure

If Day 1 and Day 2 pass:

1. Build `S_1` using residual retrieval and closure projection.
2. Run final reader.
3. Compare:

```text
initial dense top-5
raw-question second retrieval
CoT-step retrieval
AREC T=1 K=1
```

Report:

```text
Answer EM/F1
Support-Complete@5
Residual Hit Rate
correct->wrong and wrong->correct transitions
```

## Go / No-Go Criteria

Promote AREC to full implementation only if:

1. official-prompt IRCoT-style retrieval has been reproduced or a documented
   fallback has been frozen before AREC results are inspected;
2. oracle obligations produce useful closure and missing-support retrieval
   signals on the annotated subset;
3. generated closure score correlates with answer correctness and support
   completeness;
4. residual obligation retrieval has higher missing-support hit rate than
   raw-question second retrieval;
5. residual obligation retrieval is competitive with or better than
   official-prompt CoT-step retrieval;
6. AREC improves Support-Complete@5, not only Recall@20;
7. some newly retrieved gold support comes from outside the initial top-k, and
   preferably outside the initial top-100;
8. correct-to-wrong transitions do not erase wrong-to-correct gains;
9. compute cost is measured and remains feasible for at least one full1000
   dataset run.

Stop or downgrade if:

1. oracle obligations fail the closure / residual retrieval ceiling test;
2. verifier frequently marks distractors as SUPPORTS;
3. obligations are vague, duplicated, answer-restatements, or not
   retrieval-active;
4. residual retrieval mainly retrieves answer-string documents without missing
   bridge support;
5. AREC fixed-pool and AREC full are indistinguishable;
6. MuSiQue 3/4-hop support-complete does not move;
7. costs are too high relative to gains;
8. official-prompt IRCoT-style retrieval matches or beats AREC under the same
   retrieval budget.

Fallback if no-go:

```text
Do not build a full AREC paper.
Use the smoke as negative evidence and return to DAEC-L1 Findings framing.
Optionally keep verifier-assisted fixed-pool projection as a DAEC-L1 ablation.
```

## Paper Contribution If Smoke Passes

Use three contributions:

1. **Answer-conditioned proof obligations.**

   Instead of decomposing the question upfront, AREC decomposes the current
   candidate answer into verifier-checkable support obligations:

   ```text
   (q, S_t) -> a_t -> O(a_t)
   ```

   This shifts the retrieval target from question relevance to answer support
   closure.

2. **Residual evidence acquisition by closure marginal gain.**

   AREC defines obligation closure:

   ```text
   C_o(S) = 1 - prod_{d in S} (1 - v(o,d))
   ```

   and uses the exact marginal gain:

   ```text
   Delta(d | S) = sum_o v(o,d) prod_{e in S} (1 - v(o,e))
   ```

   to drive both follow-up retrieval and reader-budget projection.

3. **Controlled train-free closed-loop evaluation.**

   AREC is evaluated against same-budget question-retrieval, answer-retrieval,
   CoT-retrieval, and fixed-pool projection baselines using evidence metrics
   such as Support-Complete@k, Residual Hit Rate, and closure curves.

Do not claim:

```text
first closed-loop RAG method
first claim verification RAG method
first submodular RAG selector
end-to-end trained retrieval
universal multi-hop retriever
```

Allowed claim if smoke and full runs pass:

```text
For training-free multi-hop QA, answer-conditioned proof-obligation residuals
provide a more effective second-round retrieval target than the original
question or generated reasoning text under matched inference budgets.
```

## ACL Reviewer Risk Register

| Risk | Why it matters | Required evidence |
| --- | --- | --- |
| "This is RARR + IRCoT" | The building blocks are close to prior work | Same-budget residual retrieval beats CoT-step and answer-only retrieval |
| Wrong-answer amplification | Answer-conditioned retrieval may support the wrong answer | K-hypothesis ablation and transition table |
| Verifier false support | Semantic relatedness may be mistaken for entailment | verifier AUPRC, false SUPPORTS rate, sampled confusion matrix |
| Self-verification loop | Same LLM may answer, generate obligations, and verify itself | independent NLI verifier or self-verifier disagreement audit |
| Only a selector | Gains may come from projection inside top-100 | show new gold outside initial top-k/top-100 |
| Cost hidden | LLM verifier calls are expensive | report calls, tokens, latency |
| Prompt engineering | Obligations may be arbitrary | obligation quality audit and oracle obligation ablation |
| Weak IRCoT comparator | AREC may only beat a weak CoT prompt | official IRCoT prompt/config or documented pre-frozen fallback |
| No MuSiQue gain | Multi-hop claim depends on hard 3/4-hop cases | hop-bucket closure curves and Support-Complete@5 |

## Review Questions For Claude

Ask Claude to focus on:

1. Is the novelty over IRCoT / RARR / CRAG / FLARE / DRAGIN defensible?
2. Is the claim "training-free closed-loop evidence closure" too broad or
   appropriately scoped?
3. Is multi-hypothesis branching necessary in the main method, or should it be
   an ablation?
4. Are the verifier probability and submodular closure definitions rigorous
   enough?
5. Should the main verifier be an independent NLI model, or is a same-LLM
   verifier defensible with disagreement audits?
6. Is Day 0 oracle-obligation ceiling sufficient, or does it need a larger
   annotated subset?
7. What is the minimum smoke package that can kill or validate the idea in two
   days?
8. What baseline would an ACL reviewer most likely demand that is missing here?
9. If smoke passes only on 2Wiki but not MuSiQue, is the paper still viable?
10. Are the compute defaults `K=1, M=10` for smoke and `K=3, M=20` for paper
    realistic, or should the main method be cost-capped from the start?

## Implementation Notes If Approved For Smoke

Recommended artifact names:

```text
scripts/arec_rag_smoke_closure.py
scripts/arec_rag_smoke_residual_retrieval.py
scripts/arec_rag_oracle_obligation_ceiling.py
reports/dpathrag/arec_rag_closure_musique_limit100_20260428.md
reports/dpathrag/arec_rag_closure_musique_limit100_20260428.json
reports/dpathrag/arec_rag_residual_musique_limit100_20260428.md
reports/dpathrag/arec_rag_residual_musique_limit100_20260428.json
reports/dpathrag/arec_rag_residual_musique_limit100_rows_20260428.jsonl
```

Suggested row-level JSONL fields:

```text
qid
dataset
hop_bucket
question
gold_answers
gold_titles
initial_selected_titles
initial_answer
initial_em
initial_f1
initial_support_recall
initial_support_complete
reader_model
obligation_generator_model
verifier_model
same_model_reader_and_verifier
shared_obligation_prompt_id
official_ircot_prompt_path
answer_hypotheses
obligations
oracle_obligations
active_obligation_count
verifier_scores_current
closure_score_initial
oracle_closure_score_initial
open_obligations
raw_question_second_titles
cot_second_titles
arec_residual_titles
raw_question_missing_hits
cot_missing_hits
arec_missing_hits
oracle_arec_missing_hits
new_gold_outside_initial_topk
new_gold_outside_initial_top100
projected_titles
final_answer
final_em
final_f1
final_support_recall
final_support_complete
transition_category
latency_seconds
reader_calls
verifier_calls
retriever_calls
token_count
K
M
candidate_count_before_verification
candidate_count_after_pruning
verifier_batch_size
verifier_examples_per_second
```

Transition categories:

```text
wrong_to_correct
correct_to_wrong
wrong_to_wrong
correct_to_correct
support_incomplete_to_complete
support_complete_to_incomplete
```

## One-Sentence Current Position

AREC-RAG is the most plausible train-free successor to the failed DAEC-DAPG
retrieval line because it changes the feedback signal from question-side graph
structure to answer-side evidence closure; however, it should not be promoted
beyond an exploratory proposal until closure prediction and residual missing
support retrieval pass the two-day smoke gates.
