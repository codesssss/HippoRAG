# BSGS Week 0 + Week 1 Execution Plan

Date: 2026-04-25

Branch: `bsgs-week0-week1`

Base checkpoint:

- `2555018 Add DAEC layer1 external-pool evaluation`

## 1. Decision

This branch starts the BSGS line from the frozen DAEC Layer-1 checkpoint.

Final method boundary:

```text
DAEC = one-step set-posterior composer
BSGS = multi-step node-marginal filtering approximation
```

Do not frame DAEC and BSGS as equivalent probability spaces.

DAEC composes a final evidence set:

```text
P(S | q, o) proportional to P0(S | q) P(o | S, q)
```

BSGS avoids the combinatorial evidence-set posterior and tracks a node-level marginal:

```text
b_t(p) = sum_{S_t contains p} P(S_t | q, o_{1:t})
```

Paper-safe wording:

> BSGS is a tractable node-level marginal filtering approximation to the evidence-set posterior optimized by one-step composers such as DAEC.

## 2. Scope

Current stage:

```text
Week 0: risk audit
Week 1: oracle-slot BSGS operator validation
```

Do not implement full latent-slot BSGS in this branch before the Week 1 gate passes.

In scope:

- MuSiQue split audit.
- IRCoT-style compute-matched baseline.
- DeBERTa-NLI calibration and throughput check.
- Oracle-slot pipeline from MuSiQue gold decomposition.
- Qwen3-8B no-think slot generation quality evaluation.
- Minimal oracle-slot BSGS operator validation on the local MuSiQue-1000 protocol used by the DAEC/PropRAG/Dense comparisons.

Out of scope for Week 0/1:

- full latent-slot BSGS;
- Qwen-generated dependency DAG as the main path;
- soft binding;
- DPP evidence selection;
- full-corpus proposition extraction;
- LLM verifier;
- claiming oracle-slot results as non-oracle main results;
- modifying the frozen DAEC main path except for necessary read-only adapters.

## 3. Week 0 Tasks

### W0-T0: MuSiQue Split Audit

Goal:

Confirm which MuSiQue variant is used locally and by the current Layer-1 MuSiQue-1000 runs.

Audit questions:

- Is local `musique` data MuSiQue-Ans or MuSiQue-Full?
- What are the train/dev/test file paths?
- What are the train/dev/test sizes?
- Does the split contain unanswerable examples?
- Does it contain decomposition and supporting paragraph annotations?
- Which file/split produced current Layer-1 MuSiQue-1000?
- Is it aligned with IRCoT-style answerable-subset evaluation?
- Which split should be used for oracle-slot diagnostics?
- How do we avoid latent-slot prompt contamination from dev oracle decomposition?

Outputs:

```text
reports/week0/split_audit.md
reports/week0/split_audit.json
```

Required JSON:

```json
{
  "dataset_name": "musique",
  "variant": "MuSiQue-Ans or MuSiQue-Full or unknown",
  "train_size": 0,
  "dev_size": 0,
  "test_size": 0,
  "has_unanswerable": false,
  "has_decomposition": true,
  "has_supporting_paragraphs": true,
  "current_layer1_musique_1000_source": "",
  "recommended_eval_split": "",
  "notes": []
}
```

### W0-T1: IRCoT Baseline

Goal:

Run an IRCoT-style baseline as the compute-matched iterative comparator for BSGS.

Fixed protocol:

```text
model: Qwen3-8B no-think
retriever: same substrate as DAEC/BSGS diagnostic
max_iter: 3 or 4
top_k_per_iter: 5
reader: Qwen3-8B no-think
dataset: local MuSiQue-1000 protocol
smoke: optional limit-1/limit-50 plumbing check only; do not report as the protocol result
```

Outputs:

```text
reports/week0/ircot_musique1000.json
reports/week0/ircot_baseline.md
```

Metrics:

- answer EM;
- answer F1;
- retrieval recall;
- supporting paragraph recall;
- LLM calls per query;
- retrieved passages per query;
- latency per query.

### W0-T2: DeBERTa-NLI Calibration and Throughput

Goal:

Evaluate whether a DeBERTa-MNLI verifier can serve as a calibrated slot/proposition likelihood and semi-absorbing coefficient.

Verifier candidates:

```text
DeBERTa-v3-base-MNLI
DeBERTa-v3-large-MNLI
```

Do not use an LLM verifier.

Calibration data:

- Prefer sentence-level HotpotQA supporting facts.
- If only paragraph support is available, use weak labels and mark that limitation explicitly.

Calibration method:

- Main: temperature scaling.
- Optional: isotonic regression or beta calibration.

Outputs:

```text
reports/week0/nli_calibration.md
reports/week0/nli_calibration.json
reports/week0/nli_reliability_curve.png
reports/week0/nli_throughput.json
```

Metrics:

- ECE;
- Brier score;
- NLL;
- support / neutral / contradict distribution;
- throughput samples/sec;
- latency per 1000 pairs.

Decision:

```text
ECE <= 0.15:
  absorbing transition can be a Week 1 main operator component

ECE > 0.15:
  absorbing transition is an ablation only
  Week 1 main version uses clipped/uniform self-loop or no-absorbing baseline
```

Allowed claim:

```text
calibrated evidence conservation in expectation
```

Forbidden claim:

```text
guaranteed evidence conservation
```

### W0-T3: Oracle Slot Pipeline

Goal:

Build oracle slot sequences from MuSiQue gold decomposition for upper-bound diagnostics only.

Outputs:

```text
data/processed/musique_oracle_slots.jsonl
reports/week0/oracle_slot_pipeline.md
```

Record format:

```json
{
  "qid": "",
  "question": "",
  "answer": "",
  "slots": [
    {
      "slot_id": "s1",
      "slot_text": "",
      "input_variables": [],
      "output_variable": "x1",
      "gold_subquestion": "",
      "gold_answer": ""
    }
  ],
  "supporting_paragraphs": []
}
```

Rules:

- Oracle slots are for `BSGS-oracle-slot` only.
- Oracle decomposition must not be used to tune latent-slot prompts on the same dev split.
- Oracle-slot results must not be presented as a non-oracle main result.

### W0-T4: Qwen Slot Generation Quality

Goal:

Measure whether Qwen3-8B no-think can generate usable slot sets.

Main path:

```text
generate slot set + typed variable signatures
do not require dependency DAG
```

Prompt contract:

```text
Given a multi-hop question, decompose it into atomic information needs.

Return only a JSON list. Each item should be a slot with:
- slot_text: the atomic information need
- input_variables: known entities or variables required by this slot
- output_variable: the variable produced by this slot
- expected_answer_type: person/place/date/work/organization/number/other

Do not answer the question.
Do not infer facts.
Do not include dependency edges.
```

Outputs:

```text
reports/week0/qwen_slot_quality.md
reports/week0/qwen_slot_quality.json
data/processed/musique_qwen_slots.jsonl
```

Metrics:

- slot recall;
- slot precision;
- variable grounding accuracy;
- optional DAG edge F1.

Decision:

```text
slot recall >= 0.70:
  latent-slot route remains alive for Week 2

slot recall < 0.70:
  latent-slot route is downgraded
  Week 2 must not make a full latent-slot main claim

DAG edge F1 >= 0.50:
  DAG can be an ablation

DAG edge F1 < 0.50:
  DAG stays disabled
```

## 4. Week 1 Oracle-Slot Operator

Week 1 validates the BSGS operator under oracle slot order.

Fixed config:

```text
dataset: local MuSiQue-1000 diagnostic protocol
slot mode: oracle-slot
binding: hard normalized string / alias match
verifier: calibrated DeBERTa-NLI
transition: slot-marginalized sparse transition
absorbing: enabled only if ECE <= 0.15
evidence selection: posterior top-k
DPP: disabled
soft binding: disabled
latent slot: disabled
Qwen-generated DAG: disabled
```

### 4.1 Belief State

Maintain proposition-node posterior marginal:

```text
b_t(p) = P(p in S_t | q, o_{1:t})
```

Normalize each step:

```text
sum_p b_t(p) = 1
```

### 4.2 Slot Satisfaction

For slot `phi`:

```text
s_t(phi) = sum_i b_t(i) V(p_i => phi)
u_t(phi) = 1 - s_t(phi)
```

Week 1 uses:

```text
phi_t = oracle_slots[t]
```

### 4.3 Slot-Marginalized Sparse Transition

Do not build one dense transition matrix per slot.

First compute:

```text
M_t(j) = sum_phi pi_t(phi) L_phi(j)
```

Under Week 1 oracle slots:

```text
M_t(j) = L_{phi_t}(j)
```

Then row-normalize only over sparse graph edges:

```text
T_t(i, j) = E(i, j) M_t(j) / sum_{u in N(i)} E(i, u) M_t(u)
```

### 4.4 Semi-Absorbing Transition

If NLI calibration passes:

```text
T_tilde(i, j) = c_i 1[i=j] + (1 - c_i) T_t(i, j)
```

Prediction:

```text
b_hat_{t+1}(j) = sum_i b_t(i) T_tilde(i, j)
```

Observation:

```text
b_{t+1}(j) =
  b_hat_{t+1}(j) L(o_{t+1} | p_j, q, phi_t)
  / sum_u b_hat_{t+1}(u) L(o_{t+1} | p_u, q, phi_t)
```

### 4.5 Proposition Extraction

Week 1 may use retrieved passages plus Qwen extraction.

Prompt:

```text
Given the question, the current slot, and a retrieved passage,
extract atomic but context-rich propositions that may help satisfy the slot.

Rules:
1. Each proposition must be directly supported by the passage.
2. Keep enough context to avoid ambiguous triples.
3. Do not infer unstated facts.
4. Return the exact source span.
5. Return JSON only.

Question:
{question}

Current slot:
{slot_text}

Passage:
{passage}

Return:
[
  {
    "proposition": "...",
    "source_span": "...",
    "mentions": ["..."]
  }
]
```

Do not extract bare triples.

### 4.6 Binding

Week 1 binding:

```text
hard normalized entity match
```

Normalization:

- lowercase;
- strip punctuation;
- normalize whitespace;
- alias table if available.

No soft binding in Week 1.

### 4.7 Evidence Selection

Week 1 selection:

```text
posterior top-k
```

Score:

```text
TopK_p b_T(p) c_p
```

No DPP in Week 1.

## 5. Code Layout

Additive-only implementation layout:

```text
src/bsgs/
  __init__.py
  state.py
  transition.py
  verifier.py
  extractor.py
  slots.py
  binding.py
  metrics.py
  runner_oracle.py
  io.py
  prompts.py

tests/bsgs/
  test_belief_normalization.py
  test_absorbing_transition.py
  test_sparse_transition.py
  test_oracle_slot_loading.py
  test_slot_quality_metrics.py
  test_node_marginal_mapping.py

scripts/
  bsgs_split_audit.py
  bsgs_run_ircot_baseline.py
  bsgs_calibrate_nli.py
  bsgs_build_oracle_slots.py
  bsgs_eval_qwen_slots.py
  bsgs_run_oracle.py
  bsgs_week0_report.py
  bsgs_week1_report.py
```

Implementation rules:

- Do not large-scale refactor existing DAEC/HippoRAG code.
- Keep all new BSGS logic under `src/bsgs/`.
- Use existing reader/retriever adapters only through narrow scripts.
- Keep Week 0/1 outputs under `reports/week0/` and `reports/week1/`.

## 6. Core Data Structures

### PropositionNode

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PropositionNode:
    prop_id: str
    text: str
    source_doc_id: str
    source_span: str
    mentions: list[str] = field(default_factory=list)
    support_prob: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Slot

```python
@dataclass
class Slot:
    slot_id: str
    slot_text: str
    input_variables: list[str]
    output_variable: str
    expected_answer_type: str | None = None
    gold_subquestion: str | None = None
    gold_answer: str | None = None
```

### BeliefStateGraph

```python
@dataclass
class BeliefStateGraph:
    propositions: dict[str, PropositionNode]
    belief: dict[str, float]
    edges: dict[tuple[str, str], float]

    def normalize(self) -> None:
        ...

    def add_propositions(self, new_props: list[PropositionNode]) -> None:
        ...

    def predict(self, transition: dict[tuple[str, str], float]) -> dict[str, float]:
        ...

    def observe(self, likelihood: dict[str, float]) -> None:
        ...
```

## 7. Metrics

Answer metrics:

- answer EM;
- answer F1.

Retrieval/evidence metrics:

- supporting paragraph recall;
- evidence path recall;
- bridge entity recall;
- final evidence token count.

Mechanism metrics:

- bridge entity recall;
- evidence path recall;
- answer-in-context-but-fail rate;
- belief entropy;
- slot satisfaction curve.

Belief entropy:

```text
H(b_t) = -sum_i b_t(i) log b_t(i)
```

Answer-in-context-but-fail:

```text
final evidence contains gold answer/support
AND reader answer is wrong
```

## 8. Week 1 Gates

Do not use a single answer-F1 AND gate.

### Hard Mechanism Gate

On the local MuSiQue-1000 oracle-slot setting, BSGS-oracle-slot must satisfy at least one:

```text
bridge entity recall +5 pp over DAEC or IRCoT
OR evidence path recall +5 pp over DAEC or IRCoT
OR answer-in-context-but-fail rate clearly decreases
```

If the hard mechanism gate fails:

```text
stop BSGS full route
fallback to DAEC one-step paper
keep BSGS as negative diagnostic appendix
```

### Soft Answer Gate

| Grade | Condition | Decision |
|---|---|---|
| Green | Answer F1 >= max(DAEC, IRCoT) + 1 pp | Continue to Week 2 latent-slot |
| Yellow | Mechanism wins but answer F1 is flat or slightly down | Continue, but prioritize evidence formatting / reader consumption |
| Red | Mechanism does not win | Stop BSGS full route |

## 9. Required Reports

Week 0:

```text
reports/week0/decision.md
```

Required sections:

- split audit;
- IRCoT baseline;
- NLI calibration;
- oracle slot pipeline;
- Qwen slot quality;
- decision;
- enabled/disabled components;
- risks.

Week 1:

```text
reports/week1/operator_validation.md
reports/week1/operator_validation.json
```

Required sections:

- config;
- main comparison table: DAEC / IRCoT / BSGS-oracle-slot;
- mechanism gate;
- answer gate;
- failure taxonomy;
- decision.

## 10. Initial Execution Dependencies

`split_audit` is the only strict first step. It determines whether the
remaining Week 0 jobs should use `MuSiQue-Ans`, `MuSiQue-Full`, or the local
1000-example packaged subset.

After the split audit is clear, these jobs are independent enough to run in
parallel when compute is available:

- IRCoT baseline;
- DeBERTa-NLI calibration and throughput;
- oracle slot pipeline;
- Qwen slot quality evaluation.

Recommended dependency order:

1. `scripts/bsgs_split_audit.py`
2. launch independent Week 0 jobs:
   - `scripts/bsgs_run_ircot_baseline.py`
   - `scripts/bsgs_calibrate_nli.py`
   - `scripts/bsgs_build_oracle_slots.py`
   - `scripts/bsgs_eval_qwen_slots.py`
3. `scripts/bsgs_week0_report.py`

Only after `reports/week0/decision.md` exists and passes review:

4. `scripts/bsgs_run_oracle.py`
5. `scripts/bsgs_week1_report.py`

## 11. Current Status

Status at branch creation:

```text
planning only
no BSGS code implemented yet
no Week 0/Week 1 jobs launched yet
```

The first implementation task is `W0-T0: MuSiQue split audit`.
