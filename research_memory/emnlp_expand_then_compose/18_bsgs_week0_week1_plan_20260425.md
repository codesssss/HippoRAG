# BSGS Week 0 + Week 1 Plan

Date: 2026-04-25

Branch:

- `bsgs-week0-week1`

Base checkpoint:

- `2555018 Add DAEC layer1 external-pool evaluation`

## Decision

Start a new method line:

```text
BSGS: Belief-State Graph Scratchpad for Multi-Hop RAG
```

The current branch is not a full BSGS implementation branch yet. It is a bounded validation branch:

```text
Week 0 = risk audit
Week 1 = oracle-slot operator validation
```

## Method Boundary

DAEC and BSGS should be related but not collapsed:

- `DAEC`: one-step set-posterior composer over final evidence set `S`.
- `BSGS`: multi-step node-marginal filtering approximation over proposition nodes.

Formal bridge:

```text
DAEC: P(S | q, o) proportional to P0(S | q) P(o | S, q)
BSGS: b_t(p) = sum_{S_t contains p} P(S_t | q, o_{1:t})
```

Paper-safe claim:

> BSGS approximates the intractable evidence-set posterior through tractable node-level marginal filtering.

## Week 0 Tasks

| ID | Task | Main output | Gate |
|---|---|---|---|
| W0-T0 | MuSiQue split audit | `reports/week0/split_audit.md/json` | local split and Layer-1 source identified |
| W0-T1 | IRCoT baseline | `reports/week0/ircot_baseline.md` | MuSiQue-50/200 EM/F1 + cost |
| W0-T2 | DeBERTa-NLI calibration | `reports/week0/nli_calibration.md/json` | ECE decides absorbing main vs ablation |
| W0-T3 | Oracle slot pipeline | `data/processed/musique_oracle_slots.jsonl` | oracle slots built without prompt contamination |
| W0-T4 | Qwen slot quality | `reports/week0/qwen_slot_quality.md/json` | slot recall decides latent-slot route |

Week 0 final report:

```text
reports/week0/decision.md
```

## Week 1 Task

Run minimal BSGS operator validation:

```text
dataset: MuSiQue-200
slot mode: oracle-slot
binding: hard normalized string / alias match
verifier: calibrated DeBERTa-NLI
transition: slot-marginalized sparse transition
evidence selection: posterior top-k
```

Disabled in Week 1:

- latent slot;
- soft binding;
- DPP;
- Qwen-generated DAG;
- LLM verifier.

Week 1 outputs:

```text
reports/week1/operator_validation.md
reports/week1/operator_validation.json
```

## Week 1 Gate

Hard mechanism gate:

BSGS-oracle-slot must beat DAEC or IRCoT on at least one:

- bridge entity recall by at least 5 pp;
- evidence path recall by at least 5 pp;
- answer-in-context-but-fail rate clearly decreases.

Soft answer gate:

| Grade | Condition | Decision |
|---|---|---|
| Green | F1 >= max(DAEC, IRCoT) + 1 pp | Continue to Week 2 latent-slot |
| Yellow | Mechanism improves but F1 is flat/slightly down | Continue but prioritize reader/evidence formatting |
| Red | Mechanism does not improve | Stop full BSGS route |

## Implementation Layout

New code should be additive:

```text
src/bsgs/
tests/bsgs/
scripts/bsgs_*.py
reports/week0/
reports/week1/
```

Do not rewrite the DAEC main path.

## Immediate Next Step

Implement and run:

```text
scripts/bsgs_split_audit.py
```

It must answer:

- whether current local MuSiQue is MuSiQue-Ans or MuSiQue-Full;
- whether current Layer-1 MuSiQue-1000 matches the intended IRCoT-style answerable subset;
- which split to use for oracle-slot diagnostics;
- how to prevent oracle decomposition from contaminating latent-slot prompt evaluation.
