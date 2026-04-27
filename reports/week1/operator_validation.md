# Week 1 Oracle-Slot BSGS Operator Validation

## Config
- dataset: `musique`
- split: `local MuSiQue diagnostic`
- n: `1000`
- verifier: `calibrated DeBERTa-NLI if available; lexical likelihood fallback in this minimal runner`
- absorbing enabled: `False`
- binding: `hard normalized string / alias match`
- evidence selector: `posterior top-k`

## Main comparison
| Method | EM | F1 | Bridge Recall | Evidence Path Recall | AICBF | Latency |
|---|---:|---:|---:|---:|---:|---:|
| DAEC | `0.0000` | `0.0000` |  | `0.0000` |  | `0.0000` |
| IRCoT | `0.0000` | `0.0000` |  | `0.0000` |  | `0.0000` |
| BSGS-oracle-slot | `0.0000` | `0.0000` |  | `0.2268` |  | `0.0000` |

## Mechanism gate
- Passed: `True`
- Metric: `supporting_paragraph_recall`
- Delta: `0.2268`

## Answer gate
- Grade: `Yellow`

## Failure taxonomy
- bridge missing: pending manual analysis
- proposition extraction failure: pending manual analysis
- verifier false support: pending manual analysis
- reader failure: pending manual analysis
- entity binding error: pending manual analysis

## Decision
- `continue_bsgs_diagnostics`
