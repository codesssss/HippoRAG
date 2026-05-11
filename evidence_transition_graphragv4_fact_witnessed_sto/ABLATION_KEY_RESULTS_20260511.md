# ETv4 Key Ablations 2026-05-11

## Protocol

| Item | Setting |
| --- | --- |
| Clean baseline | `query_grounded_sto_clean_mainline_v4` |
| Ablation 1 | `query_grounded_sto_no_multi_anchor_precision_v4` |
| Ablation 2 | `query_grounded_sto_raw_sto_adjacency_v4` |
| Datasets | 2Wiki, MuSiQue, HotpotQA |
| Limit | 100 queries per dataset |
| Index substrate | Reused clean-final GPT-4o-mini OpenIE and NV-Embed artifacts |
| Reader | `gpt-4o-mini` |
| Reader max tokens | 400 |
| Reader context | top-5 full passages |

## Retrieval Results

| Dataset | Method | R@5 | Delta vs clean | all-gold@5 |
| --- | --- | ---: | ---: | ---: |
| 2Wiki | Clean | 0.9400 | +0.0000 | 0.8500 |
| 2Wiki | No multi-anchor precision | 0.9400 | +0.0000 | 0.8500 |
| 2Wiki | Raw STO adjacency | 0.9400 | +0.0000 | 0.8500 |
| MuSiQue | Clean | 0.7358 | +0.0000 | 0.4300 |
| MuSiQue | No multi-anchor precision | 0.7325 | -0.0033 | 0.4200 |
| MuSiQue | Raw STO adjacency | 0.7358 | +0.0000 | 0.4300 |
| HotpotQA | Clean | 0.9550 | +0.0000 | 0.9100 |
| HotpotQA | No multi-anchor precision | 0.9550 | +0.0000 | 0.9100 |
| HotpotQA | Raw STO adjacency | 0.9550 | +0.0000 | 0.9100 |

## Reader QA Results

| Dataset | Method | R@5 | Delta R@5 | all-gold@5 | EM | Delta EM | F1 | Delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | Clean | 0.9400 | +0.0000 | 0.8500 | 0.6200 | +0.0000 | 0.6983 | +0.0000 |
| 2Wiki | No multi-anchor precision | 0.9400 | +0.0000 | 0.8500 | 0.6300 | +0.0100 | 0.7059 | +0.0076 |
| 2Wiki | Raw STO adjacency | 0.9400 | +0.0000 | 0.8500 | 0.6200 | +0.0000 | 0.6920 | -0.0063 |
| MuSiQue | Clean | 0.7358 | +0.0000 | 0.4300 | 0.4300 | +0.0000 | 0.5440 | +0.0000 |
| MuSiQue | No multi-anchor precision | 0.7325 | -0.0033 | 0.4200 | 0.4100 | -0.0200 | 0.5421 | -0.0019 |
| MuSiQue | Raw STO adjacency | 0.7358 | +0.0000 | 0.4300 | 0.4300 | +0.0000 | 0.5538 | +0.0098 |
| HotpotQA | Clean | 0.9550 | +0.0000 | 0.9100 | 0.6400 | +0.0000 | 0.7542 | +0.0000 |
| HotpotQA | No multi-anchor precision | 0.9550 | +0.0000 | 0.9100 | 0.6500 | +0.0100 | 0.7658 | +0.0116 |
| HotpotQA | Raw STO adjacency | 0.9550 | +0.0000 | 0.9100 | 0.6400 | +0.0000 | 0.7604 | +0.0062 |

## Top-5 Change Audit

| Dataset | Ablation | Top-5 changed | Gold gains | Gold losses | Clean R@5 | Ablation R@5 | Delta R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | No multi-anchor precision | 0 | 0 | 0 | 0.9400 | 0.9400 | +0.0000 |
| 2Wiki | Raw STO adjacency | 0 | 0 | 0 | 0.9400 | 0.9400 | +0.0000 |
| MuSiQue | No multi-anchor precision | 3 | 0 | 1 | 0.7358 | 0.7325 | -0.0033 |
| MuSiQue | Raw STO adjacency | 1 | 0 | 0 | 0.7358 | 0.7358 | +0.0000 |
| HotpotQA | No multi-anchor precision | 4 | 0 | 0 | 0.9550 | 0.9550 | +0.0000 |
| HotpotQA | Raw STO adjacency | 0 | 0 | 0 | 0.9550 | 0.9550 | +0.0000 |

## Interpretation

| Question | Evidence | Current Answer |
| --- | --- | --- |
| Is the multi-anchor precision rule a major trick? | Removing it only changes retrieval on MuSiQue by -0.0033 R@5 and -0.0100 all-gold@5. | It is not the main source of performance, but it prevents one MuSiQue gold-coverage loss. |
| Does the current run prove fact-witness filtering is the main claim? | Raw STO adjacency matches clean R@5/all-gold on all three datasets. | No. This ablation weakens the claim that witness filtering alone drives retrieval gains. |
| Does raw adjacency change top-5 materially? | Only one MuSiQue query changes top-5; gold coverage is unchanged. | The current readout is dominated by graph admission/source order more than by witness filtering. |
| What should change in the paper claim? | Fact witness does not show retrieval-side necessity under this ablation. | The claim should shift from "fact witness is the sole core" toward "query-local STO document readout with constrained symbolic root promotion"; fact witness may be an interpretability/precision constraint unless stronger ablations show otherwise. |

## Immediate Next Decision

Do not add another rule. The next diagnostic should isolate whether the real
contributor is:

| Candidate contributor | Diagnostic |
| --- | --- |
| Query-local STO admission | Compare dense top5 vs STO readout with same entry |
| Branch readout | Disable branch readout while keeping same admission |
| Variable-flow traversal | Disable variable-flow traversal |
| Fact witness | Already tested here; current evidence is weak for retrieval-side necessity |
