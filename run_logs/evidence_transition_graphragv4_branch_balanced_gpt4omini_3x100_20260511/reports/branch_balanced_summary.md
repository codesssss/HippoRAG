# Evidence Transition GraphRAG V4 Branch-Balanced Summary

## Method Change

| Component | Logic |
| --- | --- |
| Query grounding | Exact query mentions remain symbolic STO roots; non-contiguous token-overlap anchors must also be corpus title endpoints. |
| Branch trigger | Branch-balanced readout runs only when the source-prior prefix already contains every symbolic root document. |
| Branch action | Each confirmed symbolic root gets one STO transition witness before falling back to fact-witnessed source-prior order. |
| No branch trigger | Use default fact-witnessed source-prior readout. |

This is intended to handle parallel multi-anchor questions without applying
root balancing to chain-style questions.

## Retrieval

Same gpt-4o-mini OpenIE/NV-Embed substrate, limit=100.

| Dataset | Old fresh STO R@5 | Anchor-contract R@5 | Branch-balanced R@5 | Delta vs old | Delta vs anchor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.9075 | 0.9075 | 0.9400 | +0.0325 | +0.0325 |
| MuSiQue | 0.7225 | 0.7333 | 0.7358 | +0.0133 | +0.0025 |
| HotpotQA | 0.9500 | 0.9550 | 0.9550 | +0.0050 | +0.0000 |

## Same Reader-Only QA

Same package-local V4 reader-only runner, fixed top5, qwen3-8b no-think reader,
limit=100.

| Dataset | Anchor EM | Branch EM | Delta EM | Anchor F1 | Branch F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.4700 | 0.5100 | +0.0400 | 0.5917 | 0.6068 | +0.0151 |
| MuSiQue | 0.3600 | 0.3800 | +0.0200 | 0.4309 | 0.4589 | +0.0280 |
| HotpotQA | 0.6100 | 0.6100 | +0.0000 | 0.7304 | 0.7304 | +0.0000 |

## PropRAG-Protocol Reference

These rows are from the earlier PropRAG/DAEC report and are not the same QA
runner as the package-local V4 reader-only rows.

| Dataset | Method | R@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: |
| 2Wiki | PropRAG | 0.9350 | 0.5800 | 0.6318 |
| 2Wiki | PropRAG + DAEC | 0.9525 | 0.5800 | 0.6435 |
| 2Wiki | Branch-balanced V4 | 0.9400 | 0.5100 | 0.6068 |
| MuSiQue | PropRAG | 0.6775 | 0.3800 | 0.4374 |
| MuSiQue | PropRAG + DAEC | 0.7175 | 0.3900 | 0.4660 |
| MuSiQue | Branch-balanced V4 | 0.7358 | 0.3800 | 0.4589 |
| HotpotQA | PropRAG | 0.9300 | 0.5700 | 0.6912 |
| HotpotQA | PropRAG + DAEC | 0.9500 | 0.5700 | 0.6912 |
| HotpotQA | Branch-balanced V4 | 0.9550 | 0.6100 | 0.7304 |

## Reflection

| Finding | Interpretation |
| --- | --- |
| 2Wiki missing gold was already in graph tail. | The bottleneck was parallel branch completion, not candidate generation. |
| Global root balancing hurt MuSiQue and HotpotQA. | Branch balancing must not be applied to chain-style graph shapes. |
| Source-confirmed branch trigger improved all retrieval rows relative to old fresh STO. | The trigger is useful because it only fires when dense/textual entry already found every symbolic branch root. |
| 2Wiki QA still trails PropRAG-protocol EM/F1 despite higher R@5 than PropRAG. | The remaining gap is reader/context answering behavior, not pure retrieval recall. |

