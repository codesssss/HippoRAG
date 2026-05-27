# Evidence Transition GraphRAG V4 Anchor-Contract Summary

## Retrieval

Same gpt-4o-mini OpenIE/NV-Embed substrate, limit=100.

| Dataset | Old fresh STO R@5 | Anchor-contract R@5 | Delta R@5 | Old all-gold@5 | Anchor all-gold@5 | Delta all-gold@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.9075 | 0.9075 | +0.0000 | 0.7200 | 0.7200 | +0.0000 |
| MuSiQue | 0.7225 | 0.7333 | +0.0108 | 0.4100 | 0.4300 | +0.0200 |
| HotpotQA | 0.9500 | 0.9550 | +0.0050 | 0.9000 | 0.9100 | +0.0100 |

## Same Reader-Only QA

Same package-local V4 reader-only runner, fixed top5, qwen3-8b no-think reader,
limit=100.

| Dataset | Old fresh STO EM | Anchor-contract EM | Delta EM | Old fresh STO F1 | Anchor-contract F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.4600 | 0.4700 | +0.0100 | 0.5817 | 0.5917 | +0.0100 |
| MuSiQue | 0.3500 | 0.3600 | +0.0100 | 0.4213 | 0.4309 | +0.0096 |
| HotpotQA | 0.6200 | 0.6100 | -0.0100 | 0.7329 | 0.7304 | -0.0025 |

HotpotQA has one exact-match regression where the reader answered
`Queen Margrethe II of Denmark` instead of `Queen Margrethe II`; this is not a
retrieval miss.

## PropRAG-Protocol Reference

These rows come from the earlier PropRAG/DAEC report and are not the same QA
runner as the package-local V4 reader-only rows above.

| Dataset | Method | R@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: |
| 2Wiki | PropRAG | 0.9350 | 0.5800 | 0.6318 |
| 2Wiki | PropRAG + DAEC | 0.9525 | 0.5800 | 0.6435 |
| 2Wiki | Anchor-contract V4 | 0.9075 | 0.4700 | 0.5917 |
| MuSiQue | PropRAG | 0.6775 | 0.3800 | 0.4374 |
| MuSiQue | PropRAG + DAEC | 0.7175 | 0.3900 | 0.4660 |
| MuSiQue | Anchor-contract V4 | 0.7333 | 0.3600 | 0.4309 |
| HotpotQA | PropRAG | 0.9300 | 0.5700 | 0.6912 |
| HotpotQA | PropRAG + DAEC | 0.9500 | 0.5700 | 0.6912 |
| HotpotQA | Anchor-contract V4 | 0.9550 | 0.6100 | 0.7304 |

## Reflection

| Attempt | Outcome | Decision |
| --- | --- | --- |
| Certificate promotion | Certificate precision was too low; not used in mainline. | Reject as likely engineering trick. |
| Root-balanced readout | MuSiQue R@5 dropped to 0.6950 despite more certified docs. | Reject. |
| Transition-closure readout | MuSiQue R@5 dropped to 0.5417. | Reject. |
| Title-only query anchors | MuSiQue R@5 dropped to 0.7042. | Reject as over-restrictive. |
| Anchor-contract grounding | Improves MuSiQue/Hotpot retrieval and does not hurt 2Wiki retrieval. | Keep as current clean V4 improvement. |

