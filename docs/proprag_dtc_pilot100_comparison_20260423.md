# PropRAG-100 vs DtC-100 Comparison

Date: 2026-04-23

## Comparison Labels

- `HippoRAG baseline -> DtC`: same-pool composition comparison. DtC selects from the same HippoRAG top-100 used to evaluate the baseline.
- `PropRAG -> DtC`: cross-system comparison. Same broad runtime budget, but different retriever/pool.

## Same-Pool DtC Result

| Dataset | HippoRAG Baseline EM | HippoRAG Baseline F1 | DtC EM | DtC F1 | Delta EM | Delta F1 | DtC R@5 | DtC R@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.4000 | 0.4531 | 0.4800 | 0.5477 | 0.0800 | 0.0946 | 0.8850 | 0.9150 |
| HotpotQA | 0.5700 | 0.6744 | 0.5900 | 0.7054 | 0.0200 | 0.0310 | 0.9550 | 0.9950 |
| MuSiQue | 0.3000 | 0.3852 | 0.3700 | 0.4484 | 0.0700 | 0.0632 | 0.7283 | 0.8667 |

## Cross-System PropRAG Comparison

| Dataset | DtC EM | DtC F1 | PropRAG EM | PropRAG F1 | F1: DtC-PropRAG | DtC R@5 | PropRAG R@5 | Fixed-Pool Comparable? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2Wiki | 0.4800 | 0.5477 | 0.6000 | 0.6468 | -0.0991 | 0.8850 | 0.9350 | No: different retrieved pools |
| HotpotQA | 0.5900 | 0.7054 | 0.5700 | 0.6912 | 0.0142 | 0.9550 | 0.9300 | No: different retrieved pools |
| MuSiQue | 0.3700 | 0.4484 | 0.4100 | 0.4630 | -0.0146 | 0.7283 | 0.6775 | No: different retrieved pools |

## Interpretation

- DtC repairable-filter pilot100 is three-dataset positive against its own HippoRAG baseline.
- PropRAG is much stronger on 2Wiki at pilot100, roughly tied/slightly lower on HotpotQA, and slightly stronger on MuSiQue F1.
- Because PropRAG does not use the same fixed top-100 pool, this table cannot be used to claim that PropRAG is a stronger/weaker evidence composer under the fixed-pool protocol.
