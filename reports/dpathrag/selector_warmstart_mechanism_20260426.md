# D-PathRAG Selector Warm-Start Mechanism Smoke

Date: 2026-04-26

## Setup

- Data: local 2Wiki 1000-example protocol
- Candidate pools: Dense pool100 and PropRAG pool100
- Split: first 800 train, final 200 held-out eval
- Selector: autoregressive no-replacement path selector
- Path length: 5
- Warm-start target: gold supporting documents in candidate pool rank order
- Input features: rank/score/text lexical features only; gold labels are not selector inputs

This is a mechanism smoke, not a final generalization experiment. Full train/dev pool exports are still needed for final claims.

## Results

### Evidence Metrics

| Pool | Method | Support Recall@5 | Support Complete@5 | Duplicate Title Rate |
|---|---|---:|---:|---:|
| Dense pool100 | rank top-5 | 0.7388 | 0.4550 | 0.0050 |
| Dense pool100 | AR selector warm-start | 0.7612 | 0.4950 | 0.0050 |
| PropRAG pool100 | rank top-5 | 0.8962 | 0.7500 | 0.0050 |
| PropRAG pool100 | AR selector warm-start | 0.8838 | 0.7200 | 0.0050 |

### Fine-Tuned Reader Metrics

Reader: `data/dpathrag/models/flan_t5_base_gold_k5_5k`

| Pool | Method | Answer EM | Answer F1 |
|---|---|---:|---:|
| Dense pool100 | rank top-5 | 0.3700 | 0.4154 |
| Dense pool100 | AR selector warm-start | 0.3800 | 0.4285 |
| PropRAG pool100 | rank top-5 | 0.4850 | 0.5426 |
| PropRAG pool100 | AR selector warm-start | 0.4600 | 0.5274 |

## Interpretation

- Dense pool: warm-start selector improves support-complete by +4.0 pp and support recall by +2.24 pp over rank top-5. This suggests path selector training can recover some buried support from a weaker widened pool.
- PropRAG pool: warm-start selector underperforms rank top-5 by -3.0 pp support-complete and -1.24 pp support recall. Current lightweight feature set is not strong enough to beat a strong PropRAG ordering.
- Answer F1 follows the evidence metrics: Dense improves by +1.31 pp F1, while PropRAG drops by -1.52 pp F1.
- Duplicate-title rate stays flat at 0.5%, so this smoke does not show the duplicate-title collapse observed in BSGS.

## Decision

Do not treat this selector as final D-PathRAG yet. The current lightweight-feature warm-start is useful as a pipeline and mechanism smoke, not as the main method.

Next selector version should replace lightweight lexical features with q-doc encoder or embedding features before any E2E claim. A practical next step is to cache dense q-doc representations for pool100 and train the same AR selector over those representations plus DAEC/rank features.
