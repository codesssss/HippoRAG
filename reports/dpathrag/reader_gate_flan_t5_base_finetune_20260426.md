# D-PathRAG Reader Gate: Flan-T5-Base Fine-Tune

Date: 2026-04-26

## Setup

- Model: `google/flan-t5-base`
- Fine-tune data: `data/dpathrag/reader_baselines/2wiki_train_gold_k5_5k.jsonl`
- Train records: 5,000
- Epochs: 2
- Learning rate: 5e-5
- Effective batch size: 8
- Max input tokens: 1024
- Output model: `data/dpathrag/models/flan_t5_base_gold_k5_5k`

## Reader Results

| Eval set | Evidence source | Rows | EM | F1 | Support Recall | Support Complete |
|---|---|---:|---:|---:|---:|---:|
| validation | gold@5 before fine-tune | 1000 | 0.3810 | 0.4640 | 1.0000 | 1.0000 |
| validation | gold@5 after fine-tune | 1000 | 0.6150 | 0.6896 | 1.0000 | 1.0000 |
| local1000 | gold@5 after fine-tune | 1000 | 0.6260 | 0.6881 | 1.0000 | 1.0000 |
| local1000 | dense@5 after fine-tune | 1000 | 0.3670 | 0.4111 | 0.7238 | 0.4290 |
| local1000 | PropRAG@5 after fine-tune | 1000 | 0.4250 | 0.4824 | 0.9028 | 0.7720 |

## Gate

- Absolute reader gate: `local1000 gold@5 F1 = 0.6881 >= 0.65` -> pass.
- Evidence sensitivity gate: `local1000 gold@5 F1 - PropRAG@5 F1 = 0.2057 >= 0.10` -> pass.
- Evidence sensitivity gate against dense: `local1000 gold@5 F1 - dense@5 F1 = 0.2770` -> pass.

## Decision

Reader gate passes. The fine-tuned Flan-T5-base reader is strong enough to distinguish gold evidence from retrieved evidence, so D-PathRAG can proceed to selector warm-start experiments.

Next step: train/evaluate selector warm-start on `local1000` and full 2Wiki dev subsets with support-complete@k, bridge recall, duplicate-title rate, and path/order diagnostics.
