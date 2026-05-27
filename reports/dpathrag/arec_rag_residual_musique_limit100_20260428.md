# AREC-RAG Day 2 Residual Retrieval Smoke

- Status: `completed`
- Dataset: `musique`
- Rows: `100`
- Verifier: `cross-encoder/nli-deberta-v3-base`
- IRCoT prompt/query source: `local_ircot_style_frozen_prompt`

| Metric | Value |
|---|---:|
| arec_missing_hit_rate | 0.0160 |
| raw_question_missing_hit_rate | 0.0080 |
| cot_missing_hit_rate | 0.0220 |
| final_support_complete | 0.2400 |

## By Hop

| Hop | Rows | AREC Hit | Raw Hit | CoT Hit | Final Complete |
|---|---:|---:|---:|---:|---:|
| 2 | 48 | 0.0063 | 0.0042 | 0.0104 | 0.4167 |
| 3 | 30 | 0.0233 | 0.0133 | 0.0333 | 0.1333 |
| 4 | 22 | 0.0273 | 0.0091 | 0.0318 | 0.0000 |
