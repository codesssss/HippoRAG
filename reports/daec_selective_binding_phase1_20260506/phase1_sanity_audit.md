# DAEC-selective Phase-1 Sanity Audit

Purpose: explain why 2Wiki/HotpotQA have exact zero paired delta between DAEC and DAEC-selective despite nonzero abstention rates.

## Gate and Equality Counts

| Dataset | Bind | Abstain | Null Rate | Selective titles = DAEC | Selective answers = DAEC | Abstain DAEC titles = Nobind | Abstain DAEC answers = Nobind |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 698 | 302 | 0.302 | 1000/1000 | 1000/1000 | 302/302 | 302/302 |
| HotpotQA | 657 | 343 | 0.343 | 1000/1000 | 1000/1000 | 343/343 | 343/343 |
| MuSiQue | 371 | 629 | 0.629 | 738/1000 | 853/1000 | 367/629 | 482/629 |

## Interpretation

- The gate does trigger on 2Wiki and HotpotQA: null rates are 0.302 and 0.343.
- The exact zero DAEC-selective vs DAEC delta is not because abstention is disabled.
- On 2Wiki and HotpotQA, every abstained query has identical DAEC and Nobind top-5 titles, so DAEC-selective changes the binding mode but not the reader input.
- MuSiQue is different: 262 abstentions change the selected evidence set, which is where the F1 repair comes from.
