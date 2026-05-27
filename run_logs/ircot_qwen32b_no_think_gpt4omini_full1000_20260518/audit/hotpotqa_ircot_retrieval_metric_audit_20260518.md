# HotpotQA IRCoT Retrieval Metric Audit

## Finding

The stored HotpotQA IRCoT retrieval R@5 of `0.1065` was a metric artifact, not an IRCoT retrieval collapse.

The existing reader input contains the correct supporting titles in the top-5 for most examples. Recomputing recall by document title gives:

| Metric | Stored exact-doc metric | Recomputed title metric |
| --- | ---: | ---: |
| R@1 | 0.0360 | 0.3955 |
| R@2 | 0.0765 | 0.7255 |
| R@5 | 0.1065 | 0.9460 |
| R@10 | 0.1065 | 0.9460 |
| R@20 | 0.1065 | 0.9460 |
| All@5 | - | 0.9010 |

## Cause

`scripts/bsgs_run_ircot_baseline.py` built HotpotQA gold docs from `sample["context"]`, while retrieved docs come from `hotpotqa_corpus.json`. `RetrievalRecall` uses exact full-document string matching. For HotpotQA, the support titles match, but the context text and corpus text often differ by whitespace/punctuation normalization.

On the first 1000 HotpotQA examples, `1767 / 2000` supporting documents have the same title but non-identical context-vs-corpus text, so exact full-document matching undercounts recall.

2Wiki is not affected in this run: stored R@5 and title R@5 are both `0.9207`.

## Fix Applied

1. `scripts/bsgs_run_ircot_baseline.py` now constructs gold docs from the corpus by title when possible and records title-based retrieval metrics.
2. `run_docs_reader_qa.py` now recomputes title recall for `ircot_reader_input_v1` payloads before reporting reader metrics.
3. The existing HotpotQA reader input was backed up and patched in place:
   - Backup: `hotpotqa_ircot_qwen32b_no_think_reader_input_limit1000.json.pre_title_metric_fix_20260518.bak`
   - Patched file: `hotpotqa_ircot_qwen32b_no_think_reader_input_limit1000.json`

Validation with reader `--skip-qa` now reports HotpotQA IRCoT R@5 as `0.9460`.
