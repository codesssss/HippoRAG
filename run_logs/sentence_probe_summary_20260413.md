# Sentence Attribution Probe Summary (2026-04-13)

## Scope

- Goal:
  - Determine whether MuSiQue positive `ce_local_repair` cases are driven by a small number of useful sentences.
  - Determine whether 2Wiki nonpositive `ce_local_repair` cases contain any localized useful sentence or are mostly semantic drift at the sentence level.
- Script:
  - `scripts/analyze_sentence_attribution_probe.py`
- Runs:
  - `run_logs/musique_sentence_probe_positive_20260413.json`
  - `run_logs/2wiki_sentence_probe_nonpositive_20260413.json`

## MuSiQue Positive Cases

- Cases analyzed: `4`
- Candidate top2 softmax mass mean / median: `0.7954 / 0.7989`
- Candidate top2 token share mean: `0.3004`
- Candidate localized rate (`top2 mass >= 0.6`): `100%`
- Candidate any goldish sentence rate: `75%`
- Candidate top2 goldish sentence rate: `75%`

### Interpretation

- Positive MuSiQue cases are strongly localized at the sentence level.
- In all 4 cases, the candidate's top-2 sentences carry most of the sentence-level CE mass.
- This supports the claim that useful signal is often concentrated in `1-2` sentences rather than spread across the whole appended document.
- However, the positive cases still look like a mix of:
  - redundancy repair
  - broader context correction
  rather than clean requirement hits.

## 2Wiki Nonpositive Cases

- Cases analyzed: `4`
- Candidate top2 softmax mass mean / median: `0.8704 / 0.9241`
- Candidate top2 token share mean: `0.2273`
- Candidate localized rate (`top2 mass >= 0.6`): `100%`
- Candidate any goldish sentence rate: `100%`
- Candidate top2 goldish sentence rate: `75%`
- Candidate goldish only outside top2 rate: `25%`

### Interpretation

- The 2Wiki failures are **not** mostly “whole-document drift”.
- In most nonpositive cases, the candidate also has a highly localized top-1/top-2 sentence signal.
- That means the failure pattern is more consistent with:
  - localized but wrong / semantically drifting evidence
  - or partially relevant evidence that still does not answer the real missing information need
- Only a minority of cases look like “there may be a useful sentence, but it is buried outside the top-2”.

## Overall Takeaway

- The probe gives **partial support** to the snippet hypothesis:
  - useful information is often highly localized in `1-2` sentences.
- But it does **not** overturn the current main conclusion:
  - the primary bottleneck still looks like `proposal semantics`, not just evidence granularity.
- A future snippet-level direction is now more defensible as a research idea, but not yet justified as the next immediate optimization step for the frozen paper track.
