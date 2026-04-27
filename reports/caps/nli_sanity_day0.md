# CAPS Day-0 NLI Sanity

- Decision: `PROCEED_DAY1`
- Status: `completed`
- Model: `cross-encoder/nli-deberta-v3-base`
- Lexical smoke: `False`
- Pairs: `200`
- Positive/negative: `100` / `100`
- AUC: `0.903`
- 95% CI: `[0.8616, 0.9401]`
- Paired win rate: `0.88`
- Mean positive score: `0.590845`
- Mean negative score: `0.001769`
- Throughput: `108.5603` pairs/s

## Gate

- `AUC >= 0.80`: proceed to CAPS Day 1.
- `0.70 <= AUC < 0.80`: verifier is marginal; replace/calibrate verifier before Day 1.
- `AUC < 0.70` or model unavailable: stop CAPS with this verifier.

## Example Pairs

- qid: `83bf3b5a0bd911eba7f7acde48001122`, label: `1`, score: `0.969212`
- doc: `Lothair II`
- hypothesis: The mother of Lothair II is Ermengarde of Tours.

- qid: `83bf3b5a0bd911eba7f7acde48001122`, label: `0`, score: `0.000519`
- doc: `Louis the Pious`
- hypothesis: The mother of Lothair II is Ermengarde of Tours.

- qid: `83bf3b5a0bd911eba7f7acde48001122`, label: `1`, score: `0.001982`
- doc: `Ermengarde of Tours`
- hypothesis: The date of death of Ermengarde of Tours is 20 March 851.
