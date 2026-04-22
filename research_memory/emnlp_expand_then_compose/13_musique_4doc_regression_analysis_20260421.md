# MuSiQue 4-doc Regression Analysis - 2026-04-21

## Context

File analyzed:

- `outputs_step0_general_musique/eval_reports/dtc_embed_pilot100_anchor2_fresh8043.json`

Relevant aggregate:

- MuSiQue pilot100 overall: baseline F1 `0.3361`, DtC v1 F1 `0.3950`, delta `+0.0589`
- MuSiQue 4-doc bucket: `22` queries, baseline F1 `0.1826`, DtC v1 F1 `0.1318`, delta `-0.0508`

## Raw 4-doc Breakdown

The 4-doc regression is not a broad failure across all 4-hop queries:

- Losses: `3 / 22`
- Wins: `1 / 22`
- Ties: `18 / 22`
- Sum F1 delta: `-1.1176`
- Average F1 delta: `-0.0508`

The negative average is dominated by two catastrophic `-1.0` cases plus one small `-0.1176` case.

## Loss Cases

### Case 24

Question:

> Who is the child of the Italian navigator who explored the eastern coast of the continent César Gaytan was born in for the English?

Baseline:

- Answer: `Sebastian Cabot.`
- F1: `1.0`
- Top titles: `San Diego`, `Vicente Yáñez Pinzón`, `Exploration of North America`, `César Gaytan`, `Sebastian Cabot (explorer)`

DtC:

- Answer: `Not mentioned.`
- F1: `0.0`
- Top titles: `San Diego`, `Vicente Yáñez Pinzón`, `César Gaytan`, `Guam`, `Exploration of North America`

Trace:

- DtC correctly selected `César Gaytan` for `s1`.
- It then selected `Guam` with positive soft coverage gain but `new_requirement_ids=[]`.
- This pushed out `Sebastian Cabot (explorer)`, which was the answer-bearing baseline top-5 document.

Diagnosis:

- The dependent requirements were not truly covered.
- The harmful action was a soft-gain insertion that did not cross a new requirement threshold.

### Case 99

Question:

> When was the region immediately north of the region where the country in which Aluf can be found is located and the Persian Gulf established?

Baseline:

- Answer: `1932.`
- F1: `1.0`
- Top titles include `Geography of Saudi Arabia`.

DtC:

- Answer: `1926.`
- F1: `0.0`
- Top titles: `Partition of the Ottoman Empire`, `Battle of Qurah and Umm al Maradim`, `Alūksne`, `Near East`, `Arabian Peninsula`

Trace:

- Reserved `Battle of Qurah and Umm al Maradim` falsely covered all three requirements.
- DtC then selected `Alūksne` and `Near East` with `new_requirement_ids=[]`.
- It dropped the answer-bearing geography document.

Diagnosis:

- There is both false semantic coverage and soft-gain insertion.
- `Aluf` vs `Alūksne` suggests lexical/semantic confounding.

### Case 11

Question:

> How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country?

Baseline:

- F1: `0.1176`

DtC:

- F1: `0.0`

Trace:

- DtC selected `A Lim` as a hard new requirement.
- It then selected `Mali` with positive soft gain but no new requirement crossing.

Diagnosis:

- Lower-severity version of the same issue: soft-gain selection can displace reader-friendly baseline context.

## Win Case

Case 69:

- Baseline answer: `Will Power.`
- DtC answer: `Mario Andretti.`
- F1 delta: `+1.0`

Trace:

- DtC selected `Arizona` for `s3`.
- DtC selected `Desert Diamond West Valley Phoenix Grand Prix` for `s4`.
- Both selections had non-empty `new_requirement_ids`.

Diagnosis:

- The successful 4-doc repair uses hard requirement crossing, not soft-gain-only insertion.

## Aggregate Trace Pattern

Across MuSiQue pilot100:

- All losses: `6`, average soft-gain selections `1.17`, average hard-crossing selections `0.50`
- All wins: `13`, average soft-gain selections `0.77`, average hard-crossing selections `0.54`

For the 4-doc bucket:

- Losses: average soft-gain selections `1.33`, hard-crossing selections `0.67`
- Win: soft-gain selections `0.00`, hard-crossing selections `2.00`

Interpretation:

- Soft-gain is not always harmful, but the catastrophic 4-doc losses are dominated by soft-gain insertions that do not cover a new requirement.
- The method's positive 4-doc case comes from actual new requirement crossing.

## Current Conclusion

The MuSiQue 4-doc regression is not strong evidence that DtC cannot handle 4-hop queries.

It is more specifically evidence that the current greedy objective is too willing to replace baseline context using positive but sub-threshold coverage improvements.

This creates a failure mode:

> soft requirement-score improvement without new requirement coverage can evict answer-bearing baseline evidence.

## Recommended Next Ablation

Add a `hard-crossing-only` / `new-requirement-only` DtC variant:

- Select a DtC candidate only if it crosses at least one previously uncovered requirement threshold.
- If no candidate crosses a new requirement, fall back to baseline fill.
- Keep soft coverage score only as a tie-break among candidates that cross new requirements.

Expected effects:

- Should recover Case 24 and Case 99 by preserving baseline fill instead of inserting `Guam` / `Alūksne`.
- May reduce some wins where soft gain is genuinely useful, so it must be tested empirically.

Do not edit running queues. Wait for current pilot100 ablations to finish before adding this variant, otherwise queued runs will silently switch code versions.
