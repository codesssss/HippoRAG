# Action-Level Audit: musique_subset20

- selected actions: `20`
- oracle-positive selected swaps: `10`
- oracle-negative sampled swaps: `10`
- dryrun executed swaps in audit: `16`
- judge executed swaps in audit: `2`
- proposal pass rate: `0.0`

## Pass Rates

| Group | Count | Passed | Pass rate |
|---|---:|---:|---:|
| oracle_positive | 10 | 0 | 0.0 |
| oracle_negative | 10 | 0 | 0.0 |
| dryrun_selected | 16 | 0 | 0.0 |
| judge_selected | 2 | 0 | 0.0 |

## Reject Reasons

- `gain_verifier_reject`: `11`
- `no_unsupported_claim`: `9`

## Sample Oracle-Positive Rejections

- question: `What year did the publisher of Labyrinth end?`
  - action: swap in `The Lord of the Rings` for `Spectrum HoloByte`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - claim: `None` `None`
  - gain verdict: `None`
  - unique-support veto: ``
  - skip reason: `no_unsupported_claim`
- question: `Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?`
  - action: swap in `Modern history` for `Korean War`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - claim: `v1` `Evidence about directive 10 2. Question: Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?`
  - gain verdict: `not_supported`
  - unique-support veto: ``
  - skip reason: `gain_verifier_reject`
- question: `How did did the people fare during the reign of the abolisher of sati partha in India?`
  - action: swap in `British Empire` for `Indian Rebellion of 1857`
  - oracle ΔEM / ΔF1: `0.0` / `0.1818`
  - claim: `v2` `Answer the question: How did did the people fare during the reign of the abolisher of sati partha in India?`
  - gain verdict: `not_supported`
  - unique-support veto: ``
  - skip reason: `gain_verifier_reject`
- question: `What city is the star of Sous les pieds des femmes from?`
  - action: swap in `Claudia Cardinale` for `C'est les vacances`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - claim: `None` `None`
  - gain verdict: `None`
  - unique-support veto: ``
  - skip reason: `no_unsupported_claim`
- question: `When did the 1979-80 European Cup winner win the FA Cup?`
  - action: swap in `List of Chelsea F.C. managers` for `History of Chelsea F.C.`
  - oracle ΔEM / ΔF1: `0.0` / `0.009`
  - claim: `None` `None`
  - gain verdict: `None`
  - unique-support veto: ``
  - skip reason: `no_unsupported_claim`
- question: `What was the 2018 population of the Italian city that's underwater?`
  - action: swap in `List of South American countries by population` for `Valencia`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - claim: `None` `None`
  - gain verdict: `None`
  - unique-support veto: ``
  - skip reason: `no_unsupported_claim`
- question: `What was the language from which the last name Sylvester originated during the era of the person crowned emperor of the west in 800 CE later known as?`
  - action: swap in `Modern history` for `Holy Roman Empire`
  - oracle ΔEM / ΔF1: `0.0` / `0.6667`
  - claim: `None` `None`
  - gain verdict: `None`
  - unique-support veto: ``
  - skip reason: `no_unsupported_claim`
- question: `What is the direction of flow of the body of water by the city where Write This Down was formed?`
  - action: swap in `Ohio River` for `Boomi River`
  - oracle ΔEM / ΔF1: `0.0` / `0.0571`
  - claim: `None` `None`
  - gain verdict: `None`
  - unique-support veto: ``
  - skip reason: `no_unsupported_claim`
- question: `What administrative territorial entity contains the place where KPRM is licensed to broadcast to?`
  - action: swap in `University of Kansas` for `KAPE`
  - oracle ΔEM / ΔF1: `0.0` / `0.5`
  - claim: `None` `None`
  - gain verdict: `None`
  - unique-support veto: ``
  - skip reason: `no_unsupported_claim`
- question: `What is an example of a railroad line in the country first to invade Manchuria?`
  - action: swap in `Allies of World War II` for `Pacific War`
  - oracle ΔEM / ΔF1: `0.0` / `0.0435`
  - claim: `v1` `Evidence about manchuria. Question: What is an example of a railroad line in the country first to invade Manchuria?`
  - gain verdict: `not_supported`
  - unique-support veto: ``
  - skip reason: `gain_verifier_reject`
