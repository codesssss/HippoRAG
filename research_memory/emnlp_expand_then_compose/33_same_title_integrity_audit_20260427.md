# Same-title Integrity Audit

Date: 2026-04-27

Canonical outputs:
- `scripts/audit_same_title_confound.py`
- `reports/paper/same_title_audit.md`
- `reports/paper/same_title_audit.json`

## Purpose

NREV audit exposed a real matched-replacement bug where destructive perturbation could replace a document with another document of the same title. Before freezing the paper story, we audited whether the same-title confound invalidates existing method conclusions for D-PathRAG, CEE, CPAG, and DAEC.

This audit is static-trace only. It does not rerun selectors or readers.

## Result

Same-title duplication does not explain the main method conclusions.

Key numbers:
- D-PathRAG selector_v1 PropRAG kfold1000:
  - added docs: `1281`
  - added gold/non-gold: `76 / 1205`
  - same-title added non-gold vs rank top-5: `0`
  - interpretation: hard-negative import is real under the audited title-level trace, not a same-title duplicate artifact.
- CEE learned edit policies:
  - learned_edit1 added gold/non-gold: `35 / 239`, same-title non-gold: `0`
  - learned_edit2 operation-level added gold/non-gold: `45 / 403`, same-title non-gold: `92`
  - learned_edit2 final-set added non-gold: `219`, final-set same-title non-gold: `0`
  - interpretation: learned_edit2 has multi-step oscillation/reinsertion, not final selected-set same-title contamination.
- CPAG:
  - anchored added gold/non-gold vs PropRAG rank: `5 / 436`
  - same-title added non-gold: `0`
  - anchored selected cross-pool gold/non-gold: `341 / 554`
  - interpretation: CPAG failure is shared cross-pool distractor agreement, not same-title duplicate inflation.
- DAEC:
  - PropRAG 2Wiki same-title non-gold: `0`, selected duplicate query rate: `0.003`
  - PropRAG HotpotQA same-title non-gold: `0`, selected duplicate query rate: `0.001`
  - Dense 2Wiki same-title non-gold: `0`, selected duplicate query rate: `0.003`
  - Dense HotpotQA same-title non-gold: `0`, selected duplicate query rate: `0.001`
  - PropRAG MuSiQue selected/base duplicate query rate: `0.364 / 0.388`
  - Dense MuSiQue selected/base duplicate query rate: `0.349 / 0.365`
  - interpretation: MuSiQue has high duplicate-title exposure inherited from the pool baseline; DAEC does not amplify it and slightly reduces selected duplicate rate.
- DAEC-ALR Step-1:
  - added gold/non-gold: `25 / 313`
  - same-title non-gold replacements: `0`

## Decision Impact

- Keep D-PathRAG hard-negative import as a valid negative diagnostic.
- Keep CPAG as a shared-distractor agreement failure, not a duplicate-title failure.
- Keep DAEC 2Wiki/HotpotQA mainline unchanged on same-title grounds.
- Mention MuSiQue duplicate-title exposure as a corpus/pool property if discussing MuSiQue trace mechanics, but do not rerun DAEC only for same-title hygiene.
- Keep the NREV same-title replacement fix as a destructive-null hygiene lesson; do not retroactively invalidate the prior main tables.

## Paper Framing

The paper should separate two phenomena:
- Same-title replacement is a serious confound for counterfactual perturbation methods such as NREV, because replacing a support with another same-title passage may not break evidence.
- Same-title duplication is not the cause of the observed D-PathRAG/CEE/CPAG/DAEC conclusions in the audited traces.

This strengthens paper honesty without reopening method exploration.
