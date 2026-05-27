# V13B Qwen Evidence-Frame Contract

Date: 2026-05-03

Status: superseded as the research mainline. This document is retained as a
diagnostic note for the schema-guided OpenIE probe. The active plan is
`docs/v13b_minimal_evidence_interface_plan_20260503.md`, which rejects a fixed
relation schema as the proposed method and moves to a schema-light minimal
evidence interface.

This note records an earlier diagnostic direction for migrated V13B. It is not
a selector patch and not a fallback recipe. Its diagnostic goal was to test
whether the query-side demand graph and the corpus-side Qwen OpenIE graph fail
because they do not speak the same evidence-frame language.

## Current Diagnosis

The migrated V13B full1000 setup first failed because it used a PropRAG dense
pool wrapped as a V13B source report, not a true structural/SFB-style substrate.
The structural source-report builder now performs low-degree endpoint closure
with `max_endpoint_doc_degree=30`, which removes hub explosion while preserving
2Wiki gains.

After that fix, the remaining limit100 evidence-frame audit shows that candidate
coverage is not the main bottleneck. Gold documents are usually present, but
Qwen OpenIE often does not materialize the retrieval-critical demand as a
role-aligned fact.

| Dataset | Retrieval-critical obligations | Gold docs covered | Candidate missing | Exact gold OpenIE frame | Endpoint/relation missing |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 220 | 218 | 2 | 94 | 112 |
| HotpotQA | 216 | 216 | 0 | 37 | 152 |
| MuSiQue | 253 | 237 | 16 | 24 | 214 |

The exact gold OpenIE frame rate is:

| Dataset | Exact frame rate | Endpoint/relation missing rate |
|---|---:|---:|
| 2Wiki | 42.73% | 50.91% |
| HotpotQA | 17.13% | 70.37% |
| MuSiQue | 9.49% | 84.59% |

Conclusion: the main bottleneck is the Qwen-native evidence graph schema, not
PPR parameters, selector aggressiveness, or candidate pool size.

## Principle

Do not repair this with:

| Forbidden repair | Why it is wrong |
|---|---|
| Relation synonym lists in the matcher | It hides OpenIE/schema failure behind brittle string exceptions. |
| Lexical fallback | It returns to dense/lexical rescue instead of improving the graph. |
| Partial obligation grounding | It makes the program look feasible while leaving evidence roles uncovered. |
| Alpha/path-length tuning | The failure happens before graph walk quality matters. |
| Copying dense candidates into structural fields | It pollutes the protocol and makes reports misleading. |

The clean repair is:

```text
Define evidence-frame contract
-> make query compiler and Qwen OpenIE emit/check the same frame schema
-> rebuild structural source report on that graph
-> only then run V13B selector
```

## Evidence-Frame Contract

Each corpus fact unit must expose the same role-aware fields used by query
obligations:

| Field | Meaning |
|---|---|
| `subject_surface` | Original subject phrase from corpus. |
| `subject_key` | Canonical endpoint key after deterministic normalization. |
| `relation_surface` | Original relation phrase from corpus. |
| `relation_key` | Schema-level evidence frame key. |
| `object_surface` | Original object phrase from corpus. |
| `object_key` | Canonical endpoint key after deterministic normalization. |
| `source_sentence` | Sentence or span that supports the fact. |
| `doc_title` | Document title used for title/entity grounding. |

The query obligation must use the same fields:

| Field | Meaning |
|---|---|
| `subject_surface/key` | Bound query entity or variable binding. |
| `relation_surface/key` | Required evidence frame. |
| `object_surface/key` | Bound query entity, literal, or variable binding. |
| `subject_is_variable` | Whether the subject role must be bound upstream. |
| `object_is_variable` | Whether the object role must be bound downstream. |

## Minimum Required Frame Families

These are schema contracts, not ranking weights.

| Family | Required frame behavior | Typical current failure |
|---|---|---|
| Temporal attributes | Birth, death, release, establishment, completion, opening, and year facts must be materialized as typed temporal relations when the object is a date/year. | `X is a 1961 Hindi movie` stays as `is a`, so `X --released_on--> 1961` is absent. |
| Place/origin attributes | Birthplace, death place, located in, headquartered in, country of origin, language/place facts must preserve location role direction. | `country_origin` lands as `set_in`, `voice_cast_includ`, or unrelated lead facts. |
| Work metadata roles | Director, writer, composer, performer, actor/character, production company must preserve work-to-person or work-to-organization role direction. | Song/film title exists, but relation becomes generic cast/include/star relation. |
| Person relation roles | Mother, father, spouse, child, predecessor/successor must be expressed as direct kinship/role frames. | Gold docs mention family context, but OpenIE does not expose the required endpoint relation. |
| Office/position roles | Appointed as, served as, president, secretary, chief justice, mayor, and residence-for roles must preserve office holder and office title. | Query relation becomes a long title relation, while OpenIE emits generic `served as`/`appointed`. |
| Event/competition roles | Won, draft, season, tournament, competition, participated in must preserve event participant and event object. | MuSiQue event chains collapse into broad `include`, `attend`, or location facts. |

## Endpoint Contract

Endpoint grounding must also be schema-level, not fuzzy lexical rescue.

| Endpoint shape | Contract |
|---|---|
| Named endpoint | Normalize deterministically and preserve document-title aliases. |
| Typed title endpoint | Strip query-side type prefixes only when they are title qualifiers, e.g. `film Billy Elliot` -> `Billy Elliot`. |
| Title-qualified endpoint | Strip parenthetical title qualifiers conservatively, e.g. `Algiers (Film)` -> `Algiers`. |
| Descriptive endpoint | Query compiler should avoid using descriptions as hard endpoints when they refer to a variable, e.g. `recently abdicated queen`. |
| All-variable obligation | Must not be counted as grounded until upstream variables have concrete bindings. |

## Next Implementation Target

The next code change should be an OpenIE schema-compliance gate, not a selector
change.

Inputs:

| Input | Purpose |
|---|---|
| Qwen OpenIE JSON | Corpus-side facts to audit. |
| Source report | Gold docs and structural candidate docs. |
| Query obligation cache | Query-side demands. |
| V13B selector JSON | Retrieval-critical obligation IDs and variable bindings. |

Outputs:

| Output | Purpose |
|---|---|
| Frame coverage by family | Shows which schema family blocks grounding. |
| Endpoint shape breakdown | Separates descriptive endpoint/query compiler failures from corpus OpenIE failures. |
| Representative failures | Gives prompt/schema repair cases without changing retrieval. |
| Pass/fail gate | A Qwen OpenIE build should not be considered a valid V13B substrate if exact frame rate is too low on gold docs. |

Initial gate proposal for diagnostic builds:

| Dataset | Minimum exact gold OpenIE frame rate before selector experiments |
|---|---:|
| 2Wiki | 60% |
| HotpotQA | 40% |
| MuSiQue | 30% |

These are not final benchmark targets. They are substrate sanity thresholds. If
the gold documents cannot produce role-aligned evidence frames, retrieval
results are measuring OpenIE failure rather than V13B graph retrieval quality.

## Current Artifacts

Evidence-frame contract audit:

```text
analyze_v13b_evidence_frame_contract.py
```

Schema gap summary:

```text
summarize_v13b_gold_openie_schema_gaps.py
```

Limit100 deg30 reports:

```text
run_logs/v13b_structural_pool100_limit100_deg30_20260502/failure_reports/
```

Validation:

```text
env PYTHONDONTWRITEBYTECODE=1 /mnt/nvme/code/HippoRAG/.venv-hipporag/bin/python -m pytest tests/v13b -q
# 135 passed
```
