# PCRS-RAG V2 Parser / Compiler Spec

Last updated: 2026-04-03

## Purpose

This document defines the executable specification for the next `PCRS-RAG V2` parser/compiler refresh.

The goal is to replace the current heuristic need-unit construction with:

- offline LLM step planning
- deterministic need-unit compilation
- explicit predicate normalization
- explicit constraint extraction
- unit-level counterfactual construction

This document exists because the current V2 implementation is still too close to lexical requirement construction.

Current failure signals from `MuSiQue-40`:

- `relation_hop` lexical predicate rate: `33 / 41 = 80.5%`
- queries with any counterfactual sets: `8 / 40 = 20.0%`
- oracle smoke: `EM delta = -0.0250`, `F1 delta = -0.0349`
- positive-vs-negative separation AUC: `0.5161`

This spec is designed to prevent the next implementation from silently drifting back to lexical heuristics.

## Scope

This spec defines only:

- LLM step-plan generation
- step JSON schema
- need-unit schema
- deterministic compiler rules
- predicate normalization
- counterfactual construction
- malformed / fallback policy
- offline diagnostics and gates
- annotation prompt for unit-doc support labeling

This spec does **not** change:

- beam / Pareto / utopia search
- reserve policy
- proposal expansion backbone
- reader / generator path

## Implementation Boundary

Target files for the next implementation round:

- `scripts/build_need_unit_cache.py`
- `scripts/requirement_beam_utils.py`
- `scripts/annotate_need_unit_support.py`
- `scripts/train_need_unit_scorer.py`

The next round should preserve the current selector boundary:

- parser/compiler changes feed `requirement_beam`
- selector search logic stays unchanged

## Versioning

Current compatibility version:

- `pcrs_rag_v2_need_units`

New target version from this spec:

- `pcrs_rag_v2_need_units_qdmr_v1`

Compatibility rule:

- old V2 cache loading remains supported
- new cache builder emits only `pcrs_rag_v2_need_units_qdmr_v1`
- diagnostics must print the version and reject mixed-version training by default

## Top-Level Pipeline

The intended V2-min pipeline is:

1. Retrieve `pool_k` candidate docs.
2. Collect:
   - `question`
   - `question_entities`
   - `seed_entities`
   - `pool_titles_head`
   - `predicted_answer_type`
3. Call an offline LLM parser to produce `qdmr_steps`.
4. Run a deterministic compiler:
   - normalize steps
   - compile `positive_need_units`
   - generate `counterfactual_sets`
   - emit `diagnostics`
5. Build doc-level annotations against the compiled units.
6. Feed the resulting cache into the existing `requirement_beam`.

The LLM is only responsible for producing a step plan.
The LLM does **not** directly produce final need units.

## LLM Parser Prompt

### System Prompt

```text
You are a query decomposition planner for multi-hop retrieval.

Your job is to decompose a question into a small sequence of semantic reasoning steps.
The steps must describe what information is needed to answer the question, not just repeat keywords.

Requirements:
1. Output 2 to 4 steps only.
2. Each step must be necessary for solving the question.
3. Use explicit intermediate variables such as X, Y, Z, ANSWER when needed.
4. Distinguish:
   - locating an entity/event,
   - retrieving a relation value,
   - checking a constraint,
   - producing the final answer.
5. Do NOT output irrelevant topical phrases.
6. Do NOT use wh-words (who/what/when/where/which/how/why/whom/whose) as entities.
7. Do NOT directly answer the question.
8. Prefer relation- and constraint-centered steps over keyword overlap.
9. Prefer steps that identify:
   - the bridge variable needed for the next hop,
   - the relation needed to retrieve it,
   - any role/time/comparison constraint required to avoid confusable distractors.
10. Avoid steps that only restate surface words from the question.

Return JSON only.
```

### User Prompt Template

```text
Question:
{question}

Question entities:
{question_entities}

Seed entities:
{seed_entities}

Candidate pool titles (for grounding only, do not copy blindly):
{pool_titles}

Predicted answer type:
{answer_type}

Return a JSON object with:
{
  "question_id": "...",
  "answer_type": "...",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "<one of: locate_entity, relation_lookup, constraint_check, answer>",
      "description": "...",
      "inputs": ["..."],
      "output_variable": "X"
    }
  ]
}

Guidelines:
- "locate_entity": identify a core entity/event if disambiguation is needed.
- "relation_lookup": retrieve a relation value from an entity or variable.
- "constraint_check": apply time, comparison, role, negation, or scope constraints.
- "answer": specify which variable or relation becomes the final answer.
- Use at most one output variable per step.
- If the question is 2-hop, usually use 3 steps.
- If no disambiguation is needed, skip locate_entity.
```

## Few-Shot Step-Plan Examples

The parser prompt should be shipped with at least these four few-shot patterns.

### Example A: Pure 2-Hop

Question:

`Who is the mascot of the university related to Randy Conrads?`

Expected step plan:

```json
{
  "question_id": "fs_a",
  "answer_type": "person_or_character",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "relation_lookup",
      "description": "Find the university associated with Randy Conrads.",
      "inputs": ["Randy Conrads"],
      "output_variable": "X"
    },
    {
      "step_id": "s2",
      "operation": "relation_lookup",
      "description": "Find the mascot of X.",
      "inputs": ["X"],
      "output_variable": "ANSWER"
    },
    {
      "step_id": "s3",
      "operation": "answer",
      "description": "Return ANSWER as the final answer.",
      "inputs": ["ANSWER"],
      "output_variable": "ANSWER"
    }
  ]
}
```

### Example B: Temporal Constraint

Question:

`Which film directed by Christopher Nolan was released before 2010?`

Expected step plan:

```json
{
  "question_id": "fs_b",
  "answer_type": "film",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "relation_lookup",
      "description": "Find films directed by Christopher Nolan.",
      "inputs": ["Christopher Nolan"],
      "output_variable": "X"
    },
    {
      "step_id": "s2",
      "operation": "constraint_check",
      "description": "Keep the film X whose release date is before 2010.",
      "inputs": ["X"],
      "output_variable": "X"
    },
    {
      "step_id": "s3",
      "operation": "answer",
      "description": "Return X as the final answer.",
      "inputs": ["X"],
      "output_variable": "ANSWER"
    }
  ]
}
```

### Example C: Comparative Constraint

Question:

`Which city has the larger population, the birthplace of Author A or the birthplace of Author B?`

Expected step plan:

```json
{
  "question_id": "fs_c",
  "answer_type": "city",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "relation_lookup",
      "description": "Find the birthplace of Author A.",
      "inputs": ["Author A"],
      "output_variable": "X"
    },
    {
      "step_id": "s2",
      "operation": "relation_lookup",
      "description": "Find the birthplace of Author B.",
      "inputs": ["Author B"],
      "output_variable": "Y"
    },
    {
      "step_id": "s3",
      "operation": "constraint_check",
      "description": "Compare the populations of X and Y and keep the city with the larger population.",
      "inputs": ["X", "Y"],
      "output_variable": "ANSWER"
    },
    {
      "step_id": "s4",
      "operation": "answer",
      "description": "Return ANSWER as the final answer.",
      "inputs": ["ANSWER"],
      "output_variable": "ANSWER"
    }
  ]
}
```

### Example D: Role Confusion / Entity Disambiguation

Question:

`Who is played by the director of The Good Shepherd in The Godfather?`

Expected step plan:

```json
{
  "question_id": "fs_d",
  "answer_type": "person_or_character",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "locate_entity",
      "description": "Identify the film The Good Shepherd.",
      "inputs": ["The Good Shepherd"],
      "output_variable": "X"
    },
    {
      "step_id": "s2",
      "operation": "relation_lookup",
      "description": "Find the director of X.",
      "inputs": ["X"],
      "output_variable": "Y"
    },
    {
      "step_id": "s3",
      "operation": "relation_lookup",
      "description": "Find the character played by Y in The Godfather.",
      "inputs": ["Y", "The Godfather"],
      "output_variable": "ANSWER"
    },
    {
      "step_id": "s4",
      "operation": "answer",
      "description": "Return ANSWER as the final answer.",
      "inputs": ["ANSWER"],
      "output_variable": "ANSWER"
    }
  ]
}
```

## Step JSON Schema

### Root Step-Plan Object

```json
{
  "question_id": "string",
  "answer_type": "string",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "relation_lookup",
      "description": "Find the mascot of X.",
      "inputs": ["X"],
      "output_variable": "ANSWER"
    }
  ]
}
```

### Root Fields

| Field | Type | Required | Rule |
|---|---|---:|---|
| `question_id` | string | yes | Stable external key or dataset-local key |
| `answer_type` | string | yes | Non-empty canonical answer-type string |
| `qdmr_steps` | array | yes | Length must be `2..4` |

### Step Fields

| Field | Type | Required | Rule |
|---|---|---:|---|
| `step_id` | string | yes | Must match `^s[1-4]$` |
| `operation` | string | yes | One of `locate_entity`, `relation_lookup`, `constraint_check`, `answer` |
| `description` | string | yes | Non-empty natural-language step |
| `inputs` | array[string] | yes | Length `1..3`; `answer` may use `["ANSWER"]` |
| `output_variable` | string | yes | One of `X`, `Y`, `Z`, `ANSWER` |

### Step Validation Rules

1. `step_id` must be contiguous and ordered:
   - allowed sequences: `s1,s2`, `s1,s2,s3`, `s1,s2,s3,s4`
2. `qdmr_steps` must follow solve order.
3. `inputs` cannot be empty.
4. `operation=answer` requires:
   - `output_variable="ANSWER"`
   - it must be the last step
5. Only these variable names are allowed:
   - `X`
   - `Y`
   - `Z`
   - `ANSWER`
6. `locate_entity` may output `X`, `Y`, or `Z`, but not `ANSWER`.
7. `constraint_check` may output the same variable it receives.
8. If the LLM omits an explicit answer step, the compiler may auto-insert one, but the raw parser response is still marked as incomplete.

### Invalid Step Examples

- `{"step_id": "1", ...}` because the format is wrong
- `{"operation": "bridge", ...}` because the operation is unsupported
- `{"inputs": [], ...}` because empty inputs are not allowed
- `{"operation": "answer", "output_variable": "X"}` because answer must output `ANSWER`
- `{"output_variable": "W"}` because only `X/Y/Z/ANSWER` are allowed

## Need-Unit Schema

### Positive Need Unit

```json
{
  "unit_id": "u_relation_1",
  "unit_type": "relation_hop",
  "target_variable": "?x",
  "subject": "Randy Conrads",
  "predicate": "associated_university",
  "raw_predicate_text": "associated with",
  "object": "?x",
  "constraints": [],
  "answer_relevance": false,
  "source_step_ids": ["s1"],
  "confidence": "high",
  "selector_enabled": true,
  "debug": {
    "parser_description": "Find the university associated with Randy Conrads."
  }
}
```

### Need-Unit Fields

| Field | Type | Required | Rule |
|---|---|---:|---|
| `unit_id` | string | yes | Stable unique id within query |
| `unit_type` | string | yes | One of `entity_locator`, `relation_hop`, `constraint_check`, `answer_slot` |
| `target_variable` | string | yes | Canonical variable form: `?x/?y/?z/?ans` |
| `subject` | string | yes | Canonical entity or variable |
| `predicate` | string | yes | Canonical predicate name |
| `raw_predicate_text` | string | no | Raw relation phrase before normalization |
| `object` | string or null | yes | Canonical entity/variable or `null` for pure constraint units |
| `constraints` | array | yes | Constraint objects, possibly empty |
| `answer_relevance` | bool | yes | True only for `answer_slot` |
| `source_step_ids` | array[string] | yes | References raw step ids |
| `confidence` | string | yes | One of `high`, `medium`, `low` |
| `selector_enabled` | bool | yes | False when the unit is kept only for diagnostics |
| `debug` | object | no | Compiler-only debug metadata |

### Constraint Object

```json
{
  "type": "temporal",
  "value": "before 2010"
}
```

Constraint fields:

| Field | Type | Required | Rule |
|---|---|---:|---|
| `type` | string | yes | One of `temporal`, `comparative`, `role`, `negation`, `scope`, `answer_type` |
| `value` | string | yes | Canonicalized constraint value |

## Counterfactual Set Schema

```json
{
  "cf_id": "cf_1",
  "transform": "role_swap",
  "source_unit_id": "u_relation_1",
  "need_units": [
    {
      "unit_id": "u_relation_1_role_swap",
      "unit_type": "relation_hop",
      "target_variable": "?x",
      "subject": "Randy Conrads",
      "predicate": "founded_by",
      "raw_predicate_text": "founded by",
      "object": "?x",
      "constraints": [],
      "answer_relevance": false,
      "source_step_ids": ["s1"],
      "confidence": "high",
      "selector_enabled": true
    },
    {
      "unit_id": "u_answer_1",
      "unit_type": "answer_slot",
      "target_variable": "?ans",
      "subject": "?x",
      "predicate": "return_as_answer",
      "object": "?ans",
      "constraints": [
        {
          "type": "answer_type",
          "value": "person_or_character"
        }
      ],
      "answer_relevance": true,
      "source_step_ids": ["s3"],
      "confidence": "high",
      "selector_enabled": true
    }
  ]
}
```

### Counterfactual Fields

| Field | Type | Required | Rule |
|---|---|---:|---|
| `cf_id` | string | yes | Stable query-local id |
| `transform` | string | yes | One of `role_swap`, `predicate_shift`, `temporal_shift`, `constraint_flip` |
| `source_unit_id` | string | yes | Positive unit that was mutated |
| `need_units` | array | yes | Full competing unit set after single-unit mutation |

## Diagnostics Schema

```json
{
  "schema_version": "pcrs_rag_v2_need_units_qdmr_v1",
  "parser_status": "ok",
  "compiler_status": "ok",
  "malformed_step_count": 0,
  "malformed_unit_count": 0,
  "dropped_step_count": 0,
  "dropped_unit_count": 0,
  "low_confidence_unit_count": 0,
  "wh_repair_count": 0,
  "answer_step_inserted": false,
  "positive_need_unit_count": 3,
  "counterfactual_set_count": 2,
  "selector_enabled_unit_count": 3,
  "notes": []
}
```

Diagnostics fields must include:

- counts for malformed / dropped / repaired items
- number of selector-enabled units
- counterfactual-set count
- parser/compiler status
- optional examples of malformed objects

## Compiler Overview

### Compiler Inputs

- `question`
- `qdmr_steps`
- `answer_type`
- `question_entities`
- `seed_entities`

### Compiler Outputs

- `positive_need_units`
- `counterfactual_sets`
- `diagnostics`

### Canonical Variable Mapping

Map raw variables to canonical variables:

| Raw | Canonical |
|---|---|
| `X` | `?x` |
| `Y` | `?y` |
| `Z` | `?z` |
| `ANSWER` | `?ans` |

If an input uses an unknown variable:

- mark the step malformed
- do not let the variable pass through

## Compiler Rule Table

### Shared Resolution Priority

When resolving subject or object:

1. explicit variables in `inputs`
2. explicit entities in `inputs`
3. canonical entities matched from `description` against `question_entities`
4. canonical entities matched from `description` against `seed_entities`
5. compiler repair fallback
6. malformed

The compiler must not infer subject/object from loose focus terms alone.

### `locate_entity -> entity_locator`

| Item | Rule |
|---|---|
| Use case | Only when the plan needs disambiguation or explicit referent resolution |
| Subject source | First non-wh entity from `inputs`, else from entity mentions in `description` |
| Predicate | Always `identify` |
| Object | The output variable if present, else `?x` |
| Constraints | Empty |
| Confidence | `high` if subject matches question/seed entities, else `medium` |
| Drop rule | Drop if subject cannot be repaired from entities |
| Selector rule | Enabled unless the subject is unresolved |

Compilation template:

```json
{
  "unit_type": "entity_locator",
  "subject": "<entity_or_mention>",
  "predicate": "identify",
  "object": "?x",
  "constraints": [],
  "answer_relevance": false
}
```

### `relation_lookup -> relation_hop`

| Item | Rule |
|---|---|
| Use case | Retrieve a relation value from an entity or variable |
| Subject source | Prefer first input item |
| Object source | Output variable if present, else second input if it is a variable or entity |
| Predicate source | Extract from `description`, then normalize with the predicate spec |
| Constraints | Extract any role/time/comparison text and move it out of predicate text |
| Confidence | `high` on canonical predicate match, `medium` on alias match, `low` on fallback |
| Drop rule | Drop if canonical predicate cannot be resolved and no safe fallback exists |
| Selector rule | Disabled when `confidence=low` |

Compilation template:

```json
{
  "unit_type": "relation_hop",
  "subject": "<entity_or_variable>",
  "predicate": "<normalized_predicate>",
  "object": "<entity_or_variable>",
  "constraints": [],
  "answer_relevance": false
}
```

### `constraint_check -> constraint_check`

| Item | Rule |
|---|---|
| Use case | Apply temporal/comparative/role/negation/scope constraints |
| Subject source | Prefer first input variable, else most recent variable-bearing unit |
| Predicate | Always `satisfy_constraint` |
| Object | `null` |
| Constraints | One or more canonical constraint objects |
| Confidence | `high` when constraint phrase maps cleanly, else `medium` |
| Drop rule | Drop if no canonical constraint is extracted |
| Selector rule | Enabled unless all constraints are malformed |

Compilation template:

```json
{
  "unit_type": "constraint_check",
  "subject": "<variable_or_entity>",
  "predicate": "satisfy_constraint",
  "object": null,
  "constraints": [
    {
      "type": "temporal|comparative|role|negation|scope",
      "value": "..."
    }
  ],
  "answer_relevance": false
}
```

### `answer -> answer_slot`

| Item | Rule |
|---|---|
| Use case | Bind the final answer variable |
| Subject source | First input variable if present, else last live variable |
| Predicate | Always `return_as_answer` |
| Object | `?ans` |
| Constraints | Must include `{"type":"answer_type","value":"<answer_type>"}` |
| Confidence | Always `high` if subject exists |
| Drop rule | Drop only if no source variable can be identified |
| Selector rule | Always enabled |

Compilation template:

```json
{
  "unit_type": "answer_slot",
  "subject": "<variable_or_entity>",
  "predicate": "return_as_answer",
  "object": "?ans",
  "constraints": [
    {
      "type": "answer_type",
      "value": "<predicted_answer_type>"
    }
  ],
  "answer_relevance": true
}
```

## Shared Compiler Rules

### Rule A: Step Normalization

For every raw step:

1. trim whitespace
2. lowercase the `operation`
3. normalize `step_id`
4. normalize variables:
   - `X -> ?x`
   - `Y -> ?y`
   - `Z -> ?z`
   - `ANSWER -> ?ans`
5. remove repeated punctuation
6. keep the original `description` in debug metadata

### Rule B: Wh-Word Repair

Blocked tokens:

- `who`
- `what`
- `when`
- `where`
- `which`
- `how`
- `why`
- `whom`
- `whose`

If subject or object is a blocked token:

1. try to repair from non-variable `inputs`
2. else try `question_entities`
3. else try `seed_entities`
4. else mark malformed and drop the unit

Repair success increments `wh_repair_count`.

### Rule C: Explicit Constraint Split

Constraint phrases must be removed from the predicate text and stored separately.

Canonical constraint types:

- `temporal`
- `comparative`
- `role`
- `negation`
- `scope`
- `answer_type`

Examples:

- `before 2010 -> {"type":"temporal","value":"before 2010"}`
- `after 2012 -> {"type":"temporal","value":"after 2012"}`
- `largest -> {"type":"comparative","value":"largest"}`
- `played by -> {"type":"role","value":"played_by"}`
- `not founded by -> {"type":"negation","value":"not"}`
- `except -> {"type":"scope","value":"except"}`

### Rule D: Answer-Slot Auto-Insert

If the parser does not emit an explicit answer step:

1. use the last output variable from the final non-answer step
2. if none exists, use the object of the last `relation_hop`
3. emit an `answer_slot`
4. set `diagnostics.answer_step_inserted = true`

### Rule E: Unit Count Cap

Maximum units per query:

- `entity_locator <= 1`
- `relation_hop <= 2`
- `constraint_check <= 2`
- `answer_slot = 1`
- total `<= 6`

If truncation is required, keep in this priority order:

1. `answer_slot`
2. highest-confidence `relation_hop`
3. highest-confidence `constraint_check`
4. `entity_locator`

Dropped units must remain visible in diagnostics.

## Predicate Normalization Spec

This section has the highest priority.

### Canonical Predicate Vocabulary

Initial canonical predicates:

- `identify`
- `associated_university`
- `mascot`
- `founded_by`
- `directed_by`
- `starred_in`
- `played_by`
- `written_by`
- `published_by`
- `author_of`
- `publisher`
- `birth_place`
- `date_of_birth`
- `nationality`
- `capital`
- `located_in`
- `headquartered_in`
- `parent_company`
- `owner_of`
- `child_of`
- `spouse_of`
- `educated_at`
- `won`
- `nominated_for`
- `release_date`
- `start_date`
- `end_date`
- `population`
- `governor_of`
- `screenwriter_of`
- `composer_of`
- `designer_of`
- `named_after`
- `return_as_answer`
- `satisfy_constraint`

### Predicate Alias Table

| Raw phrase family | Canonical predicate |
|---|---|
| `mascot of`, `mascot for` | `mascot` |
| `university associated with`, `college associated with`, `university related to` | `associated_university` |
| `director of`, `directed by` | `directed_by` |
| `actor in`, `star in`, `starred in` | `starred_in` |
| `played by`, `portrayed by` | `played_by` |
| `written by`, `writer of`, `author of` | `written_by` |
| `publisher of`, `published by` | `published_by` |
| `birthplace of`, `born in`, `place of birth` | `birth_place` |
| `date of birth`, `birth date` | `date_of_birth` |
| `nationality of`, `citizen of` | `nationality` |
| `capital of` | `capital` |
| `located in`, `located at`, `part of`, `in the state of` | `located_in` |
| `headquartered in`, `headquarters in` | `headquartered_in` |
| `parent company of` | `parent_company` |
| `owner of`, `owned by` | `owner_of` |
| `child of`, `son of`, `daughter of` | `child_of` |
| `spouse of`, `married to` | `spouse_of` |
| `educated at`, `studied at` | `educated_at` |
| `won`, `received` | `won` |
| `nominated for` | `nominated_for` |
| `released in`, `release date of` | `release_date` |
| `began in`, `started in`, `start date of` | `start_date` |
| `ended in`, `ceased in`, `end date of` | `end_date` |
| `population of` | `population` |
| `governor of` | `governor_of` |
| `screenwriter of` | `screenwriter_of` |
| `composer of` | `composer_of` |
| `designer of`, `designed by` | `designer_of` |
| `named after` | `named_after` |

### Matching Priority

Apply predicate normalization in this order:

1. exact canonical predicate match
2. exact alias-table phrase match
3. longest alias-table substring match in `description`
4. relation template match:
   - `<relation> of X`
   - `X's <relation>`
   - `<relation> for X`
5. variable-centered relation pattern:
   - `Find the <relation> of X`
   - `Return the <relation> for X`
6. if no match:
   - keep `raw_predicate_text`
   - set `confidence=low`
   - set `selector_enabled=false`

The compiler must not construct predicates by concatenating leftover focus tokens.

### Direct-Reject Predicate Patterns

These patterns must be rejected as lexical artifacts unless mapped to a canonical predicate:

- generic two-token leftovers such as `person_goals`
- `publisher_end`
- `birthplace_abolished`
- `region_immediately`
- `body_water`
- `many_times`
- `de_la`
- any predicate composed only from:
  - stopwords
  - wh-words
  - generic placeholders like `person`, `thing`, `place`, `many`, `part`, `type`, `kind`

If such a phrase appears:

- store it as `raw_predicate_text`
- mark the unit `confidence=low`
- set `selector_enabled=false`

### Low-Confidence Predicate Policy

A `relation_hop` is low confidence if any of these hold:

1. no canonical predicate match exists
2. the raw predicate comes only from unmatched leftover tokens
3. the matched phrase is shorter than two meaningful tokens and not in the canonical list
4. the phrase contains only generic nouns without a canonical mapping

Low-confidence units:

- remain in cache for diagnostics
- do not contribute to selector scoring
- count toward `low_confidence_unit_count`

## Constraint Normalization Spec

### Temporal

Canonical forms:

- `before 2010`
- `after 2012`
- `in 1999`
- `during 2005`
- `first term`
- `second term`
- `earlier than X`
- `later than X`

### Comparative

Canonical forms:

- `largest`
- `smallest`
- `higher`
- `lower`
- `earlier`
- `later`
- `greater population`
- `less population`
- `second`
- `first`

### Role

Canonical forms:

- `directed_by`
- `starred_in`
- `played_by`
- `founded_by`
- `written_by`

### Negation

Canonical forms:

- `not`
- `without`
- `excluding`

### Scope

Canonical forms:

- `except`
- `including`
- `among`

If multiple constraints are extracted from one step:

- sort them by type order:
  - `role`
  - `temporal`
  - `comparative`
  - `negation`
  - `scope`
- keep at most `2` per unit

## Counterfactual Mapping Table

### Global Rules

- generate negatives only from selector-enabled positive units
- do not generate negatives from `answer_slot`
- each positive unit may emit at most `2` negatives
- each query may emit at most `6` counterfactual sets
- skip any mutation that would produce a low-confidence predicate

### `role_swap`

| Positive | Negative |
|---|---|
| `directed_by` | `starred_in` |
| `starred_in` | `directed_by` |
| `played_by` | `directed_by` |
| `mascot` | `founded_by` |
| `founded_by` | `mascot` |
| `written_by` | `published_by` |
| `published_by` | `written_by` |

Allowed unit types:

- `relation_hop`
- `entity_locator` only if a disambiguation relation hint exists in debug metadata

### `predicate_shift`

| Positive | Negative |
|---|---|
| `birth_place` | `nationality` |
| `nationality` | `birth_place` |
| `won` | `nominated_for` |
| `nominated_for` | `won` |
| `associated_university` | `educated_at` |
| `educated_at` | `associated_university` |
| `publisher` | `owner_of` |
| `owner_of` | `publisher` |

Allowed unit types:

- `relation_hop`
- `entity_locator` only when the unit stores a relation hint for disambiguation

### `temporal_shift`

Allowed source units:

- `constraint_check`

Rules:

- `in YYYY -> in (YYYY-1)` by default
- `before YYYY -> after YYYY`
- `after YYYY -> before YYYY`
- `earlier -> later`
- `later -> earlier`
- `first term -> second term`
- `second term -> first term`

### `constraint_flip`

Allowed source units:

- `constraint_check`

Rules:

- `largest -> smallest`
- `smallest -> largest`
- `higher -> lower`
- `lower -> higher`
- `except -> including`
- `including -> except`
- `not -> <remove negation>`

### Counterfactual Construction Procedure

For each positive unit in order:

1. skip if `selector_enabled=false`
2. skip if `unit_type=answer_slot`
3. enumerate allowed transforms for that unit type
4. validate that the transformed unit remains canonical
5. replace only the mutated source unit in the full unit set
6. copy the rest of the positive units unchanged
7. stop when the query-level cap is hit

## Malformed / Fallback Policy

### Malformed Step Reasons

Allowed malformed reasons:

- `unknown_operation`
- `invalid_step_id`
- `invalid_output_variable`
- `empty_inputs`
- `wh_subject`
- `wh_object`
- `missing_relation_predicate`
- `constraint_parse_failed`
- `missing_answer_source`

### Malformed Unit Reasons

Allowed malformed reasons:

- `normalization_failed`
- `low_confidence_predicate`
- `missing_subject`
- `missing_object`
- `empty_constraints`

### Query-Level Fallback Triggers

A query falls back to the legacy V1-style requirement builder if any of these hold:

1. `positive_need_units` would be empty
2. no selector-enabled `relation_hop` survives
3. no `answer_slot` can be built
4. `malformed_step_count / raw_step_count > 0.34`
5. `low_confidence_relation_hop_count == total_relation_hop_count`

Fallback behavior:

- preserve the raw step plan in diagnostics
- mark `parser_status = fallback`
- mark `compiler_status = fallback`
- write `fallback_reason`
- allow cache construction to continue

### Selector Admission Policy

Units with:

- `confidence=high`
- `confidence=medium`

may enter selector scoring.

Units with:

- `confidence=low`

must not enter selector scoring.

Counterfactual units with low confidence must not be materialized into `counterfactual_sets`.

## Offline Gates

### Parser Gate

Required thresholds:

- `positive_need_units_nonempty_rate >= 0.95`
- `wh_subject_rate < 0.05`
- `malformed_unit_rate < 0.02`
- `lexical_relation_predicate_rate <= 0.20`
- `queries_with_any_counterfactual_sets >= 0.70`

### Coverage Gate

Required thresholds before QA:

- cache-level positive-vs-negative AUC materially above random
- selected-doc positive mean > selected-doc negative mean
- finalist leakage range no longer near zero on most queries

### Fallback Gate

Track:

- fallback rate
- low-confidence relation-hop rate
- answer-step auto-insert rate
- counterfactual disabled-by-confidence rate

## Annotation Prompt

This prompt is for `(question, need_unit, doc)` labeling.

### Annotation System Prompt

```text
You are a retrieval supervision annotator.

Your task is to judge whether a document aligns with and supports a semantic need unit for a question.

You must separately assess:
1. alignment: whether the document is about the same subject / variable / relation target
2. support: whether the document supports the unit
3. contradiction: whether the document suggests an incompatible value or relation
4. NEI: whether the document is related but does not provide enough evidence

Return JSON only.
Do not explain.
```

### Annotation User Prompt

```text
Question:
{question}

Need unit:
{need_unit_json}

Document title:
{doc_title}

Document text:
{doc_text}

Return:
{
  "alignment": "<one of: aligned, partially_aligned, not_aligned>",
  "support_label": "<one of: supported, contradicted, nei>",
  "confidence": "<one of: high, medium, low>",
  "short_rationale": "one short sentence"
}

Guidelines:
- "aligned" means the document is about the same entity / variable / relation target.
- "supported" means the document provides evidence for the unit.
- "contradicted" means the document gives incompatible evidence.
- "nei" means there is not enough evidence to support or contradict.
- A document can be topically related but still be "nei".
```

### Annotation Score Mapping

Map the annotation output to numeric training targets:

| Label | Value |
|---|---:|
| `aligned` | `1.0` |
| `partially_aligned` | `0.5` |
| `not_aligned` | `0.0` |
| `supported` | `1.0` |
| `contradicted` | `1.0` contradiction target |
| `nei` | `1.0` NEI target |

Derived score:

```text
coverage_score = alignment_score * max(0, support_prob - contradiction_prob)
```

## Worked Examples

### Worked Example 1: Pure 2-Hop

Question:

`Who is the mascot of the university related to Randy Conrads?`

Step plan:

```json
{
  "question_id": "ex_1",
  "answer_type": "person_or_character",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "relation_lookup",
      "description": "Find the university associated with Randy Conrads.",
      "inputs": ["Randy Conrads"],
      "output_variable": "X"
    },
    {
      "step_id": "s2",
      "operation": "relation_lookup",
      "description": "Find the mascot of X.",
      "inputs": ["X"],
      "output_variable": "ANSWER"
    },
    {
      "step_id": "s3",
      "operation": "answer",
      "description": "Return ANSWER as the final answer.",
      "inputs": ["ANSWER"],
      "output_variable": "ANSWER"
    }
  ]
}
```

Compiled positive need units:

```json
[
  {
    "unit_id": "u_relation_1",
    "unit_type": "relation_hop",
    "target_variable": "?x",
    "subject": "Randy Conrads",
    "predicate": "associated_university",
    "raw_predicate_text": "associated with",
    "object": "?x",
    "constraints": [],
    "answer_relevance": false,
    "source_step_ids": ["s1"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_relation_2",
    "unit_type": "relation_hop",
    "target_variable": "?ans",
    "subject": "?x",
    "predicate": "mascot",
    "raw_predicate_text": "mascot of",
    "object": "?ans",
    "constraints": [],
    "answer_relevance": false,
    "source_step_ids": ["s2"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_answer_1",
    "unit_type": "answer_slot",
    "target_variable": "?ans",
    "subject": "?ans",
    "predicate": "return_as_answer",
    "object": "?ans",
    "constraints": [
      {
        "type": "answer_type",
        "value": "person_or_character"
      }
    ],
    "answer_relevance": true,
    "source_step_ids": ["s3"],
    "confidence": "high",
    "selector_enabled": true
  }
]
```

Counterfactual sets:

```json
[
  {
    "cf_id": "cf_1",
    "transform": "predicate_shift",
    "source_unit_id": "u_relation_1",
    "need_units": [
      {
        "unit_id": "u_relation_1_cf",
        "unit_type": "relation_hop",
        "target_variable": "?x",
        "subject": "Randy Conrads",
        "predicate": "educated_at",
        "raw_predicate_text": "educated at",
        "object": "?x",
        "constraints": [],
        "answer_relevance": false,
        "source_step_ids": ["s1"],
        "confidence": "high",
        "selector_enabled": true
      }
    ]
  },
  {
    "cf_id": "cf_2",
    "transform": "role_swap",
    "source_unit_id": "u_relation_2",
    "need_units": [
      {
        "unit_id": "u_relation_2_cf",
        "unit_type": "relation_hop",
        "target_variable": "?ans",
        "subject": "?x",
        "predicate": "founded_by",
        "raw_predicate_text": "founded by",
        "object": "?ans",
        "constraints": [],
        "answer_relevance": false,
        "source_step_ids": ["s2"],
        "confidence": "high",
        "selector_enabled": true
      }
    ]
  }
]
```

Diagnostics:

```json
{
  "schema_version": "pcrs_rag_v2_need_units_qdmr_v1",
  "parser_status": "ok",
  "compiler_status": "ok",
  "malformed_step_count": 0,
  "malformed_unit_count": 0,
  "low_confidence_unit_count": 0,
  "answer_step_inserted": false
}
```

### Worked Example 2: Temporal Constraint

Question:

`Which film directed by Christopher Nolan was released before 2010?`

Compiled units:

```json
[
  {
    "unit_id": "u_relation_1",
    "unit_type": "relation_hop",
    "target_variable": "?x",
    "subject": "Christopher Nolan",
    "predicate": "directed_by",
    "raw_predicate_text": "films directed by",
    "object": "?x",
    "constraints": [],
    "answer_relevance": false,
    "source_step_ids": ["s1"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_constraint_1",
    "unit_type": "constraint_check",
    "target_variable": "?x",
    "subject": "?x",
    "predicate": "satisfy_constraint",
    "object": null,
    "constraints": [
      {
        "type": "temporal",
        "value": "before 2010"
      }
    ],
    "answer_relevance": false,
    "source_step_ids": ["s2"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_answer_1",
    "unit_type": "answer_slot",
    "target_variable": "?ans",
    "subject": "?x",
    "predicate": "return_as_answer",
    "object": "?ans",
    "constraints": [
      {
        "type": "answer_type",
        "value": "film"
      }
    ],
    "answer_relevance": true,
    "source_step_ids": ["s3"],
    "confidence": "high",
    "selector_enabled": true
  }
]
```

Counterfactual unit:

```json
{
  "cf_id": "cf_1",
  "transform": "temporal_shift",
  "source_unit_id": "u_constraint_1",
  "need_units": [
    {
      "unit_id": "u_constraint_1_cf",
      "unit_type": "constraint_check",
      "target_variable": "?x",
      "subject": "?x",
      "predicate": "satisfy_constraint",
      "object": null,
      "constraints": [
        {
          "type": "temporal",
          "value": "after 2010"
        }
      ],
      "answer_relevance": false,
      "source_step_ids": ["s2"],
      "confidence": "high",
      "selector_enabled": true
    }
  ]
}
```

### Worked Example 3: Comparative Constraint

Question:

`Which city has the larger population, the birthplace of Author A or the birthplace of Author B?`

Compiled units:

```json
[
  {
    "unit_id": "u_relation_1",
    "unit_type": "relation_hop",
    "target_variable": "?x",
    "subject": "Author A",
    "predicate": "birth_place",
    "raw_predicate_text": "birthplace of",
    "object": "?x",
    "constraints": [],
    "answer_relevance": false,
    "source_step_ids": ["s1"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_relation_2",
    "unit_type": "relation_hop",
    "target_variable": "?y",
    "subject": "Author B",
    "predicate": "birth_place",
    "raw_predicate_text": "birthplace of",
    "object": "?y",
    "constraints": [],
    "answer_relevance": false,
    "source_step_ids": ["s2"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_constraint_1",
    "unit_type": "constraint_check",
    "target_variable": "?ans",
    "subject": "?ans",
    "predicate": "satisfy_constraint",
    "object": null,
    "constraints": [
      {
        "type": "comparative",
        "value": "larger population"
      }
    ],
    "answer_relevance": false,
    "source_step_ids": ["s3"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_answer_1",
    "unit_type": "answer_slot",
    "target_variable": "?ans",
    "subject": "?ans",
    "predicate": "return_as_answer",
    "object": "?ans",
    "constraints": [
      {
        "type": "answer_type",
        "value": "city"
      }
    ],
    "answer_relevance": true,
    "source_step_ids": ["s4"],
    "confidence": "high",
    "selector_enabled": true
  }
]
```

Counterfactual:

```json
{
  "cf_id": "cf_1",
  "transform": "constraint_flip",
  "source_unit_id": "u_constraint_1",
  "need_units": [
    {
      "unit_id": "u_constraint_1_cf",
      "unit_type": "constraint_check",
      "target_variable": "?ans",
      "subject": "?ans",
      "predicate": "satisfy_constraint",
      "object": null,
      "constraints": [
        {
          "type": "comparative",
          "value": "smaller population"
        }
      ],
      "answer_relevance": false,
      "source_step_ids": ["s3"],
      "confidence": "high",
      "selector_enabled": true
    }
  ]
}
```

### Worked Example 4: Role Confusion

Question:

`Who is played by the director of The Good Shepherd in The Godfather?`

Compiled units:

```json
[
  {
    "unit_id": "u_entity_1",
    "unit_type": "entity_locator",
    "target_variable": "?x",
    "subject": "The Good Shepherd",
    "predicate": "identify",
    "object": "?x",
    "constraints": [],
    "answer_relevance": false,
    "source_step_ids": ["s1"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_relation_1",
    "unit_type": "relation_hop",
    "target_variable": "?y",
    "subject": "?x",
    "predicate": "directed_by",
    "raw_predicate_text": "director of",
    "object": "?y",
    "constraints": [],
    "answer_relevance": false,
    "source_step_ids": ["s2"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_relation_2",
    "unit_type": "relation_hop",
    "target_variable": "?ans",
    "subject": "?y",
    "predicate": "played_by",
    "raw_predicate_text": "played by",
    "object": "?ans",
    "constraints": [
      {
        "type": "scope",
        "value": "in The Godfather"
      }
    ],
    "answer_relevance": false,
    "source_step_ids": ["s3"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_answer_1",
    "unit_type": "answer_slot",
    "target_variable": "?ans",
    "subject": "?ans",
    "predicate": "return_as_answer",
    "object": "?ans",
    "constraints": [
      {
        "type": "answer_type",
        "value": "person_or_character"
      }
    ],
    "answer_relevance": true,
    "source_step_ids": ["s4"],
    "confidence": "high",
    "selector_enabled": true
  }
]
```

Counterfactual:

```json
{
  "cf_id": "cf_1",
  "transform": "role_swap",
  "source_unit_id": "u_relation_1",
  "need_units": [
    {
      "unit_id": "u_relation_1_cf",
      "unit_type": "relation_hop",
      "target_variable": "?y",
      "subject": "?x",
      "predicate": "starred_in",
      "raw_predicate_text": "actor in",
      "object": "?y",
      "constraints": [],
      "answer_relevance": false,
      "source_step_ids": ["s2"],
      "confidence": "high",
      "selector_enabled": true
    }
  ]
}
```

### Worked Example 5: Entity Disambiguation with Auto-Inserted Answer Step

Question:

`Which Springfield is the capital of Illinois?`

Parser output without answer step:

```json
{
  "question_id": "ex_5",
  "answer_type": "city",
  "qdmr_steps": [
    {
      "step_id": "s1",
      "operation": "locate_entity",
      "description": "Identify the Springfield that is in Illinois.",
      "inputs": ["Springfield", "Illinois"],
      "output_variable": "X"
    },
    {
      "step_id": "s2",
      "operation": "constraint_check",
      "description": "Keep X if it is the capital of Illinois.",
      "inputs": ["X", "Illinois"],
      "output_variable": "X"
    }
  ]
}
```

Compiled units:

```json
[
  {
    "unit_id": "u_entity_1",
    "unit_type": "entity_locator",
    "target_variable": "?x",
    "subject": "Springfield",
    "predicate": "identify",
    "object": "?x",
    "constraints": [
      {
        "type": "scope",
        "value": "in Illinois"
      }
    ],
    "answer_relevance": false,
    "source_step_ids": ["s1"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_constraint_1",
    "unit_type": "constraint_check",
    "target_variable": "?x",
    "subject": "?x",
    "predicate": "satisfy_constraint",
    "object": null,
    "constraints": [
      {
        "type": "scope",
        "value": "capital of Illinois"
      }
    ],
    "answer_relevance": false,
    "source_step_ids": ["s2"],
    "confidence": "high",
    "selector_enabled": true
  },
  {
    "unit_id": "u_answer_auto_1",
    "unit_type": "answer_slot",
    "target_variable": "?ans",
    "subject": "?x",
    "predicate": "return_as_answer",
    "object": "?ans",
    "constraints": [
      {
        "type": "answer_type",
        "value": "city"
      }
    ],
    "answer_relevance": true,
    "source_step_ids": [],
    "confidence": "high",
    "selector_enabled": true
  }
]
```

Diagnostics:

```json
{
  "schema_version": "pcrs_rag_v2_need_units_qdmr_v1",
  "parser_status": "ok",
  "compiler_status": "ok",
  "malformed_step_count": 0,
  "malformed_unit_count": 0,
  "low_confidence_unit_count": 0,
  "answer_step_inserted": true
}
```

## Implementation Checklist

Before any major code rewrite, the implementation must satisfy these design checkpoints:

1. LLM parser returns only step plans.
2. Compiler never constructs predicates from leftover focus-term concatenation.
3. Constraints are explicit objects, not embedded inside predicate text.
4. Low-confidence relation units are blocked from selector scoring.
5. Counterfactual construction is skipped only by explicit policy, not because no mutation path exists.
6. Diagnostics persist enough detail to audit:
   - parser output quality
   - predicate normalization success
   - low-confidence units
   - fallback decisions

## Next Implementation Order

Recommended landing order:

1. Add step-plan prompt support and parser cache fields.
2. Implement strict schema validation and step normalization.
3. Implement predicate normalization with a direct-reject list.
4. Implement deterministic compiler with diagnostics.
5. Implement counterfactual mapping table and generation.
6. Rebuild a small cache and run offline gates.
7. Only after offline gates pass, connect the new cache into oracle QA runs.
