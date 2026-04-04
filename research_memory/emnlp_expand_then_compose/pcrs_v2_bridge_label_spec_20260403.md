# PCRS-RAG V2 Bridge Label Spec

Date: 2026-04-03

## Purpose

This note defines a strict supervision rule for `bridge_support`.

The current atomic scorer does not fail because the classifier is too weak first.
It fails because the training data almost never teaches what a true bridge is.

This spec exists to keep the next annotation round from collapsing back to:

- topical relatedness
- entity overlap
- title similarity
- vague usefulness

## Primary Question

For a `(question, need_unit, doc)` triple, ask:

> does this document contribute a necessary intermediate variable or relation that helps close the reasoning chain for this specific need unit, even if it does not directly complete the unit alone?

If the answer is yes, it may be `bridge_support`.

If the document is only related to the topic, it is not bridge support.

## Label Set

This wave uses a bridge-first view.

Primary label:

1. `bridge_support`
2. `not_bridge`

Optional subtype:

1. `full_support`
2. `bridge_support`
3. `nei`
4. `contradiction`

The primary optimization target is still:

`is_bridge_support`

## Definition: Bridge Support

A document is `bridge_support` for a need unit only if it satisfies both conditions below.

### Condition A: Intermediate Utility

The document provides at least one of the following:

1. an intermediate entity needed by an adjacent hop
2. a relation value that becomes the input variable for the next hop
3. one half of a title-body split fact that closes a variable transition
4. a role or constraint resolution that makes the next hop consumable

### Condition B: Chain Continuity

The contribution must be usable in a continuous reasoning chain.

This means at least one of the following:

1. the doc output can be consumed by the next hop variable
2. the doc resolves the subject that another hop queries over
3. the doc disambiguates a node that would otherwise point to the wrong branch
4. the doc supplies a bridge entity that is explicitly reused downstream

If Condition A holds but the variable chain is not continuous, the label is not bridge support.

## Definition: Full Support

A document is `full_support` if it directly supports the need unit itself.

Examples:

1. it directly gives the requested relation value for the unit
2. it directly confirms the required time/role/comparison constraint
3. it directly supplies the answer slot target for the unit

Rule:

If a document directly closes the unit, prefer `full_support` over `bridge_support`.

## Definition: NEI

A document is `nei` if it does not provide enough information to support or contradict the need unit.

This includes:

1. topic-adjacent docs
2. shared-entity docs with no useful chain continuation
3. docs that mention the same subject but not the needed relation or bridge variable
4. docs that contain relevant words but do not advance the chain

## Definition: Contradiction

A document is `contradiction` if it clearly pushes the unit toward the wrong relation, role, time, or comparison outcome.

Examples:

1. wrong role
2. wrong relation slot
3. opposite temporal/comparative constraint
4. explicit incompatible entity binding

## Positive Criteria for Bridge Support

Label `bridge_support` only if at least one of these is true:

1. the doc introduces the missing middle entity for a multi-hop chain
2. the doc anchors the correct subject while another hop consumes the relation
3. the doc provides a relation value that the current unit cannot finish without
4. the doc is not directly answering the unit, but removing it would break downstream closure
5. title and body together close a relation transition that the unit depends on

## Negative Criteria

Do not label `bridge_support` for any of these alone:

1. same question entity appears
2. same wiki topic appears
3. title is similar
4. predicate words overlap
5. answer type matches
6. document feels generally useful

These are all bridge-like distractors unless variable continuity is explicit.

## Bridge-Like Hard Negative

A `bridge_like_hard_negative` is a document that looks bridge-like structurally but does not actually close the chain.

Typical patterns:

1. title contains a shared entity, but relation goes to the wrong branch
2. predicate looks relevant, but the object cannot be consumed by the next hop
3. document overlaps with the same local topic, but not the required variable slot
4. document supplies a plausible distractor entity instead of the true bridge entity

This class is critical because the current scorer is drifting toward exactly these docs.

## Decision Rules

Use this order:

1. If the doc directly supports the unit, label `full_support`.
2. Else if the doc contributes a necessary intermediate and preserves chain continuity, label `bridge_support`.
3. Else if the doc clearly pushes the wrong relation or constraint, label `contradiction`.
4. Else label `nei`.

## Examples

### Example 1: True Bridge

Question:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Need unit:

`death_place(designer_of(Southeast Library)) -> ?y`

Candidate doc:

`Southeast Library`

Why it can be bridge:

1. it may identify the correct designer
2. that designer becomes the input to the next hop
3. it does not answer the final unit directly

Label:

`bridge_support`

### Example 2: Topic Neighbor, Not Bridge

Same question and unit.

Candidate doc:

`Davenport Public Library`

Why not bridge:

1. same library topic family
2. but no continuous variable chain to the needed designer/death-place path

Label:

`nei`

### Example 3: Direct Support

Need unit:

`publisher(Labyrinth) -> ?x`

Candidate doc:

`Acornsoft`

If the doc directly states the publisher relation, it is:

`full_support`

not `bridge_support`.

### Example 4: Bridge-Like Hard Negative

Question:

`When was Lady Godiva's birthplace abolished?`

Candidate doc:

`Lady Godiva Rides Again`

Why not bridge:

1. strong lexical overlap with `Lady Godiva`
2. wrong entity branch
3. no continuous path to birthplace or abolition

Label:

`nei`

## Annotation Priority

When building the first probe set, prioritize:

1. regressed smoke10 queries
2. known 3-hop chain cases
3. docs with non-zero bridge structural features but low support
4. docs that replaced a better baseline bridge candidate
5. high-overlap wrong-branch distractors

## Output Contract

For seed labels used by training, store:

1. `sample_id`
2. `label`
3. `support_subtype`
4. `label_source`
5. `label_confidence`
6. `reason`

Recommended values:

1. `label_source = assistant_seed_bridge_spec_v1`
2. `label_confidence in {high, medium}`

Only `high` confidence labels should be fed into the first retraining round by default.
