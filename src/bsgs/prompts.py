"""Prompt templates for BSGS Week-0/Week-1 scripts."""

QWEN_SLOT_GENERATION_PROMPT = """Given a multi-hop question, decompose it into atomic information needs.

Return only a JSON list. Each item should be a slot with:
- slot_text: the atomic information need
- input_variables: known entities or variables required by this slot
- output_variable: the variable produced by this slot
- expected_answer_type: person/place/date/work/organization/number/other

Do not answer the question.
Do not infer facts.
Do not include dependency edges.

Question:
{question}
"""


PROPOSITION_EXTRACTION_PROMPT = """Given the question, the current slot, and a retrieved passage,
extract atomic but context-rich propositions that may help satisfy the slot.

Rules:
1. Each proposition must be directly supported by the passage.
2. Keep enough context to avoid ambiguous triples.
3. Do not infer unstated facts.
4. Return the exact source span.
5. Return JSON only.

Question:
{question}

Current slot:
{slot_text}

Passage:
{passage}

Return:
[
  {{
    "proposition": "...",
    "source_span": "...",
    "mentions": ["..."]
  }}
]
"""
