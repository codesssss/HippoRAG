from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
- Use specific typed relations instead of generic ones. Follow these evidence-frame guidelines:

Temporal attributes:
  Use "born on", "born in", "died on", "died in", "released in", "established in", "founded in", "opened in" with the actual date or year as the object. Do NOT fold dates into generic relations like "is a" or "occurred during".
  Example: ["Alice", "born on", "1 January 1990"] not ["Alice", "is a", "person born in 1990"]

Place and origin attributes:
  Use "born in", "located in", "headquartered in", "country of origin", "from" with the place as the object. Preserve the direction: the entity is the subject, the place is the object.
  Example: ["Radio City", "located in", "India"] not ["India", "includes", "Radio City"]

Work metadata roles:
  Use "directed by", "written by", "composed by", "performed by", "produced by" with the person as the object. For the reverse direction use "directed", "wrote", "composed".
  Example: ["El Tonto", "directed by", "Charlie Day"] not ["El Tonto", "cast includes", "Charlie Day"]

Person relation roles:
  Use "mother", "father", "spouse", "child", "predecessor", "successor" as the relation. The subject is the person, the object is the related person.
  Example: ["Lothair II", "mother", "Ermengarde of Tours"] not ["Ermengarde of Tours", "family member of", "Lothair II"]

Office and position roles:
  Use "served as", "appointed as", "president of", "secretary of" preserving the office holder as subject and the title or organization as object.

Event and competition roles:
  Use "won", "participated in", "drafted by", "played in" with the event or competition as the object.
  Example: ["Player", "drafted by", "Team"] not ["Team", "includes", "Player"]

"""

v13b_one_shot_paragraph = """Ermengarde of Tours
Ermengarde of Tours (804 – 20 March 851) was a Frankish empress. She was the wife of Lothair I and the mother of Lothair II. She was born in Tours, France."""

v13b_one_shot_entities = """{"named_entities":
    ["Ermengarde of Tours", "804", "20 March 851", "Lothair I", "Lothair II", "Tours", "France"]
}
"""

v13b_one_shot_output = """{"triples": [
            ["Ermengarde of Tours", "born on", "804"],
            ["Ermengarde of Tours", "died on", "20 March 851"],
            ["Ermengarde of Tours", "born in", "Tours"],
            ["Tours", "located in", "France"],
            ["Ermengarde of Tours", "spouse", "Lothair I"],
            ["Ermengarde of Tours", "mother", "Lothair II"],
            ["Lothair II", "mother", "Ermengarde of Tours"]
    ]
}
"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=v13b_one_shot_paragraph,
    named_entity_json=v13b_one_shot_entities,
)


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": v13b_one_shot_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)},
]
