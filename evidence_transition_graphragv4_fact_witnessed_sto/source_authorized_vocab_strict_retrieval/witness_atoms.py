"""Source-warranted fact atoms for witness feasibility analysis.

Atoms are not generated propositions.  They are normalized OpenIE triples whose
subject and object can be grounded back to a sentence in the source passage.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence, Tuple

from .certificate_graph import EvidenceNode, Triple
from .normalize import normalize_text, phrase_occurs, title_aliases, tokens


WITNESS_ATOM_CONTRACT: Mapping[str, bool | str] = {
    "atom_unit": "source_warranted_openie_fact_atom",
    "uses_llm_generated_propositions": False,
    "uses_role_vocabulary": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
}


@dataclass(frozen=True)
class SourceWarrantedAtom:
    source_doc_index: int
    source_title: str
    subject: str
    predicate: str
    object: str
    normalized_subject: str
    normalized_predicate: str
    normalized_object: str
    source_sentence: str
    evidence_unit_id: str

    @property
    def endpoint_entities(self) -> Tuple[str, str]:
        return (self.normalized_subject, self.normalized_object)


def source_warranted_atoms_from_node(node: EvidenceNode) -> Tuple[SourceWarrantedAtom, ...]:
    """Return source-warranted OpenIE atoms for one passage."""

    output = []
    seen = set()
    for triple in node.triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, predicate, obj = (str(triple[0]), str(triple[1]), str(triple[2]))
        normalized_subject = normalize_text(subject)
        normalized_predicate = normalize_text(predicate)
        normalized_object = normalize_text(obj)
        if not normalized_subject or not normalized_object:
            continue
        source_sentence = source_sentence_for_atom(
            node=node,
            subject=subject,
            predicate=predicate,
            obj=obj,
        )
        if not source_sentence:
            continue
        evidence_unit_id = "::".join(
            (normalized_subject, normalized_predicate, normalized_object)
        )
        key = (int(node.doc_index), evidence_unit_id, normalize_text(source_sentence))
        if key in seen:
            continue
        seen.add(key)
        output.append(
            SourceWarrantedAtom(
                source_doc_index=int(node.doc_index),
                source_title=node.display_title,
                subject=subject,
                predicate=predicate,
                object=obj,
                normalized_subject=normalized_subject,
                normalized_predicate=normalized_predicate,
                normalized_object=normalized_object,
                source_sentence=source_sentence,
                evidence_unit_id=evidence_unit_id,
            )
        )
    return tuple(output)


def source_sentence_for_atom(
    *,
    node: EvidenceNode,
    subject: object,
    predicate: object,
    obj: object,
) -> str:
    """Find a sentence in node text warranting the triple endpoints."""

    subject_aliases = _endpoint_aliases(subject, node)
    object_aliases = _endpoint_aliases(obj, node)
    predicate_terms = tuple(
        token for token in tokens(predicate) if len(token) >= 4
    )
    passage_text = str(node.text).split("\n", 1)[1] if "\n" in str(node.text) else str(node.text)
    for sentence in _sentences(passage_text):
        if not _any_phrase_occurs(subject_aliases, sentence):
            continue
        if not _any_phrase_occurs(object_aliases, sentence):
            continue
        if predicate_terms and not any(token in set(tokens(sentence)) for token in predicate_terms):
            continue
        return sentence.strip()
    for sentence in _sentences(passage_text):
        if _any_phrase_occurs(subject_aliases, sentence) and _any_phrase_occurs(
            object_aliases,
            sentence,
        ):
            return sentence.strip()
    return ""


def atom_entities_for_doc(
    *,
    node: EvidenceNode,
    atoms: Sequence[SourceWarrantedAtom],
) -> Tuple[str, ...]:
    entities = []
    for alias in title_aliases(node.display_title):
        if alias and alias not in entities:
            entities.append(alias)
    for atom in atoms:
        for entity in atom.endpoint_entities:
            if entity and entity not in entities:
                entities.append(entity)
    return tuple(entities)


def title_entities_for_doc(node: EvidenceNode) -> Tuple[str, ...]:
    """Return normalized title aliases used as passage entity bindings."""

    aliases = []
    raw_title = str(node.display_title or "").strip()
    for alias in title_aliases(raw_title):
        if alias and alias not in aliases:
            aliases.append(alias)
    for extra in _witness_title_aliases(raw_title):
        if extra and extra not in aliases:
            aliases.append(extra)
    return tuple(aliases)


def _witness_title_aliases(title: object) -> Tuple[str, ...]:
    """Return extra title aliases for witness-only entity binding.

    These aliases are deterministic title-shape reductions.  They are not role
    vocab or query-specific dataset routing.
    """

    raw = str(title or "").strip()
    output = []
    for parenthetical in re.findall(r"\(([^)]{2,80})\)", raw):
        normalized = normalize_text(parenthetical)
        if normalized:
            output.append(normalized)
    normalized_title = normalize_text(raw)
    for prefix in (
        "geography of ",
        "history of ",
        "list of ",
        "timeline of ",
        "outline of ",
    ):
        if normalized_title.startswith(prefix):
            output.append(normalized_title[len(prefix) :].strip())
    if normalized_title.endswith(" history"):
        output.append(normalized_title[: -len(" history")].strip())
    possessive_prefix = normalized_title.split(" s ", 1)[0].strip()
    if possessive_prefix and possessive_prefix != normalized_title:
        output.append(possessive_prefix)
    return tuple(dict.fromkeys(alias for alias in output if alias))


def _endpoint_aliases(endpoint: object, node: EvidenceNode) -> Tuple[str, ...]:
    normalized = normalize_text(endpoint)
    aliases = []
    if normalized:
        aliases.append(normalized)
    node_title_aliases = title_aliases(node.display_title)
    if normalized and any(
        normalized == alias
        or phrase_occurs(alias, normalized)
        or phrase_occurs(normalized, alias)
        for alias in node_title_aliases
    ):
        aliases.extend(alias for alias in node_title_aliases if alias)
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _any_phrase_occurs(phrases: Sequence[str], text: object) -> bool:
    return any(phrase_occurs(phrase, text) for phrase in phrases if phrase)


def _sentences(text: str) -> Tuple[str, ...]:
    parts = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text))
        if sentence.strip()
    ]
    return tuple(parts)
