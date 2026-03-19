from argparse import ArgumentTypeError
from dataclasses import dataclass, asdict
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class CausalRelation:
    source_fact_id: str
    target_fact_id: str
    relation_type: Literal["causes", "enables", "prevents"]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalRawOutput:
    chunk_id: str
    response: str
    causal_relations: List[CausalRelation]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None
    retrieval_trace: Optional[Dict[str, Any]] = None
    qa_trace: Optional[Dict[str, Any]] = None


    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
            "retrieval_trace": self.retrieval_trace,
            "qa_trace": self.qa_trace,
        }


_ANSWER_PATTERNS = (
    re.compile(r"(?is)\banswer\s*:\s*(.+)"),
    re.compile(r"(?is)\bfinal\s+answer\s*:\s*(.+)"),
)
_ANSWER_STOP_PATTERN = re.compile(r"\n\s*(?:question|thought|reasoning|explanation)\s*:", re.IGNORECASE)
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:the\s+answer\s+is|answer\s+is|the\s+final\s+answer\s+is|final\s+answer\s+is)\s+",
    re.IGNORECASE,
)
_WEAK_ANSWER_PATTERNS = (
    re.compile(r"(?is)\bso\s+the\s+answer\s+is\s*:\s*([^\n]+)"),
    re.compile(r"(?is)\bthe\s+answer\s+is\s*:\s*([^\n]+)"),
    re.compile(r"(?is)\bso\s+the\s+answer\s+is\s+([^\n.?!]+(?:[.?!])?)"),
    re.compile(r"(?is)\bthe\s+answer\s+is\s+([^\n.?!]+(?:[.?!])?)"),
)


def extract_answer_from_response(response_content: Any) -> Tuple[str, Dict[str, Any]]:
    response_type = type(response_content).__name__
    if response_content is None:
        return "", {
            "used_fallback": True,
            "error_type": "none_response",
            "response_type": response_type,
        }

    if not isinstance(response_content, str):
        return str(response_content), {
            "used_fallback": True,
            "error_type": "non_string_response",
            "response_type": response_type,
        }

    raw_text = response_content
    raw_text = raw_text.strip()
    if not raw_text:
        return "", {
            "used_fallback": True,
            "error_type": "empty_response",
            "response_type": response_type,
        }

    answer_text = None
    for pattern in _ANSWER_PATTERNS:
        matches = list(pattern.finditer(raw_text))
        if matches:
            answer_text = matches[-1].group(1).strip()
            break

    if answer_text is None:
        for pattern in _WEAK_ANSWER_PATTERNS:
            matches = list(pattern.finditer(raw_text))
            if matches:
                answer_text = matches[-1].group(1).strip()
                break
        if answer_text is None:
            return raw_text, {
                "used_fallback": True,
                "error_type": "missing_answer_marker",
                "response_type": response_type,
            }

    stop_match = _ANSWER_STOP_PATTERN.search(answer_text)
    if stop_match:
        answer_text = answer_text[:stop_match.start()].strip()

    answer_lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if answer_lines:
        answer_text = answer_lines[0]

    answer_text = _ANSWER_PREFIX_PATTERN.sub("", answer_text).strip()
    answer_text = answer_text.strip("*`'\" \t\r\n")

    if not answer_text:
        return raw_text, {
            "used_fallback": True,
            "error_type": "empty_answer_after_parse",
            "response_type": response_type,
        }

    return answer_text, {
        "used_fallback": False,
        "error_type": "weak_answer_marker" if all(
            not pattern.search(raw_text) for pattern in _ANSWER_PATTERNS
        ) else None,
        "response_type": response_type,
    }

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()


def normalize_triple(triple: Union[List[Any], Tuple[Any, Any, Any]]) -> Tuple[str, str, str]:
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        raise ValueError(f"Invalid triple for normalization: {triple}")
    normalized = text_processing(list(triple))
    return tuple(str(item) for item in normalized)


def compute_fact_id(triple: Union[List[Any], Tuple[Any, Any, Any]]) -> str:
    normalized_triple = normalize_triple(triple)
    return compute_mdhash_id(str(normalized_triple), prefix="fact-")


def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, CausalRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }
    causal_output_dict = {
        chunk_item['idx']: CausalRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            causal_relations=[
                CausalRelation(
                    source_fact_id=str(relation["source_fact_id"]),
                    target_fact_id=str(relation["target_fact_id"]),
                    relation_type=str(relation["relation_type"]),
                    confidence=float(relation.get("confidence", 0.0)),
                )
                for relation in chunk_item.get('extracted_causal_relations', [])
                if isinstance(relation, dict)
                and "source_fact_id" in relation
                and "target_fact_id" in relation
                and "relation_type" in relation
            ]
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict, causal_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
