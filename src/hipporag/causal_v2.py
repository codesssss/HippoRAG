import ast
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .embedding_store import EmbeddingStore
from .prompts.linking import get_query_instruction
from .utils.llm_utils import fix_broken_generated_json
from .utils.logging_utils import get_logger
from .utils.misc_utils import compute_mdhash_id, text_processing

logger = get_logger(__name__)


_CAUSAL_RELATION_MAP = {
    "cause": "cause",
    "causes": "cause",
    "caused": "cause",
    "caused_by": "cause",
    "enable": "enable",
    "enables": "enable",
    "enabled": "enable",
    "prevent": "prevent",
    "prevents": "prevent",
    "prevented": "prevent",
}

_CAUSAL_RELATION_TO_TEXT = {
    "cause": "causes",
    "enable": "enables",
    "prevent": "prevents",
}

_GENERAL_RELATION_MAP = {
    "director_of": "director_of",
    "director": "director_of",
    "directed": "director_of",
    "directed_by": "director_of",
    "film_director": "director_of",
    "born_on": "born_on",
    "date_of_birth": "born_on",
    "birth_date": "born_on",
    "died_on": "died_on",
    "date_of_death": "died_on",
    "death_date": "died_on",
    "citizen_of": "citizen_of",
    "country_of_citizenship": "citizen_of",
    "nationality": "citizen_of",
    "born_in": "born_in",
    "place_of_birth": "born_in",
    "birth_place": "born_in",
    "parent_of": "parent_of",
    "parent": "parent_of",
    "father": "parent_of",
    "mother": "parent_of",
    "spouse_of": "spouse_of",
    "spouse": "spouse_of",
    "husband": "spouse_of",
    "wife": "spouse_of",
    "married_to": "spouse_of",
    "related_to": "related_to",
}

_GENERAL_RELATION_TO_TEXT = {
    "director_of": "is director of",
    "born_on": "was born on",
    "died_on": "died on",
    "citizen_of": "is citizen of",
    "born_in": "was born in",
    "parent_of": "is parent of",
    "spouse_of": "is spouse of",
    "related_to": "is related to",
}

_QUERY_ENTITY_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "from", "by",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
    "same", "both", "film", "films", "movie", "movies", "song", "songs",
    "director", "directors", "performer", "performers", "composer", "composers",
    "father", "mother", "husband", "wife", "born", "birth", "place", "study",
    "studied", "released", "release", "country", "nationality", "older", "younger",
    "later", "earlier", "died", "die", "death", "located", "location", "city",
}


def _normalize_text(value: str) -> str:
    normalized = text_processing(value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _canonicalize_relation(value: Any, graph_mode: str = "causal") -> Optional[str]:
    if value is None:
        return None
    normalized = _normalize_text(str(value)).replace(" ", "_")
    if graph_mode == "general":
        return _GENERAL_RELATION_MAP.get(normalized)
    return _CAUSAL_RELATION_MAP.get(normalized)


def _clip_confidence(value: Any, default: float = 0.75) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return float(max(0.0, min(1.0, confidence)))


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        fixed = fix_broken_generated_json(raw_text)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(fixed)
    if not isinstance(parsed, dict):
        raise ValueError("Extraction output must be a JSON object.")
    return parsed


def _stringify_list(values: Sequence[str]) -> list[str]:
    return [str(value) for value in values]


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _text_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    union = left_tokens | right_tokens
    jaccard = (len(left_tokens & right_tokens) / len(union)) if union else 0.0
    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    containment = 1.0 if left_norm in right_norm or right_norm in left_norm else 0.0
    return max(jaccard, ratio, containment)


class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass
class ExtractedChunk:
    chunk_id: str
    title: str
    content: str
    events: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    raw_response_preview: str
    parse_error: Optional[str] = None


class SemanticIntentRouter:
    def __init__(self, embedding_model, config) -> None:
        self.embedding_model = embedding_model
        self.config = config
        self.anchor_bank = self._load_anchor_bank()
        self.anchor_group_to_scores: dict[str, float] = {}
        self.anchor_ids: list[str] = []
        self.anchor_labels: list[str] = []
        self.anchor_texts: list[str] = []
        for label, texts in self.anchor_bank.items():
            for idx, text in enumerate(texts):
                self.anchor_ids.append(f"{label}:{idx}")
                self.anchor_labels.append(label)
                self.anchor_texts.append(text)
        if self.anchor_texts:
            self.anchor_embeddings = np.asarray(
                self.embedding_model.batch_encode(
                    self.anchor_texts,
                    instruction=get_query_instruction("query_to_passage"),
                    norm=True,
                ),
                dtype=float,
            )
        else:
            self.anchor_embeddings = np.zeros((0, 0), dtype=float)

    def _load_anchor_bank(self) -> dict[str, list[str]]:
        anchor_path = getattr(self.config, "causal_router_anchor_path", None)
        if anchor_path:
            candidate = anchor_path
        else:
            candidate = os.path.join(
                os.path.dirname(__file__),
                "resources",
                "causal_router_anchor_bank.json",
            )
        with open(candidate, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            str(label): [str(text) for text in texts]
            for label, texts in data.items()
            if isinstance(texts, list)
        }

    def route(self, query_embedding: np.ndarray) -> dict[str, Any]:
        if self.anchor_embeddings.size == 0:
            return {
                "label": "standard",
                "is_causal": False,
                "score": 0.0,
                "margin": 0.0,
                "best_causal_label": "effect",
                "best_causal_score": 0.0,
                "standard_score": 0.0,
                "scores_by_label": {},
            }
        query_vector = np.asarray(query_embedding, dtype=float).reshape(-1)
        similarity = np.dot(self.anchor_embeddings, query_vector.T)
        scores_by_label: dict[str, float] = defaultdict(float)
        for label, score in zip(self.anchor_labels, similarity.tolist()):
            if score > scores_by_label[label]:
                scores_by_label[label] = float(score)

        causal_labels = ("cause", "effect", "prevention")
        best_causal_label = max(causal_labels, key=lambda label: scores_by_label.get(label, 0.0))
        best_causal_score = float(scores_by_label.get(best_causal_label, 0.0))
        standard_score = float(scores_by_label.get("standard", 0.0))
        margin = best_causal_score - standard_score
        causal_threshold = float(getattr(self.config, "causal_router_causal_threshold", 0.62))
        standard_threshold = float(getattr(self.config, "causal_router_standard_threshold", 0.62))
        margin_threshold = float(getattr(self.config, "causal_router_margin_threshold", 0.02))

        is_causal = bool(
            best_causal_score >= causal_threshold
            and margin >= margin_threshold
            and not (standard_score >= standard_threshold and standard_score > best_causal_score)
        )
        label = best_causal_label if is_causal else "standard"
        return {
            "label": label,
            "is_causal": is_causal,
            "score": best_causal_score if is_causal else standard_score,
            "margin": margin,
            "best_causal_label": best_causal_label,
            "best_causal_score": best_causal_score,
            "standard_score": standard_score,
            "scores_by_label": {label: float(score) for label, score in scores_by_label.items()},
        }


class CausalV2Engine:
    def __init__(self, hipporag) -> None:
        self.hipporag = hipporag
        self.config = hipporag.global_config
        self.index_dir = os.path.join(hipporag.working_dir, "causal_v2")
        os.makedirs(self.index_dir, exist_ok=True)
        self.event_store = EmbeddingStore(
            hipporag.embedding_model,
            os.path.join(self.index_dir, "event_embeddings"),
            self.config.embedding_batch_size,
            "causal_event_v2",
        )
        self.events_path = os.path.join(self.index_dir, "event_nodes.json")
        self.edges_path = os.path.join(self.index_dir, "causal_edges.json")
        self.chunk_map_path = os.path.join(self.index_dir, "chunk_to_event_ids.json")
        self.manifest_path = os.path.join(self.index_dir, "manifest.json")
        self.router = SemanticIntentRouter(hipporag.embedding_model, self.config)
        self.event_nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.chunk_to_event_ids: dict[str, list[str]] = {}
        self.adjacency_out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.adjacency_in: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.event_ids: list[str] = []
        self.event_embeddings = np.zeros((0, 0), dtype=float)
        self.loaded = False

    def _graph_mode(self) -> str:
        return str(getattr(self.config, "causal_v2_graph_mode", "causal")).lower()

    def _manifest_version(self) -> int:
        return 2

    def index(self, chunk_rows: dict[str, dict[str, Any]]) -> None:
        current_chunk_ids = sorted(chunk_rows.keys())
        manifest = self.read_manifest()
        if (
            not self.config.force_index_from_scratch
            and self._has_complete_index()
            and manifest.get("chunk_ids", []) == current_chunk_ids
            and int(manifest.get("version", 0) or 0) == self._manifest_version()
            and str(manifest.get("graph_mode", "causal")).lower() == self._graph_mode()
        ):
            self.load()
            logger.info("Loaded causal V2 index from disk.")
            return

        logger.info("Building causal V2 index from scratch.")
        extracted_chunks = self._extract_chunks(chunk_rows)
        self._build_index(chunk_rows, extracted_chunks)
        self.load()

    def _has_complete_index(self) -> bool:
        return all(
            os.path.exists(path)
            for path in (self.events_path, self.edges_path, self.chunk_map_path, self.manifest_path)
        )

    def _extract_chunks(self, chunk_rows: dict[str, dict[str, Any]]) -> list[ExtractedChunk]:
        chunk_items = list(chunk_rows.items())
        if not chunk_items:
            return []

        max_workers = max(1, int(getattr(self.config, "causal_v2_extraction_workers", 4)))
        extraction_results: list[ExtractedChunk] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._extract_single_chunk, chunk_id, row["content"]): chunk_id
                for chunk_id, row in chunk_items
            }
            for future in tqdm(
                as_completed(future_to_chunk),
                total=len(future_to_chunk),
                desc="Causal V2 extraction",
            ):
                extraction_results.append(future.result())
        extraction_results.sort(key=lambda item: item.chunk_id)
        return extraction_results

    def _extract_single_chunk(self, chunk_id: str, content: str) -> ExtractedChunk:
        title, _, body = content.partition("\n")
        passage = body.strip() or title.strip()
        retry_attempts = max(1, int(getattr(self.config, "causal_v2_extraction_retry_attempts", 2)))
        max_tokens = int(getattr(self.config, "causal_v2_extraction_max_tokens", 768))
        last_error = None
        last_raw = ""

        for _ in range(retry_attempts):
            messages = self._build_extraction_messages(title=title.strip(), passage=passage)
            raw_response = ""
            try:
                infer_result = self.hipporag.llm_model.infer(
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                )
                if isinstance(infer_result, tuple) and len(infer_result) == 3:
                    raw_response, metadata, _ = infer_result
                else:
                    raw_response, metadata = infer_result
                if isinstance(metadata, dict) and metadata.get("finish_reason") == "length":
                    raw_response = fix_broken_generated_json(raw_response)
                parsed = _safe_json_loads(raw_response)
                events, edges = self._validate_extraction_payload(parsed, graph_mode=self._graph_mode())
                return ExtractedChunk(
                    chunk_id=chunk_id,
                    title=title.strip(),
                    content=content,
                    events=events,
                    edges=edges,
                    raw_response_preview=str(raw_response)[:400],
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_error = str(exc)
                last_raw = str(raw_response)[:400]

        logger.warning("Causal V2 extraction failed for chunk %s: %s", chunk_id, last_error)
        return ExtractedChunk(
            chunk_id=chunk_id,
            title=title.strip(),
            content=content,
            events=[],
            edges=[],
            raw_response_preview=last_raw,
            parse_error=last_error,
        )

    def _build_extraction_messages(self, title: str, passage: str) -> list[dict[str, str]]:
        if self._graph_mode() == "general":
            schema_description = (
                "Return one JSON object with keys `entities` and `relations`.\n"
                "`entities` must be a list of objects with fields: `id`, `text`, `type`, `quote`.\n"
                "`relations` must be a list of objects with fields: `source`, `target`, `type`, `confidence`, `quote`.\n"
                "`type` must be one of exactly: director_of, born_on, died_on, citizen_of, born_in, parent_of, spouse_of, related_to.\n"
                "Use these canonical directions: director_of PERSON->WORK, born_on PERSON->DATE, died_on PERSON->DATE, "
                "citizen_of PERSON->COUNTRY, born_in PERSON->PLACE, parent_of PARENT->CHILD, spouse_of PERSON->PERSON.\n"
                "Use related_to only when the passage directly supports a relation but none of the typed relations above fits.\n"
                "Only extract relations directly supported by the passage. Every entity and every relation must include a short verbatim quote from the passage.\n"
                "If no useful relation is present, return {\"entities\": [], \"relations\": []}."
            )
            example_output = (
                "{\n"
                "  \"entities\": [\n"
                "    {\"id\": \"n1\", \"text\": \"John Wallop\", \"type\": \"person\", \"quote\": \"John Wallop, 2nd Earl of Portsmouth\"},\n"
                "    {\"id\": \"n2\", \"text\": \"Coulson Wallop\", \"type\": \"person\", \"quote\": \"Coulson Wallop\"}\n"
                "  ],\n"
                "  \"relations\": [\n"
                "    {\"source\": \"n1\", \"target\": \"n2\", \"type\": \"parent_of\", \"confidence\": 0.9, \"quote\": \"Coulson Wallop was the eldest son of John Wallop\"}\n"
                "  ]\n"
                "}"
            )
            system_text = (
                "You extract a compact relation graph from a single passage. "
                "You must follow the JSON schema exactly and return only JSON."
            )
        else:
            schema_description = (
                "Return one JSON object with keys `events` and `causal_edges`.\n"
                "`events` must be a list of objects with fields: "
                "`event_id`, `text`, `quote`.\n"
                "`causal_edges` must be a list of objects with fields: "
                "`source_event_id`, `target_event_id`, `relation_type`, `confidence`, `quote`.\n"
                "`relation_type` must be one of: cause, enable, prevent.\n"
                "Only extract relations directly supported by the passage. "
                "Every event and every edge must include a short verbatim quote from the passage.\n"
                "If there is no causal content, return {\"events\": [], \"causal_edges\": []}."
            )
            example_output = (
                "{\n"
                "  \"events\": [\n"
                "    {\"event_id\": \"e1\", \"text\": \"rates were cut\", \"quote\": \"the central bank cut interest rates\"},\n"
                "    {\"event_id\": \"e2\", \"text\": \"borrowing increased\", \"quote\": \"borrowing increased after the cut\"}\n"
                "  ],\n"
                "  \"causal_edges\": [\n"
                "    {\"source_event_id\": \"e1\", \"target_event_id\": \"e2\", \"relation_type\": \"cause\", \"confidence\": 0.9, \"quote\": \"borrowing increased after the cut\"}\n"
                "  ]\n"
                "}"
            )
            system_text = (
                "You extract causal event graphs from a single passage. "
                "You must follow the JSON schema exactly and return only JSON."
            )
        return [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": (
                    f"Title: {title}\n\n"
                    f"Passage:\n{passage}\n\n"
                    f"{schema_description}\n\n"
                    "Example output:\n"
                    f"{example_output}"
                ),
            },
        ]

    @staticmethod
    def _validate_extraction_payload(
        payload: dict[str, Any],
        graph_mode: str = "causal",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if graph_mode == "general":
            raw_events = payload.get("entities", payload.get("events", []))
            raw_edges = payload.get("relations", payload.get("causal_edges", []))
        else:
            raw_events = payload.get("events", [])
            raw_edges = payload.get("causal_edges", [])
        if not isinstance(raw_events, list) or not isinstance(raw_edges, list):
            raise ValueError("Extraction payload must contain list fields for nodes and edges.")

        events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            event_id = str(raw_event.get("event_id") or raw_event.get("id") or "").strip()
            event_text = str(raw_event.get("text", "")).strip()
            quote = str(raw_event.get("quote", "")).strip()
            if not event_id or not event_text or not quote or event_id in seen_ids:
                continue
            events.append({
                "event_id": event_id,
                "text": event_text,
                "quote": quote,
                "entity_type": str(raw_event.get("type", "")).strip() if graph_mode == "general" else "",
            })
            seen_ids.add(event_id)

        valid_event_ids = {event["event_id"] for event in events}
        edges: list[dict[str, Any]] = []
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            source_event_id = str(raw_edge.get("source_event_id") or raw_edge.get("source") or "").strip()
            target_event_id = str(raw_edge.get("target_event_id") or raw_edge.get("target") or "").strip()
            relation_key = raw_edge.get("type") if graph_mode == "general" else raw_edge.get("relation_type")
            relation_type = _canonicalize_relation(relation_key, graph_mode=graph_mode)
            quote = str(raw_edge.get("quote", "")).strip()
            if (
                not source_event_id
                or not target_event_id
                or not quote
                or relation_type is None
                or source_event_id == target_event_id
                or source_event_id not in valid_event_ids
                or target_event_id not in valid_event_ids
            ):
                continue
            edges.append({
                "source_event_id": source_event_id,
                "target_event_id": target_event_id,
                "relation_type": relation_type,
                "confidence": _clip_confidence(raw_edge.get("confidence"), default=0.75),
                "quote": quote,
            })
        return events, edges

    def _build_index(self, chunk_rows: dict[str, dict[str, Any]], extracted_chunks: list[ExtractedChunk]) -> None:
        surface_events: list[dict[str, Any]] = []
        surface_edges: list[dict[str, Any]] = []
        chunk_to_event_ids: dict[str, set[str]] = defaultdict(set)
        relation_counter: Counter[str] = Counter()
        num_chunks_with_events = 0
        num_chunks_with_edges = 0
        num_parse_failures = 0
        parse_failure_examples: list[dict[str, Any]] = []

        for extracted in extracted_chunks:
            if extracted.parse_error:
                num_parse_failures += 1
                if len(parse_failure_examples) < 5:
                    parse_failure_examples.append({
                        "chunk_id": extracted.chunk_id,
                        "title": extracted.title,
                        "parse_error": extracted.parse_error,
                        "raw_response_preview": extracted.raw_response_preview,
                    })
            if extracted.events:
                num_chunks_with_events += 1
            if extracted.edges:
                num_chunks_with_edges += 1
            local_event_id_to_temp_id: dict[str, str] = {}
            for raw_event in extracted.events:
                temp_event_id = f"{extracted.chunk_id}:{raw_event['event_id']}"
                local_event_id_to_temp_id[raw_event["event_id"]] = temp_event_id
                surface_events.append({
                    "temp_event_id": temp_event_id,
                    "text": raw_event["text"],
                    "quote": raw_event["quote"],
                    "entity_type": raw_event.get("entity_type", ""),
                    "chunk_id": extracted.chunk_id,
                    "doc_title": extracted.title,
                })
            for raw_edge in extracted.edges:
                source_temp_id = local_event_id_to_temp_id.get(raw_edge["source_event_id"])
                target_temp_id = local_event_id_to_temp_id.get(raw_edge["target_event_id"])
                if source_temp_id is None or target_temp_id is None:
                    continue
                relation_counter[str(raw_edge["relation_type"])] += 1
                surface_edges.append({
                    "source_temp_event_id": source_temp_id,
                    "target_temp_event_id": target_temp_id,
                    "relation_type": raw_edge["relation_type"],
                    "confidence": raw_edge["confidence"],
                    "quote": raw_edge["quote"],
                    "chunk_id": extracted.chunk_id,
                    "doc_title": extracted.title,
                })

        temp_to_event_id, event_nodes = self._resolve_events(surface_events)
        edge_groups: dict[tuple[str, str, str], dict[str, Any]] = {}

        for surface_edge in surface_edges:
            source_event_id = temp_to_event_id.get(surface_edge["source_temp_event_id"])
            target_event_id = temp_to_event_id.get(surface_edge["target_temp_event_id"])
            if source_event_id is None or target_event_id is None or source_event_id == target_event_id:
                continue
            edge_key = (source_event_id, target_event_id, surface_edge["relation_type"])
            edge = edge_groups.setdefault(
                edge_key,
                {
                    "source_event_id": source_event_id,
                    "target_event_id": target_event_id,
                    "relation_type": surface_edge["relation_type"],
                    "confidence": 0.0,
                    "quotes": [],
                    "chunk_ids": set(),
                    "doc_titles": set(),
                },
            )
            edge["confidence"] = max(edge["confidence"], float(surface_edge["confidence"]))
            edge["chunk_ids"].add(surface_edge["chunk_id"])
            edge["doc_titles"].add(surface_edge["doc_title"])
            if surface_edge["quote"] and surface_edge["quote"] not in edge["quotes"]:
                edge["quotes"].append(surface_edge["quote"])

        for surface_event in surface_events:
            canonical_event_id = temp_to_event_id.get(surface_event["temp_event_id"])
            if canonical_event_id is None:
                continue
            chunk_to_event_ids[surface_event["chunk_id"]].add(canonical_event_id)

        self.event_nodes = {
            event_id: {
                **event_node,
                    "surface_forms": sorted(set(event_node["surface_forms"])),
                    "quotes": sorted(set(event_node["quotes"])),
                    "entity_types": sorted(set(event_node.get("entity_types", []))),
                    "chunk_ids": sorted(set(event_node["chunk_ids"])),
                    "doc_titles": sorted(set(event_node["doc_titles"])),
                }
            for event_id, event_node in event_nodes.items()
        }
        self.edges = {}
        for source_event_id, target_event_id, relation_type in sorted(edge_groups.keys()):
            edge = edge_groups[(source_event_id, target_event_id, relation_type)]
            if edge["confidence"] < float(getattr(self.config, "causal_v2_min_edge_confidence", 0.7)):
                continue
            edge_id = compute_mdhash_id(
                f"{source_event_id}:{relation_type}:{target_event_id}",
                prefix="causal_edge_v2-",
            )
            self.edges[edge_id] = {
                "edge_id": edge_id,
                "source_event_id": source_event_id,
                "target_event_id": target_event_id,
                "relation_type": relation_type,
                "confidence": float(edge["confidence"]),
                "quotes": edge["quotes"][:3],
                "chunk_ids": sorted(edge["chunk_ids"]),
                "doc_titles": sorted(edge["doc_titles"]),
            }
        self.chunk_to_event_ids = {
            chunk_id: sorted(event_ids)
            for chunk_id, event_ids in chunk_to_event_ids.items()
        }

        final_relation_counter: Counter[str] = Counter(
            edge["relation_type"] for edge in self.edges.values()
        )
        canonical_event_texts = [event_node["canonical_text"] for event_node in self.event_nodes.values()]
        if canonical_event_texts:
            self.event_store.insert_strings(canonical_event_texts)

        with open(self.events_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"events": list(self.event_nodes.values())},
                handle,
                indent=2,
                ensure_ascii=False,
            )
        with open(self.edges_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"edges": list(self.edges.values())},
                handle,
                indent=2,
                ensure_ascii=False,
            )
        with open(self.chunk_map_path, "w", encoding="utf-8") as handle:
            json.dump(self.chunk_to_event_ids, handle, indent=2, ensure_ascii=False)
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "chunk_ids": sorted(chunk_rows.keys()),
                    "num_chunks": len(chunk_rows),
                    "num_chunks_with_events": int(num_chunks_with_events),
                    "num_chunks_with_edges": int(num_chunks_with_edges),
                    "num_parse_failures": int(num_parse_failures),
                    "parse_failure_examples": parse_failure_examples,
                    "num_surface_events": len(surface_events),
                    "num_surface_edges": len(surface_edges),
                    "num_events": len(self.event_nodes),
                    "num_edges": len(self.edges),
                    "surface_relation_counts": dict(sorted(relation_counter.items())),
                    "final_relation_counts": dict(sorted(final_relation_counter.items())),
                    "graph_mode": self._graph_mode(),
                    "version": self._manifest_version(),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def read_manifest(self) -> dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        return data

    def _resolve_events(
        self,
        surface_events: list[dict[str, Any]],
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        if not surface_events:
            return {}, {}

        exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for surface_event in surface_events:
            normalized_text = _normalize_text(surface_event["text"])
            if not normalized_text:
                continue
            exact_groups[normalized_text].append(surface_event)

        exact_group_ids = [f"group-{idx}" for idx, _ in enumerate(exact_groups.values())]
        group_id_to_members = dict(zip(exact_group_ids, exact_groups.values()))
        group_id_to_surface_forms = {
            group_id: [member["text"] for member in members]
            for group_id, members in group_id_to_members.items()
        }

        if exact_group_ids:
            group_texts = [Counter(group_id_to_surface_forms[group_id]).most_common(1)[0][0] for group_id in exact_group_ids]
            group_embeddings = np.asarray(
                self.hipporag.embedding_model.batch_encode(group_texts, norm=True),
                dtype=float,
            )
            uf = _UnionFind(exact_group_ids)
            if len(exact_group_ids) > 1:
                knn = min(len(exact_group_ids), 10)
                neighbors = np.dot(group_embeddings, group_embeddings.T)
                similarity_threshold = float(getattr(self.config, "causal_er_similarity_threshold", 0.92))
                text_threshold = float(getattr(self.config, "causal_er_text_threshold", 0.55))
                for row_idx, group_id in enumerate(exact_group_ids):
                    nearest = np.argsort(neighbors[row_idx])[::-1][1:knn]
                    for col_idx in nearest.tolist():
                        other_group_id = exact_group_ids[col_idx]
                        if neighbors[row_idx, col_idx] < similarity_threshold:
                            continue
                        if _text_similarity(group_texts[row_idx], group_texts[col_idx]) < text_threshold:
                            continue
                        uf.union(group_id, other_group_id)
            cluster_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for group_id, members in group_id_to_members.items():
                cluster_members[uf.find(group_id)].extend(members)
        else:
            cluster_members = {}

        temp_to_event_id: dict[str, str] = {}
        event_nodes: dict[str, dict[str, Any]] = {}
        for members in cluster_members.values():
            surface_counter = Counter(member["text"] for member in members if member["text"])
            canonical_text = surface_counter.most_common(1)[0][0]
            event_id = compute_mdhash_id(canonical_text, prefix="causal_event_v2-")
            event_nodes[event_id] = {
                "event_id": event_id,
                "canonical_text": canonical_text,
                "surface_forms": [member["text"] for member in members if member["text"]],
                "quotes": [member["quote"] for member in members if member["quote"]],
                "entity_types": [member["entity_type"] for member in members if member.get("entity_type")],
                "chunk_ids": [member["chunk_id"] for member in members],
                "doc_titles": [member["doc_title"] for member in members],
            }
            for member in members:
                temp_to_event_id[member["temp_event_id"]] = event_id
        return temp_to_event_id, event_nodes

    def load(self) -> None:
        if not self._has_complete_index():
            self.event_nodes = {}
            self.edges = {}
            self.chunk_to_event_ids = {}
            self.adjacency_out = defaultdict(list)
            self.adjacency_in = defaultdict(list)
            self.event_ids = []
            self.event_embeddings = np.zeros((0, 0), dtype=float)
            self.loaded = True
            return

        with open(self.events_path, "r", encoding="utf-8") as handle:
            self.event_nodes = {
                event["event_id"]: event
                for event in json.load(handle).get("events", [])
            }
        with open(self.edges_path, "r", encoding="utf-8") as handle:
            self.edges = {
                edge["edge_id"]: edge
                for edge in json.load(handle).get("edges", [])
            }
        with open(self.chunk_map_path, "r", encoding="utf-8") as handle:
            self.chunk_to_event_ids = {
                str(chunk_id): [str(event_id) for event_id in event_ids]
                for chunk_id, event_ids in json.load(handle).items()
            }

        self.adjacency_out = defaultdict(list)
        self.adjacency_in = defaultdict(list)
        for edge in self.edges.values():
            self.adjacency_out[edge["source_event_id"]].append(edge)
            self.adjacency_in[edge["target_event_id"]].append(edge)

        self.event_ids = sorted(self.event_nodes.keys())
        canonical_texts = [self.event_nodes[event_id]["canonical_text"] for event_id in self.event_ids]
        if canonical_texts:
            self.event_store.insert_strings(canonical_texts)
            self.event_embeddings = np.asarray(self.event_store.get_embeddings(self.event_ids), dtype=float)
        else:
            self.event_embeddings = np.zeros((0, 0), dtype=float)
        self.loaded = True

    def route_query(self, query: str, query_embedding: np.ndarray) -> dict[str, Any]:
        route_info = self.router.route(query_embedding=query_embedding)
        route_info["query"] = query
        return route_info

    def retrieve_subgraph(
        self,
        query: str,
        query_embedding: np.ndarray,
        route_info: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.loaded:
            self.load()

        probe_mode = str(getattr(self.config, "causal_v2_probe_mode", "router")).lower()
        use_general_graph = self._graph_mode() == "general"
        probe_forced = (probe_mode == "always" or use_general_graph) and not route_info.get("is_causal", False)
        probe_attempted = bool(self.event_ids) and (route_info.get("is_causal", False) or probe_forced or use_general_graph)
        if use_general_graph:
            probe_route_label = "general"
        else:
            probe_route_label = (
                route_info.get("label")
                if route_info.get("is_causal", False)
                else route_info.get("best_causal_label", "effect")
            )

        if not probe_attempted:
            return {
                "probe_attempted": False,
                "probe_forced": False,
                "probe_route_label": None,
                "seed_event_ids": [],
                "seed_event_texts": [],
                "query_entities": [],
                "chains": [],
                "selected_chains": [],
                "serialized_contexts": [],
                "causal_context_doc_ids": [],
                "chain_selection_trace": {
                    "mode": str(getattr(self.config, "causal_context_injection_mode", "all")).lower(),
                    "candidate_chain_count": 0,
                    "entity_overlap_chain_count": 0,
                    "selected_chain_count": 0,
                    "selected_chain_scores": [],
                },
            }

        event_scores = np.dot(self.event_embeddings, np.asarray(query_embedding, dtype=float).reshape(-1).T)
        ranked_event_indices = np.argsort(event_scores)[::-1]
        seed_top_k = max(1, int(getattr(self.config, "causal_event_top_k", 8)))
        seed_event_ids = [self.event_ids[idx] for idx in ranked_event_indices[:seed_top_k].tolist()]
        seed_event_texts = [self.event_nodes[event_id]["canonical_text"] for event_id in seed_event_ids]
        seed_scores = {
            event_id: float(event_scores[idx])
            for idx, event_id in zip(ranked_event_indices[:seed_top_k].tolist(), seed_event_ids)
        }

        chains = self._collect_chains(seed_event_ids, seed_scores, probe_route_label)
        selected_chains, query_entities, selection_trace = self._select_chains_for_injection(
            query=query,
            chains=chains,
        )
        serialized_contexts = [chain["serialized"] for chain in selected_chains]
        causal_context_doc_ids: list[str] = []
        for chain in selected_chains:
            for chunk_id in chain["chunk_ids"]:
                if chunk_id not in causal_context_doc_ids:
                    causal_context_doc_ids.append(chunk_id)

        return {
            "probe_attempted": True,
            "probe_forced": probe_forced,
            "probe_route_label": probe_route_label,
            "seed_event_ids": seed_event_ids,
            "seed_event_texts": seed_event_texts,
            "query_entities": query_entities,
            "chains": chains,
            "selected_chains": selected_chains,
            "serialized_contexts": serialized_contexts,
            "causal_context_doc_ids": causal_context_doc_ids,
            "chain_selection_trace": selection_trace,
        }

    def _select_chains_for_injection(
        self,
        query: str,
        chains: Sequence[dict[str, Any]],
    ) -> Tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
        mode = str(getattr(self.config, "causal_context_injection_mode", "all")).lower()
        chain_top_k = max(1, int(getattr(self.config, "causal_chain_top_k", 6)))
        min_chain_score = float(getattr(self.config, "causal_context_min_chain_score", 0.1))
        query_entities = self._extract_query_entities(query)

        if mode != "selective":
            selected_chains = list(chains[:chain_top_k])
            return selected_chains, query_entities, {
                "mode": mode,
                "candidate_chain_count": int(len(chains)),
                "entity_overlap_chain_count": int(len(chains)),
                "selected_chain_count": int(len(selected_chains)),
                "selected_chain_scores": [float(chain["score"]) for chain in selected_chains],
            }

        overlap_chains = [
            chain for chain in chains
            if self._chain_overlaps_query_entities(query_entities, chain.get("serialized", ""))
        ]
        selected_chains: list[dict[str, Any]] = []
        if overlap_chains and float(overlap_chains[0].get("score", 0.0)) >= min_chain_score:
            selected_chains = [overlap_chains[0]]

        return selected_chains, query_entities, {
            "mode": mode,
            "candidate_chain_count": int(len(chains)),
            "entity_overlap_chain_count": int(len(overlap_chains)),
            "selected_chain_count": int(len(selected_chains)),
            "selected_chain_scores": [float(chain["score"]) for chain in selected_chains],
            "min_chain_score": min_chain_score,
        }

    def _extract_query_entities(self, query: str) -> list[str]:
        raw_query = str(query or "")
        candidates: list[str] = []

        for quoted in re.findall(r"\"([^\"]+)\"", raw_query):
            cleaned = self._clean_query_entity_candidate(quoted)
            if cleaned:
                candidates.append(cleaned)

        title_case_pattern = re.compile(
            r"\b(?:[A-Z0-9][A-Za-z0-9'&./-]*)(?:\s+(?:[A-Z0-9][A-Za-z0-9'&./-]*))*"
        )
        for match in title_case_pattern.finditer(raw_query):
            cleaned = self._clean_query_entity_candidate(match.group(0))
            if cleaned:
                candidates.append(cleaned)
                if " " in cleaned:
                    for token in cleaned.split():
                        token_clean = self._clean_query_entity_candidate(token)
                        if token_clean and len(token_clean) >= 5:
                            candidates.append(token_clean)

        return _dedupe_preserve_order(candidates)

    def _clean_query_entity_candidate(self, value: str) -> Optional[str]:
        cleaned = str(value or "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"(?:'s|’s)\b", "", cleaned)
        cleaned = cleaned.strip(" ,;:.!?()[]{}\"'`")
        if not cleaned:
            return None

        normalized = _normalize_text(cleaned)
        if not normalized:
            return None
        if normalized in _QUERY_ENTITY_STOPWORDS:
            return None
        tokens = normalized.split()
        if not tokens:
            return None
        if all(token in _QUERY_ENTITY_STOPWORDS for token in tokens):
            return None
        if len(normalized) <= 2:
            return None
        return cleaned

    def _chain_overlaps_query_entities(self, query_entities: Sequence[str], chain_text: str) -> bool:
        if not query_entities:
            return False
        normalized_chain = _normalize_text(chain_text)
        for entity in query_entities:
            normalized_entity = _normalize_text(entity)
            if len(normalized_entity) <= 2:
                continue
            if normalized_entity in normalized_chain:
                return True
        return False

    def _chunk_overlaps_query_entities(self, query_entities: Sequence[str], chunk_text: str) -> bool:
        if not query_entities:
            return False
        normalized_chunk = _normalize_text(chunk_text)
        if not normalized_chunk:
            return False
        for entity in query_entities:
            normalized_entity = _normalize_text(entity)
            if len(normalized_entity) <= 2:
                continue
            if normalized_entity in normalized_chunk:
                return True
        return False

    def _collect_chains(
        self,
        seed_event_ids: Sequence[str],
        seed_scores: dict[str, float],
        route_label: str,
    ) -> list[dict[str, Any]]:
        max_hops = max(1, int(getattr(self.config, "causal_v2_max_hops", 2)))
        collected: list[dict[str, Any]] = []
        for seed_event_id in seed_event_ids:
            seed_score = float(seed_scores.get(seed_event_id, 0.0))
            frontier = [(seed_event_id, [], [seed_event_id], seed_score)]
            while frontier:
                current_event_id, path_edges, path_nodes, path_score = frontier.pop(0)
                if len(path_edges) >= max_hops:
                    continue
                neighbors = self._neighbors_for_route(current_event_id, route_label)
                for edge, next_event_id in neighbors:
                    if route_label == "cause":
                        new_edges = [edge] + path_edges
                        new_nodes = [next_event_id] + path_nodes
                    else:
                        new_edges = path_edges + [edge]
                        new_nodes = path_nodes + [next_event_id]
                    if next_event_id in path_nodes:
                        continue
                    new_score = path_score * max(float(edge["confidence"]), 1e-3) * (1.0 / (len(new_edges) + 0.25))
                    chain = self._serialize_chain(new_nodes, new_edges, new_score, seed_event_id)
                    if chain is not None:
                        collected.append(chain)
                    frontier.append((next_event_id, new_edges, new_nodes, new_score))

        collected.sort(key=lambda item: item["score"], reverse=True)
        deduped: list[dict[str, Any]] = []
        seen_signatures: set[tuple[str, ...]] = set()
        for chain in collected:
            signature = tuple(chain["edge_ids"])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped.append(chain)
        return deduped

    def _neighbors_for_route(self, event_id: str, route_label: str) -> list[tuple[dict[str, Any], str]]:
        if route_label == "general":
            neighbors: list[tuple[dict[str, Any], str]] = []
            seen_pairs: set[tuple[str, str]] = set()
            for edge in self.adjacency_out.get(event_id, []):
                pair = (edge["edge_id"], edge["target_event_id"])
                if pair in seen_pairs:
                    continue
                neighbors.append((edge, edge["target_event_id"]))
                seen_pairs.add(pair)
            for edge in self.adjacency_in.get(event_id, []):
                pair = (edge["edge_id"], edge["source_event_id"])
                if pair in seen_pairs:
                    continue
                neighbors.append((edge, edge["source_event_id"]))
                seen_pairs.add(pair)
            return neighbors
        if route_label == "cause":
            return [(edge, edge["source_event_id"]) for edge in self.adjacency_in.get(event_id, [])]
        if route_label == "effect":
            return [(edge, edge["target_event_id"]) for edge in self.adjacency_out.get(event_id, [])]
        if route_label == "prevention":
            return [
                (edge, edge["target_event_id"]) for edge in self.adjacency_out.get(event_id, [])
                if edge["relation_type"] == "prevent"
            ] + [
                (edge, edge["source_event_id"]) for edge in self.adjacency_in.get(event_id, [])
                if edge["relation_type"] == "prevent"
            ]
        return []

    def _serialize_chain(
        self,
        path_nodes: Sequence[str],
        path_edges: Sequence[dict[str, Any]],
        score: float,
        seed_event_id: str,
    ) -> Optional[dict[str, Any]]:
        if not path_edges:
            return None

        chunk_ids: list[str] = []
        evidence_parts: list[str] = []
        for edge in path_edges:
            source_text = self.event_nodes[edge["source_event_id"]]["canonical_text"]
            target_text = self.event_nodes[edge["target_event_id"]]["canonical_text"]
            if self._graph_mode() == "general":
                relation_text = _GENERAL_RELATION_TO_TEXT.get(edge["relation_type"], edge["relation_type"])
            else:
                relation_text = _CAUSAL_RELATION_TO_TEXT.get(edge["relation_type"], edge["relation_type"])
            doc_title = edge["doc_titles"][0] if edge.get("doc_titles") else "Unknown title"
            quote = edge["quotes"][0] if edge.get("quotes") else ""
            if quote:
                evidence_parts.append(
                    f"[{doc_title}] \"{quote}\" shows that `{source_text}` {relation_text} `{target_text}`."
                )
            else:
                evidence_parts.append(
                    f"[{doc_title}] `{source_text}` {relation_text} `{target_text}`."
                )
            for chunk_id in edge.get("chunk_ids", []):
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)

        serialized = " ".join(evidence_parts)
        return {
            "seed_event_id": seed_event_id,
            "path_nodes": list(path_nodes),
            "edge_ids": [edge["edge_id"] for edge in path_edges],
            "chunk_ids": chunk_ids,
            "score": float(score),
            "serialized": serialized,
        }

    def propose_candidate_doc_injections(
        self,
        dense_sorted_doc_ids: np.ndarray,
        dense_sorted_doc_scores: np.ndarray,
        query_entities: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "graph_mode": self._graph_mode(),
            "seed_top_n": 0,
            "max_docs": max(0, int(getattr(self.config, "causal_v2_candidate_injection_max_docs", 5))),
            "hops": max(1, int(getattr(self.config, "causal_v2_candidate_injection_hops", 1))),
            "used_query_entity_overlap_gate": bool(getattr(self.config, "causal_v2_candidate_injection_require_query_entity_overlap", True)),
            "candidate_seed_chunk_count": 0,
            "query_entity_seed_chunk_count": 0,
            "seed_node_count": 0,
            "proposed_doc_count": 0,
            "injected_doc_ids": [],
            "injected_chunk_ids": [],
            "injected_doc_scores": {},
            "noop_reason": None,
        }
        if self._graph_mode() != "general":
            trace["noop_reason"] = "graph_mode_not_general"
            return trace
        if not self.loaded:
            self.load()
        if len(dense_sorted_doc_ids) == 0 or not self.event_ids:
            trace["noop_reason"] = "empty_dense_or_graph"
            return trace

        seed_top_n = max(0, min(int(getattr(self.config, "causal_v2_candidate_injection_top_n", 20)), len(dense_sorted_doc_ids)))
        trace["seed_top_n"] = seed_top_n
        if seed_top_n == 0:
            trace["noop_reason"] = "seed_top_n_zero"
            return trace

        dense_doc_id_list = [int(doc_id) for doc_id in dense_sorted_doc_ids.tolist()]
        dense_score_by_doc_id = {
            int(doc_id): float(score)
            for doc_id, score in zip(dense_sorted_doc_ids.tolist(), dense_sorted_doc_scores.tolist())
        }
        seed_doc_ids = dense_doc_id_list[:seed_top_n]
        seed_chunk_ids = [self.hipporag.passage_node_keys[doc_id] for doc_id in seed_doc_ids]
        trace["candidate_seed_chunk_count"] = len(seed_chunk_ids)

        require_query_overlap = bool(getattr(self.config, "causal_v2_candidate_injection_require_query_entity_overlap", True))
        normalized_query_entities = [str(entity) for entity in (query_entities or []) if str(entity).strip()]
        seed_doc_chunk_pairs = list(zip(seed_doc_ids, seed_chunk_ids))
        if require_query_overlap:
            if not normalized_query_entities:
                trace["noop_reason"] = "no_query_entities"
                return trace
            filtered_seed_doc_chunk_pairs: list[tuple[int, str]] = []
            for doc_id, chunk_id in seed_doc_chunk_pairs:
                chunk_row = self.hipporag.chunk_embedding_store.get_row(str(chunk_id))
                chunk_text = str(chunk_row.get("content", ""))
                if self._chunk_overlaps_query_entities(normalized_query_entities, chunk_text):
                    filtered_seed_doc_chunk_pairs.append((int(doc_id), str(chunk_id)))
            trace["query_entity_seed_chunk_count"] = len(filtered_seed_doc_chunk_pairs)
            if not filtered_seed_doc_chunk_pairs:
                trace["noop_reason"] = "no_query_overlap_seed_docs"
                return trace
            seed_doc_chunk_pairs = filtered_seed_doc_chunk_pairs
        else:
            trace["query_entity_seed_chunk_count"] = len(seed_doc_chunk_pairs)

        seed_chunk_id_set = {chunk_id for _, chunk_id in seed_doc_chunk_pairs}

        frontier: list[tuple[str, int, float]] = []
        seen_state: set[tuple[str, int]] = set()
        for doc_id, chunk_id in seed_doc_chunk_pairs:
            seed_weight = max(dense_score_by_doc_id.get(doc_id, 0.0), 1e-3)
            for event_id in self.chunk_to_event_ids.get(str(chunk_id), []):
                state = (event_id, 0)
                if state in seen_state:
                    continue
                seen_state.add(state)
                frontier.append((event_id, 0, seed_weight))
        trace["seed_node_count"] = len(frontier)
        if not frontier:
            trace["noop_reason"] = "no_seed_nodes"
            return trace

        proposed_scores: dict[int, float] = {}
        max_hops = trace["hops"]
        while frontier:
            current_event_id, depth, current_score = frontier.pop(0)
            if depth >= max_hops:
                continue
            for edge, next_event_id in self._neighbors_for_route(current_event_id, "general"):
                next_score = current_score * max(float(edge.get("confidence", 0.0)), 1e-3) * (1.0 / (depth + 1.0))
                next_node = self.event_nodes.get(next_event_id, {})
                for chunk_id in next_node.get("chunk_ids", []):
                    chunk_id = str(chunk_id)
                    if chunk_id in seed_chunk_id_set:
                        continue
                    doc_idx = self.hipporag.passage_node_key_to_doc_idx.get(chunk_id)
                    if doc_idx is None:
                        continue
                    proposed_scores[doc_idx] = max(proposed_scores.get(doc_idx, 0.0), float(next_score))
                next_state = (next_event_id, depth + 1)
                if next_state not in seen_state:
                    seen_state.add(next_state)
                    frontier.append((next_event_id, depth + 1, next_score))

        trace["proposed_doc_count"] = len(proposed_scores)
        if not proposed_scores:
            trace["noop_reason"] = "no_proposed_docs"
            return trace

        ranked_injected = sorted(proposed_scores.items(), key=lambda item: item[1], reverse=True)[: trace["max_docs"]]
        trace["injected_doc_ids"] = [int(doc_id) for doc_id, _ in ranked_injected]
        trace["injected_chunk_ids"] = [str(self.hipporag.passage_node_keys[int(doc_id)]) for doc_id, _ in ranked_injected]
        trace["injected_doc_scores"] = {int(doc_id): float(score) for doc_id, score in ranked_injected}
        return trace
