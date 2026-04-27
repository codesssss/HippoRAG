#!/usr/bin/env python3
"""Q-BindCert Phase-0 blocking sanity scaffold.

This script is intentionally not a full Q-BindCert implementation.  It runs the
Phase-0 gates that decide whether a full implementation is justified.

Important status semantics:
- `PASS` / `FAIL`: an oracle-controlled automatic test can decide the gate.
- `NEEDS_MANUAL_AUDIT`: automatic proxy evidence is present, but the agreed
  gate requires human inspection before Phase 1 can proceed.
- `NOT_RUN`: the required generation/check was not executed.
- `FAIL_PROXY`: a silver/proxy check failed; this is sufficient to stop unless
  a manual audit overturns it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Sequence
import urllib.request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import read_json, write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import normalize_answer  # noqa: E402


PREDICATE_BUCKETS = (
    "family_relation",
    "date_of_birth",
    "date_of_death",
    "place_of_birth",
    "place_of_death",
    "located_in",
    "nationality",
    "occupation",
    "creator",
    "directed_by",
    "written_by",
    "performed_by",
    "produced_by",
    "cast_member",
    "spouse_of",
    "member_of",
    "part_of",
    "founded_by",
    "owned_by",
    "won",
    "nominated_for",
    "publication_date",
    "comparison",
    "attribute",
    "other",
)

TYPE_LABELS = ("Person", "Place", "Work", "Organization", "Date", "Number", "Other")

DATE_RE = re.compile(r"\b(?:\d{1,2}\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{3,4}|\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{3,4}\b|\b(?:1[5-9]\d{2}|20\d{2})\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"^[\d,.\s%-]+$")
CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,5}\b")


BUCKET_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("date_of_death", ("date of death", "death date", "died", "die", "deceased")),
    ("date_of_birth", ("date of birth", "birth date", "born")),
    ("place_of_birth", ("place of birth", "birthplace", "born in")),
    ("place_of_death", ("place of death", "death place", "died in")),
    ("family_relation", ("mother", "father", "parent", "son", "daughter", "child", "sibling", "brother", "sister")),
    ("spouse_of", ("spouse", "wife", "husband", "married")),
    ("directed_by", ("director", "directed", "directed by")),
    ("written_by", ("writer", "written", "author", "screenplay")),
    ("performed_by", ("performer", "performed", "performed by", "singer", "artist", "band", "actor", "actress")),
    ("produced_by", ("producer", "produced")),
    ("cast_member", ("starring", "cast", "played by")),
    ("creator", ("creator", "created", "founded by", "designer", "developer")),
    ("founded_by", ("founder", "founded")),
    ("owned_by", ("owner", "owned")),
    ("member_of", ("member of", "part of", "team", "club")),
    ("part_of", ("part of", "subsidiary", "division")),
    ("nationality", ("nationality", "country of citizenship", "citizen")),
    ("located_in", ("located", "location", "country", "city", "place", "headquarters")),
    ("occupation", ("occupation", "profession", "job")),
    ("won", ("award received", "won", "winner", "award")),
    ("nominated_for", ("nominated", "nomination")),
    ("publication_date", ("publication", "released", "release date", "published")),
    ("comparison", ("larger", "smaller", "older", "younger", "earlier", "later", "more", "less")),
)

BUCKET_ALIASES = {
    "father": "family_relation",
    "mother": "family_relation",
    "parent": "family_relation",
    "child": "family_relation",
    "child_of": "family_relation",
    "son": "family_relation",
    "daughter": "family_relation",
    "sibling": "family_relation",
    "brother": "family_relation",
    "sister": "family_relation",
    "performer": "performed_by",
    "release_date": "publication_date",
    "date_released": "publication_date",
    "country_of_citizenship": "nationality",
    "country_of_origin": "located_in",
    "citizenship": "nationality",
}


def norm_text(value: Any) -> str:
    return normalize_answer(str(value or ""))


def title_key(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def entity_key(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_norm(haystack: Any, needle: Any) -> bool:
    h = norm_text(haystack)
    n = norm_text(needle)
    return bool(h and n and (n in h or h in n))


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / float(len(vals)) if vals else 0.0


def sha1_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def strip_think_blocks(text: str) -> str:
    raw = str(text or "")
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw)
    raw = raw.replace("```json", "```")
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return raw.strip()


def parse_json_payload(text: str) -> Any | None:
    raw = strip_think_blocks(text)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start_candidates = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(raw), start, -1):
        snippet = raw[start:end].strip()
        if not snippet:
            continue
        try:
            return json.loads(snippet)
        except Exception:
            continue
    return None


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def load_cache_rows(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return load_jsonl(p, limit=limit)
    payload = read_json(p)
    rows = list(payload.get("records") if isinstance(payload, dict) else payload)
    return rows[: int(limit)] if int(limit) > 0 else rows


def bucketize_relation(value: Any) -> str:
    text = norm_text(value)
    key = text.replace(" ", "_")
    if key in BUCKET_ALIASES:
        return BUCKET_ALIASES[key]
    for bucket, patterns in BUCKET_PATTERNS:
        if any(norm_text(pattern) in text for pattern in patterns):
            return bucket
    return "attribute" if text else "other"


def canonical_bucket(value: Any) -> str:
    raw = str(value or "")
    if raw in PREDICATE_BUCKETS:
        return raw
    return bucketize_relation(raw)


def infer_answer_type(answer: Any, question: str = "") -> str:
    raw = str(answer or "").strip()
    q = norm_text(question)
    if not raw:
        return "Other"
    if norm_text(raw) in {"yes", "no"}:
        return "Boolean"
    if DATE_RE.search(raw):
        return "Date"
    if NUMBER_RE.fullmatch(raw) and re.search(r"\d", raw):
        return "Number"
    if q.startswith("where") or "what city" in q or "which city" in q or "what country" in q:
        return "Place"
    if q.startswith("when") or "what year" in q:
        return "Date"
    if q.startswith("how many") or "number of" in q:
        return "Number"
    if q.startswith("who"):
        return "Person"
    if any(token in q for token in ("film", "song", "album", "book", "novel", "work")):
        return "Work"
    return "Other"


def infer_entity_type(text: Any, question: str = "") -> str:
    raw = str(text or "").strip()
    if DATE_RE.search(raw):
        return "Date"
    if NUMBER_RE.fullmatch(raw) and re.search(r"\d", raw):
        return "Number"
    return infer_answer_type(raw, question) if norm_text(raw) == norm_text(question) else "Other"


def candidate_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    docs = []
    for idx, cand in enumerate(row.get("candidates") or []):
        docs.append(
            {
                "doc_id": cand.get("doc_id", idx),
                "rank": int(cand.get("rank") or idx + 1),
                "title": str(cand.get("title") or ""),
                "text": str(cand.get("text") or ""),
                "gold_support": int(cand.get("gold_support") or 0),
            }
        )
    return docs


def gold_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    gold_keys = {title_key(title) for title in row.get("gold_titles") or [] if title_key(title)}
    docs = []
    seen: set[str] = set()
    for doc in candidate_docs(row):
        key = title_key(doc.get("title"))
        if key in seen:
            continue
        if int(doc.get("gold_support") or 0) or key in gold_keys:
            docs.append(doc)
            seen.add(key)
    return docs


def title_to_doc_text(docs: Sequence[dict[str, Any]]) -> dict[str, str]:
    return {title_key(doc.get("title")): f"{doc.get('title', '')}\n{doc.get('text', '')}" for doc in docs}


def source_title_for_evidence(row: dict[str, Any], head: str, tail: str) -> str:
    gold = [str(title) for title in row.get("gold_titles") or []]
    head_key = title_key(head)
    tail_key = title_key(tail)
    for title in gold:
        if title_key(title) == head_key:
            return title
    for title in gold:
        if title_key(title) == tail_key:
            return title
    return gold[0] if gold else str(head)


def oracle_propositions(row: dict[str, Any]) -> list[dict[str, Any]]:
    props: list[dict[str, Any]] = []
    question = str(row.get("question") or "")
    for idx, ev in enumerate(row.get("evidences") or []):
        if not isinstance(ev, (list, tuple)) or len(ev) < 3:
            continue
        head, rel, tail = str(ev[0]), str(ev[1]), str(ev[2])
        props.append(
            {
                "prop_id": f"{row.get('qid') or row.get('query_idx')}::oracle::{idx}",
                "demand_id": f"u{idx + 1}",
                "source_title": source_title_for_evidence(row, head, tail),
                "predicate_text": rel,
                "predicate_bucket": bucketize_relation(rel),
                "args": [
                    {"text": head, "type": infer_entity_type(head, question)},
                    {"text": tail, "type": infer_entity_type(tail, question)},
                ],
                "source_span": f"{head} --{rel}--> {tail}",
            }
        )
    return props


def oracle_program(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or "")
    var_by_entity: dict[str, str] = {}
    entity_by_var: dict[str, str] = {"y": answer}
    next_var = 1

    def var_for(entity: str) -> str:
        nonlocal next_var
        key = norm_text(entity)
        if key == norm_text(answer):
            return "y"
        if key not in var_by_entity:
            var = f"x{next_var}"
            next_var += 1
            var_by_entity[key] = var
            entity_by_var[var] = entity
        return var_by_entity[key]

    demands = []
    for idx, ev in enumerate(row.get("evidences") or []):
        if not isinstance(ev, (list, tuple)) or len(ev) < 3:
            continue
        head, rel, tail = str(ev[0]), str(ev[1]), str(ev[2])
        demands.append(
            {
                "id": f"u{idx + 1}",
                "predicate_bucket": bucketize_relation(rel),
                "predicate_text": rel,
                "args": [var_for(head), var_for(tail)],
                "arg_entities": [head, tail],
            }
        )
    dependencies = []
    for left in demands:
        for right in demands:
            if left["id"] >= right["id"]:
                continue
            shared = sorted(set(left["args"]) & set(right["args"]))
            for var in shared:
                dependencies.append([left["id"], right["id"], f"shared:{var}"])
    return {
        "program_id": "oracle",
        "question_type": str(row.get("type") or ""),
        "answer_type": infer_answer_type(answer, str(row.get("question") or "")),
        "variables": [{"name": var, "entity": entity_by_var[var], "type": infer_entity_type(entity_by_var[var])} for var in sorted(entity_by_var)],
        "demands": demands,
        "dependencies": dependencies,
        "answer_variable": "y",
    }


def prop_matches_demand(prop: dict[str, Any], demand: dict[str, Any], expected_answer: str) -> bool:
    if canonical_bucket(prop.get("predicate_bucket")) != canonical_bucket(demand.get("predicate_bucket")):
        return False
    prop_args = [str(arg.get("text") or "") for arg in prop.get("args") or []]
    demand_entities = []
    for var, entity in zip(demand.get("args") or [], demand.get("arg_entities") or []):
        demand_entities.append(str(expected_answer) if var == "y" else str(entity))
    if len(prop_args) < len(demand_entities):
        return False
    return all(contains_norm(prop_arg, demand_entity) for prop_arg, demand_entity in zip(prop_args, demand_entities))


def certificate_for_answer(program: dict[str, Any], props: Sequence[dict[str, Any]], answer: str) -> dict[str, Any]:
    covered = []
    evidence = []
    theta: dict[str, str] = {}
    binding_conflicts = []
    for demand in program.get("demands") or []:
        matched = None
        for prop in props:
            if prop_matches_demand(prop, demand, answer):
                matched = prop
                break
        if matched is None:
            continue
        covered.append(str(demand.get("id")))
        evidence.append(matched)
        for var, arg in zip(demand.get("args") or [], matched.get("args") or []):
            value = str(answer) if var == "y" else str(arg.get("text") or "")
            if var in theta and norm_text(theta[var]) != norm_text(value):
                binding_conflicts.append({"var": var, "left": theta[var], "right": value})
            theta[var] = value

    demand_ids = {str(d.get("id")) for d in program.get("demands") or []}
    covered_ids = set(covered)
    literal_answer_terminal = any(
        contains_norm(arg.get("text"), answer)
        for prop in evidence
        for arg in prop.get("args") or []
    )
    program_answer_type = str(program.get("answer_type") or "Other")
    candidate_answer_type = infer_answer_type(answer)
    is_boolean_comparison = program_answer_type == "Boolean" and normalize_question_type(program.get("question_type")) == "comparison"
    if is_boolean_comparison:
        answer_terminal = candidate_answer_type == "Boolean" and demand_ids.issubset(covered_ids)
    else:
        answer_terminal = literal_answer_terminal
    prop_ids = [str(prop.get("prop_id") or "") for prop in evidence if prop.get("prop_id")]
    has_duplicate_evidence = len(prop_ids) != len(set(prop_ids))
    complete = bool(demand_ids and demand_ids.issubset(covered_ids) and answer_terminal and not binding_conflicts)
    key = (
        int(complete),
        len(covered_ids),
        int(answer_terminal),
        int(not binding_conflicts),
        -len(evidence),
    )
    reject_reasons = []
    if not answer_terminal:
        reject_reasons.append("answer_terminal")
    if binding_conflicts:
        reject_reasons.append("binding")
    if program_answer_type != "Other" and candidate_answer_type != program_answer_type:
        reject_reasons.append("type")
    if not complete and len(covered_ids) < len(demand_ids):
        reject_reasons.append("demand_incomplete")
    return {
        "answer": answer,
        "complete": complete,
        "covered_demands": sorted(covered_ids),
        "demand_count": len(demand_ids),
        "answer_terminal": answer_terminal,
        "binding_consistent": not binding_conflicts,
        "binding_conflicts": binding_conflicts,
        "has_duplicate_evidence": has_duplicate_evidence,
        "evidence_titles": [str(prop.get("source_title") or "") for prop in evidence],
        "theta": theta,
        "key": key,
        "reject_reasons": reject_reasons,
    }


def plausible_wrong_answer(row: dict[str, Any]) -> str:
    answer = str(row.get("answer") or "")
    qtype = infer_answer_type(answer, str(row.get("question") or ""))
    candidates: list[tuple[int, str]] = []
    for doc in candidate_docs(row):
        blob = f"{doc.get('title', '')}\n{doc.get('text', '')}"
        if qtype == "Date":
            values = DATE_RE.findall(blob)
        elif qtype == "Number":
            values = re.findall(r"\b\d[\d,.\s%-]{0,12}\b", blob)
        else:
            values = [str(doc.get("title") or "")]
            values.extend(CAPITALIZED_RE.findall(str(doc.get("text") or "")[:900]))
        for value in values:
            clean = re.sub(r"\s+", " ", str(value)).strip(" ,.;:")
            if not clean or norm_text(clean) == norm_text(answer):
                continue
            if norm_text(answer) and (norm_text(clean) in norm_text(answer) or norm_text(answer) in norm_text(clean)):
                continue
            if len(clean.split()) > 8:
                continue
            candidates.append((int(doc.get("rank") or 999999), clean))
    seen = set()
    for _rank, value in sorted(candidates, key=lambda item: item[0]):
        key = norm_text(value)
        if key in seen:
            continue
        seen.add(key)
        return value
    return ""


class JsonlCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.values: dict[str, Any] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.values[str(row["key"])] = row.get("value")

    def get(self, key: str) -> Any | None:
        return self.values.get(str(key))

    def set(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.values[str(key)] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value, "metadata": metadata or {}}, ensure_ascii=False) + "\n")


def chat_completion_raw(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }
    req = urllib.request.Request(
        str(base_url).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str((payload.get("choices") or [{}])[0].get("message", {}).get("content") or "")


def probe_llm(base_url: str, model: str, timeout: int) -> dict[str, Any]:
    try:
        raw = chat_completion_raw(
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": "/no_think\nReturn exactly: OK"}],
            max_tokens=16,
            timeout=timeout,
        )
        stripped = strip_think_blocks(raw)
        return {"available": "ok" in stripped.lower() or "ok" in raw.lower(), "raw": raw, "stripped": stripped, "error": ""}
    except Exception as exc:
        return {"available": False, "raw": "", "error": str(exc)}


def extraction_messages(row: dict[str, Any], docs: Sequence[dict[str, Any]], max_doc_chars: int) -> list[dict[str, str]]:
    doc_lines = []
    for idx, doc in enumerate(docs, start=1):
        text = f"{doc.get('title', '')}\n{doc.get('text', '')}"[: int(max_doc_chars)]
        doc_lines.append(f"[{idx}] TITLE: {doc.get('title', '')}\nTEXT:\n{text}")
    prompt = {
        "role": "user",
        "content": (
            "/no_think\n"
            "Extract only evidence-grounded typed propositions from the provided gold support documents.\n"
            "Return strict JSON with this schema: {\"propositions\": [{\"predicate_bucket\": string, "
            "\"predicate_text\": string, \"args\": [{\"text\": string, \"type\": string}], "
            "\"source_title\": string, \"source_span\": string}]}.\n"
            f"Allowed predicate_bucket values: {', '.join(PREDICATE_BUCKETS)}.\n"
            f"Allowed arg type values: {', '.join(TYPE_LABELS)}.\n"
            "Each proposition must have at least two args in subject-object order. "
            "For example, 'Aas Ka Panchhi was released in 1961' becomes args "
            "[{\"text\":\"Aas Ka Panchhi\",\"type\":\"Work\"},{\"text\":\"1961\",\"type\":\"Date\"}]. "
            "For comparison questions, extract factual propositions, not the comparison result.\n"
            "Do not hallucinate facts not stated in the documents. Keep source_span short and copied from the document.\n\n"
            f"Question: {row.get('question')}\n"
            f"Gold answer: {row.get('answer')}\n\n"
            "Documents:\n" + "\n\n".join(doc_lines)
        ),
    }
    return [{"role": "system", "content": "You are a strict information extraction engine. Output JSON only."}, prompt]


def program_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = {
        "role": "user",
        "content": (
            "/no_think\n"
            "Compile the question into a small lattice of up to 3 query programs for multi-hop evidence retrieval.\n"
            "Return strict JSON with schema: {\"programs\": [{\"program_id\": string, \"question_type\": "
            "\"compositional|comparison|inference|bridge|other\", \"answer_type\": \"Person|Place|Work|Organization|Date|Number|Boolean|Other\", "
            "\"variables\": [{\"name\": string, \"type\": string}], \"demands\": [{\"id\": string, "
            "\"predicate_bucket\": string, \"predicate_text\": string, \"args\": [string, string]}], "
            "\"dependencies\": [[string, string, string]], \"answer_variable\": \"y\"}]}.\n"
            f"Allowed predicate_bucket values: {', '.join(PREDICATE_BUCKETS)}.\n"
            "Use variables like x1, x2, y. The final answer slot must be variable y. "
            "The answer variable is the value returned to the user. If the question asks 'Which film/person/place ...', "
            "then y is that film/person/place entity, not the date/number used for comparison. "
            "Use auxiliary variables such as x1_date or x2_country for comparison attributes. Output JSON only.\n\n"
            f"Question: {row.get('question')}"
        ),
    }
    return [{"role": "system", "content": "You are a strict query compiler. Output JSON only."}, prompt]


def llm_generate(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    base_url: str,
    model: str,
    llm_limit: int,
    timeout: int,
    max_tokens: int,
    max_doc_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache = JsonlCache(output_dir / "phase0_llm_cache.jsonl")
    extraction_rows = []
    program_rows = []
    started = time.perf_counter()
    errors = Counter()
    for row in list(rows)[: int(llm_limit)]:
        qid = str(row.get("qid") or row.get("query_idx"))
        docs = gold_docs(row)
        for kind, messages in (
            ("extraction", extraction_messages(row, docs, max_doc_chars)),
            ("program", program_messages(row)),
        ):
            key = sha1_json(
                {
                    "prompt_version": "qbindcert_phase0_v2",
                    "kind": kind,
                    "model": model,
                    "qid": qid,
                    "question": row.get("question"),
                    "docs": [d.get("title") for d in docs],
                }
            )
            raw = cache.get(key)
            if raw is None:
                try:
                    raw = chat_completion_raw(
                        base_url=base_url,
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    cache.set(key, raw, {"kind": kind, "qid": qid})
                except Exception as exc:
                    raw = f"ERROR: {exc}"
                    errors[f"{kind}_call_failed"] += 1
            parsed = parse_json_payload(str(raw))
            item = {
                "qid": qid,
                "query_idx": row.get("query_idx"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "type": row.get("type"),
                "gold_titles": row.get("gold_titles"),
                "evidences": row.get("evidences"),
                "raw": raw,
                "parsed": parsed,
                "parse_ok": parsed is not None,
            }
            if kind == "extraction":
                item["gold_docs"] = docs
                extraction_rows.append(item)
            else:
                program_rows.append(item)
    runtime = {
        "llm_rows": min(len(rows), int(llm_limit)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "seconds_per_query": round((time.perf_counter() - started) / float(max(1, min(len(rows), int(llm_limit)))), 3),
        "errors": dict(errors),
    }
    return extraction_rows, program_rows, runtime


def propositions_from_parsed(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        props = parsed.get("propositions") or parsed.get("props") or []
    elif isinstance(parsed, list):
        props = parsed
    else:
        props = []
    return [p for p in props if isinstance(p, dict)]


def programs_from_parsed(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        programs = parsed.get("programs") or []
    elif isinstance(parsed, list):
        programs = parsed
    else:
        programs = []
    return [p for p in programs if isinstance(p, dict)]


def evaluate_extraction_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    predicate_hits = 0
    arg_hits = 0
    oracle_total = 0
    provenance_hits = 0
    hallucination_ok = 0
    extracted_total = 0
    parse_ok = 0
    samples = []
    for row in rows:
        if row.get("parse_ok"):
            parse_ok += 1
        oracle = oracle_propositions(row)
        extracted = propositions_from_parsed(row.get("parsed"))
        docs_by_title = title_to_doc_text(row.get("gold_docs") or [])
        oracle_total += len(oracle)
        extracted_total += len(extracted)
        for expected in oracle:
            best = None
            for prop in extracted:
                args = [str(arg.get("text") or "") for arg in prop.get("args") or []]
                exp_args = [str(arg.get("text") or "") for arg in expected.get("args") or []]
                arg_match = len(args) >= 2 and all(any(contains_norm(a, e) for a in args) for e in exp_args)
                bucket_match = canonical_bucket(prop.get("predicate_bucket")) == canonical_bucket(expected.get("predicate_bucket"))
                if bucket_match or arg_match:
                    best = prop
                    if bucket_match:
                        predicate_hits += 1
                    if arg_match:
                        arg_hits += 1
                    break
            if best is None:
                continue
        for prop in extracted:
            source_key = title_key(prop.get("source_title"))
            source_text = docs_by_title.get(source_key, "")
            span = str(prop.get("source_span") or "")
            args = [str(arg.get("text") or "") for arg in prop.get("args") or []]
            if source_text and (not span or contains_norm(source_text, span) or any(contains_norm(source_text, arg) for arg in args)):
                provenance_hits += 1
            if source_text and all((not arg) or contains_norm(source_text, arg) or any(contains_norm(arg, e) for ev in row.get("evidences") or [] for e in ev) for arg in args):
                hallucination_ok += 1
        if len(samples) < 20:
            samples.append(
                {
                    "qid": row.get("qid"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "oracle": oracle,
                    "extracted": extracted,
                    "parse_ok": row.get("parse_ok"),
                }
            )
    metrics = {
        "rows": len(rows),
        "parse_ok_rate": safe_div(parse_ok, len(rows)),
        "predicate_bucket_correct_proxy": safe_div(predicate_hits, oracle_total),
        "argument_extraction_correct_proxy": safe_div(arg_hits, oracle_total),
        "source_span_provenance_correct_proxy": safe_div(provenance_hits, extracted_total),
        "no_hallucinated_proposition_proxy": safe_div(hallucination_ok, extracted_total),
        "oracle_propositions": oracle_total,
        "extracted_propositions": extracted_total,
    }
    thresholds = {
        "predicate_bucket_correct_proxy": 0.80,
        "argument_extraction_correct_proxy": 0.80,
        "source_span_provenance_correct_proxy": 0.90,
        "no_hallucinated_proposition_proxy": 0.90,
    }
    status = gate_status_proxy(metrics, thresholds, ran=bool(rows))
    return {"name": "typed_proposition_extraction", "status": status, "metrics": metrics, "thresholds": thresholds, "samples": samples}


def program_bucket_sequence(program: dict[str, Any]) -> list[str]:
    return [canonical_bucket(d.get("predicate_bucket")) for d in program.get("demands") or [] if d.get("predicate_bucket")]


def program_is_connected(program: dict[str, Any]) -> bool:
    demands = list(program.get("demands") or [])
    if len(demands) <= 1:
        return True
    seen_vars: set[str] = set()
    for demand in demands:
        args = {str(arg) for arg in demand.get("args") or []}
        if seen_vars and seen_vars & args:
            return True
        seen_vars |= args
    deps = program.get("dependencies") or []
    return bool(deps)


def program_answer_variable_correct(program: dict[str, Any]) -> bool:
    if str(program.get("answer_variable") or "") != "y":
        return False
    return any("y" in [str(arg) for arg in demand.get("args") or []] for demand in program.get("demands") or [])


def normalize_question_type(value: Any) -> str:
    raw = norm_text(value)
    if "comparison" in raw:
        return "comparison"
    if "inference" in raw:
        return "inference"
    if "compositional" in raw or "composition" in raw or "bridge" in raw:
        return "compositional"
    return raw or "other"


def program_matches_oracle(program: dict[str, Any], oracle: dict[str, Any]) -> bool:
    oracle_buckets = Counter(d.get("predicate_bucket") for d in oracle.get("demands") or [])
    prog_buckets = Counter(program_bucket_sequence(program))
    bucket_ok = all(prog_buckets[bucket] >= count for bucket, count in oracle_buckets.items())
    hop_ok = abs(len(program.get("demands") or []) - len(oracle.get("demands") or [])) <= 1
    answer_ok = program_answer_variable_correct(program)
    connected_ok = program_is_connected(program)
    return bool(bucket_ok and hop_ok and answer_ok and connected_ok)


def evaluate_program_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    top1_correct = 0
    any_correct = 0
    answer_var_correct = 0
    dependency_correct = 0
    operator_correct = 0
    parse_ok = 0
    error_modes = Counter()
    samples = []
    for row in rows:
        oracle = oracle_program(row)
        programs = programs_from_parsed(row.get("parsed"))
        if row.get("parse_ok"):
            parse_ok += 1
        if not programs:
            error_modes["parse_failed"] += 1
            continue
        top1 = programs[0]
        if program_matches_oracle(top1, oracle):
            top1_correct += 1
        else:
            oracle_buckets = Counter(d.get("predicate_bucket") for d in oracle.get("demands") or [])
            prog_buckets = Counter(program_bucket_sequence(top1))
            if not all(prog_buckets[b] >= c for b, c in oracle_buckets.items()):
                error_modes["operator_or_relation_wrong"] += 1
            if not program_answer_variable_correct(top1):
                error_modes["answer_variable_wrong"] += 1
            if not program_is_connected(top1):
                error_modes["dependency_wrong"] += 1
        if any(program_matches_oracle(program, oracle) for program in programs[:3]):
            any_correct += 1
        if any(program_answer_variable_correct(program) for program in programs[:3]):
            answer_var_correct += 1
        if any(program_is_connected(program) for program in programs[:3]):
            dependency_correct += 1
        expected_type = normalize_question_type(row.get("type"))
        if any(normalize_question_type(program.get("question_type")) == expected_type for program in programs[:3]):
            operator_correct += 1
        else:
            error_modes["type_wrong"] += 1
        if len(samples) < 20:
            samples.append(
                {
                    "qid": row.get("qid"),
                    "question": row.get("question"),
                    "dataset_type": row.get("type"),
                    "oracle_program": oracle,
                    "programs": programs[:3],
                    "top1_correct_proxy": program_matches_oracle(top1, oracle),
                    "any_correct_proxy": any(program_matches_oracle(program, oracle) for program in programs[:3]),
                }
            )
    metrics = {
        "rows": len(rows),
        "parse_ok_rate": safe_div(parse_ok, len(rows)),
        "top3_any_program_correct_proxy": safe_div(any_correct, len(rows)),
        "top1_program_correct_proxy": safe_div(top1_correct, len(rows)),
        "answer_variable_correct_proxy": safe_div(answer_var_correct, len(rows)),
        "dependency_binding_correct_proxy": safe_div(dependency_correct, len(rows)),
        "question_type_operator_correct_proxy": safe_div(operator_correct, len(rows)),
        "error_modes": dict(error_modes),
    }
    thresholds = {
        "top3_any_program_correct_proxy": 0.75,
        "top1_program_correct_proxy": 0.60,
        "answer_variable_correct_proxy": 0.75,
        "dependency_binding_correct_proxy": 0.75,
        "question_type_operator_correct_proxy": 0.80,
    }
    status = gate_status_proxy(metrics, thresholds, ran=bool(rows))
    return {"name": "query_program_lattice", "status": status, "metrics": metrics, "thresholds": thresholds, "samples": samples}


def build_entity_pairs(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    pairs = []
    for row in rows:
        titles = [str(title) for title in row.get("gold_titles") or []]
        for ev in row.get("evidences") or []:
            if not isinstance(ev, (list, tuple)):
                continue
            for entity in (ev[0], ev[2]) if len(ev) >= 3 else []:
                for title in titles:
                    if title_key(entity) == title_key(title):
                        pairs.append({"left": str(entity), "right": title, "label": "alias", "expected_merge": True, "source": "evidence_title"})
        docs = candidate_docs(row)
        for i, left in enumerate(docs[:15]):
            for right in docs[i + 1 : 15]:
                same_title = title_key(left.get("title")) == title_key(right.get("title"))
                if same_title:
                    pairs.append(
                        {
                            "left": left.get("title"),
                            "right": right.get("title"),
                            "label": "same_title_duplicate",
                            "expected_merge": True,
                            "source": "candidate_same_title",
                        }
                    )
                elif len(pairs) < int(limit):
                    pairs.append(
                        {
                            "left": left.get("title"),
                            "right": right.get("title"),
                            "label": "different_title",
                            "expected_merge": False,
                            "source": "candidate_negative",
                        }
                    )
                if len(pairs) >= int(limit):
                    return pairs[: int(limit)]
    return pairs[: int(limit)]


def normalizer_merge(left: Any, right: Any) -> bool:
    lk = entity_key(left)
    rk = entity_key(right)
    if not lk or not rk:
        return False
    return lk == rk


def same_title_duplicate(left: Any, right: Any) -> bool:
    return bool(title_key(left) and title_key(left) == title_key(right))


def evaluate_entity_normalization(rows: Sequence[dict[str, Any]], pair_limit: int) -> dict[str, Any]:
    pairs = build_entity_pairs(rows, pair_limit)
    alias_pairs = [pair for pair in pairs if pair["expected_merge"]]
    negative_pairs = [pair for pair in pairs if not pair["expected_merge"]]
    same_title_pairs = [pair for pair in pairs if pair["label"] == "same_title_duplicate"]
    alias_correct = sum(1 for pair in alias_pairs if normalizer_merge(pair["left"], pair["right"]))
    false_merge = sum(1 for pair in negative_pairs if normalizer_merge(pair["left"], pair["right"]))
    same_title_correct = sum(1 for pair in same_title_pairs if same_title_duplicate(pair["left"], pair["right"]))
    metrics = {
        "pairs": len(pairs),
        "alias_pairs": len(alias_pairs),
        "negative_pairs": len(negative_pairs),
        "same_title_pairs": len(same_title_pairs),
        "alias_merge_correct_proxy": safe_div(alias_correct, len(alias_pairs)),
        "false_merge_rate_proxy": safe_div(false_merge, len(negative_pairs)),
        "same_title_duplicate_detection_proxy": safe_div(same_title_correct, len(same_title_pairs)) if same_title_pairs else 1.0,
    }
    thresholds = {
        "alias_merge_correct_proxy": 0.85,
        "false_merge_rate_proxy_max": 0.10,
        "same_title_duplicate_detection_proxy": 0.90,
    }
    if not pairs:
        status = "NOT_RUN"
    elif metrics["alias_merge_correct_proxy"] >= 0.85 and metrics["false_merge_rate_proxy"] <= 0.10 and metrics["same_title_duplicate_detection_proxy"] >= 0.90:
        status = "NEEDS_MANUAL_AUDIT"
    else:
        status = "FAIL_PROXY"
    return {
        "name": "entity_normalization",
        "status": status,
        "metrics": metrics,
        "thresholds": thresholds,
        "pairs": pairs[: int(pair_limit)],
    }


def evaluate_oracle_certificate_search(rows: Sequence[dict[str, Any]], limit: int) -> dict[str, Any]:
    cert_rows = []
    for row in list(rows)[: int(limit)]:
        program = oracle_program(row)
        props = oracle_propositions(row)
        cert = certificate_for_answer(program, props, str(row.get("answer") or ""))
        cert_rows.append(
            {
                "qid": row.get("qid"),
                "query_idx": row.get("query_idx"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "program": program,
                "certificate": cert,
            }
        )
    complete = sum(1 for row in cert_rows if row["certificate"]["complete"])
    answer_agree = sum(1 for row in cert_rows if row["certificate"]["complete"] and norm_text(row["certificate"]["answer"]) == norm_text(row["answer"]))
    no_dup = sum(1 for row in cert_rows if not row["certificate"]["has_duplicate_evidence"])
    metrics = {
        "rows": len(cert_rows),
        "complete_certificate_found": safe_div(complete, len(cert_rows)),
        "certificate_answer_agrees_with_gold": safe_div(answer_agree, len(cert_rows)),
        "no_non_contributing_duplicate_evidence": safe_div(no_dup, len(cert_rows)),
    }
    thresholds = {
        "complete_certificate_found": 0.70,
        "certificate_answer_agrees_with_gold": 0.65,
        "no_non_contributing_duplicate_evidence": 0.90,
    }
    status = gate_status(metrics, thresholds, ran=bool(cert_rows))
    return {"name": "oracle_certificate_search", "status": status, "metrics": metrics, "thresholds": thresholds, "rows": cert_rows}


def evaluate_answer_contrastive_oracle(rows: Sequence[dict[str, Any]], limit: int) -> dict[str, Any]:
    cert_rows = []
    hard_rows = [row for row in rows if str(row.get("type") or "") in {"compositional", "inference", "comparison"}]
    for row in hard_rows[: int(limit)]:
        wrong = plausible_wrong_answer(row)
        if not wrong:
            continue
        program = oracle_program(row)
        props = oracle_propositions(row)
        gold_cert = certificate_for_answer(program, props, str(row.get("answer") or ""))
        wrong_cert = certificate_for_answer(program, props, wrong)
        cert_rows.append(
            {
                "qid": row.get("qid"),
                "query_idx": row.get("query_idx"),
                "question": row.get("question"),
                "gold_answer": row.get("answer"),
                "wrong_answer": wrong,
                "gold_certificate": gold_cert,
                "wrong_certificate": wrong_cert,
                "gold_beats_wrong": tuple(gold_cert["key"]) > tuple(wrong_cert["key"]),
                "wrong_rejected_by": wrong_cert["reject_reasons"],
                "query_program_error_caused_wrong_win": False,
            }
        )
    gold_beats = sum(1 for row in cert_rows if row["gold_beats_wrong"])
    rejected_structural = sum(
        1
        for row in cert_rows
        if any(reason in {"answer_terminal", "binding", "type"} for reason in row["wrong_rejected_by"])
    )
    program_error_wrong = sum(1 for row in cert_rows if row["query_program_error_caused_wrong_win"])
    metrics = {
        "rows": len(cert_rows),
        "gold_certificate_beats_wrong": safe_div(gold_beats, len(cert_rows)),
        "wrong_rejected_by_answer_terminal_binding_type": safe_div(rejected_structural, len(cert_rows)),
        "query_program_error_caused_wrong_win": safe_div(program_error_wrong, len(cert_rows)),
    }
    thresholds = {
        "gold_certificate_beats_wrong": 0.65,
        "wrong_rejected_by_answer_terminal_binding_type": 0.40,
        "query_program_error_caused_wrong_win_max": 0.25,
    }
    if not cert_rows:
        status = "NOT_RUN"
    elif (
        metrics["gold_certificate_beats_wrong"] >= 0.65
        and metrics["wrong_rejected_by_answer_terminal_binding_type"] >= 0.40
        and metrics["query_program_error_caused_wrong_win"] <= 0.25
    ):
        status = "PASS"
    else:
        status = "FAIL"
    return {"name": "answer_contrastive_oracle_certificate", "status": status, "metrics": metrics, "thresholds": thresholds, "rows": cert_rows}


def gate_status(metrics: dict[str, Any], thresholds: dict[str, float], *, ran: bool) -> str:
    if not ran:
        return "NOT_RUN"
    for key, threshold in thresholds.items():
        if key.endswith("_max"):
            metric_key = key[: -len("_max")]
            if float(metrics.get(metric_key, 1.0)) > float(threshold):
                return "FAIL"
        elif float(metrics.get(key, 0.0)) < float(threshold):
            return "FAIL"
    return "PASS"


def gate_status_proxy(metrics: dict[str, Any], thresholds: dict[str, float], *, ran: bool) -> str:
    if not ran:
        return "NOT_RUN"
    hard = gate_status(metrics, thresholds, ran=ran)
    return "NEEDS_MANUAL_AUDIT" if hard == "PASS" else "FAIL_PROXY"


def overall_decision(tests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    statuses = {test["name"]: test["status"] for test in tests}
    if any(status in {"FAIL", "FAIL_PROXY"} for status in statuses.values()):
        decision = "STOP_PHASE0_FAIL"
        phase1_justified = False
    elif any(status in {"NOT_RUN", "NEEDS_MANUAL_AUDIT"} for status in statuses.values()):
        decision = "PHASE0_NOT_CLEARED"
        phase1_justified = False
    else:
        decision = "PHASE0_CLEARED"
        phase1_justified = True
    return {
        "decision": decision,
        "phase1_justified": phase1_justified,
        "statuses": statuses,
        "notes": [
            "Manual-quality gates cannot be promoted from proxy PASS to final PASS by this script.",
            "Proceed to Phase 1 only if all statuses become PASS after manual audit.",
        ],
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Q-BindCert Phase-0 Sanity",
        "",
        "## Decision",
        "",
        f"- decision: `{report['decision']['decision']}`",
        f"- phase1_justified: `{report['decision']['phase1_justified']}`",
        f"- rows: `{report['config']['limit']}`",
        f"- llm_run: `{report['config']['run_llm']}`",
        f"- llm_rows: `{report['runtime'].get('llm', {}).get('llm_rows', 0)}`",
        "",
        "## Gate Summary",
        "",
        "| Test | Status | Key metrics |",
        "|---|---|---|",
    ]
    for test in report["tests"]:
        metrics = test.get("metrics") or {}
        compact = []
        for key, value in metrics.items():
            if key in {"rows", "pairs", "parse_ok_rate"} or isinstance(value, (int, float)):
                if isinstance(value, float):
                    compact.append(f"{key}={value:.4f}")
                else:
                    compact.append(f"{key}={value}")
        lines.append(f"| `{test['name']}` | `{test['status']}` | {'; '.join(compact)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS`/`FAIL` are used only for oracle-controlled automatic tests.",
            "- `NEEDS_MANUAL_AUDIT` means the proxy check did not falsify the gate, but Phase 1 is still blocked.",
            "- `FAIL_PROXY` is sufficient to stop unless a manual audit explicitly overturns the proxy.",
            "- This run does not implement full Q-BindCert and does not use reader/verifier feedback.",
            "",
            "## LLM Probe",
            "",
            f"- available: `{report['runtime'].get('llm_probe', {}).get('available')}`",
            f"- raw: `{report['runtime'].get('llm_probe', {}).get('raw', '')}`",
            f"- error: `{report['runtime'].get('llm_probe', {}).get('error', '')}`",
            "",
            "## Outputs",
            "",
        ]
    )
    for label, out_path in report.get("outputs", {}).items():
        lines.append(f"- {label}: `{out_path}`")
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--output-dir", default="reports/qbindcert")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--oracle-limit", type=int, default=50)
    parser.add_argument("--entity-pair-limit", type=int, default=200)
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--llm-limit", type=int, default=20)
    parser.add_argument("--llm-base-url", default="http://localhost:8043/v1")
    parser.add_argument("--llm-model", default="qwen3-8b-train")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-doc-chars", type=int, default=1400)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_cache_rows(args.cache_jsonl, limit=int(args.limit))
    runtime: dict[str, Any] = {
        "started_at_unix": started,
        "llm_probe": probe_llm(str(args.llm_base_url), str(args.llm_model), timeout=min(int(args.timeout), 15)),
    }

    extraction_rows: list[dict[str, Any]] = []
    program_rows: list[dict[str, Any]] = []
    if args.run_llm:
        if not runtime["llm_probe"]["available"]:
            runtime["llm"] = {"llm_rows": 0, "elapsed_seconds": 0.0, "errors": {"probe_failed": 1}}
        else:
            extraction_rows, program_rows, llm_runtime = llm_generate(
                rows,
                output_dir=output_dir,
                base_url=str(args.llm_base_url),
                model=str(args.llm_model),
                llm_limit=int(args.llm_limit),
                timeout=int(args.timeout),
                max_tokens=int(args.max_tokens),
                max_doc_chars=int(args.max_doc_chars),
            )
            runtime["llm"] = llm_runtime
    else:
        runtime["llm"] = {"llm_rows": 0, "elapsed_seconds": 0.0, "errors": {}}

    tests = [
        evaluate_extraction_rows(extraction_rows),
        evaluate_program_rows(program_rows),
        evaluate_entity_normalization(rows, int(args.entity_pair_limit)),
        evaluate_oracle_certificate_search(rows, int(args.oracle_limit)),
        evaluate_answer_contrastive_oracle(rows, int(args.oracle_limit)),
    ]
    decision = overall_decision(tests)

    extraction_path = output_dir / "phase0_extraction_samples.jsonl"
    program_path = output_dir / "phase0_program_samples.jsonl"
    entity_path = output_dir / "phase0_entity_pairs.jsonl"
    certificate_path = output_dir / "phase0_certificate_rows.jsonl"
    contrastive_path = output_dir / "phase0_contrastive_rows.jsonl"
    json_path = output_dir / "phase0_sanity.json"
    md_path = output_dir / "phase0_sanity.md"

    write_jsonl(extraction_rows, extraction_path)
    write_jsonl(program_rows, program_path)
    write_jsonl(tests[2].get("pairs") or [], entity_path)
    write_jsonl(tests[3].get("rows") or [], certificate_path)
    write_jsonl(tests[4].get("rows") or [], contrastive_path)

    report = {
        "config": {
            "cache_jsonl": str(args.cache_jsonl),
            "output_dir": str(output_dir),
            "limit": int(args.limit),
            "oracle_limit": int(args.oracle_limit),
            "entity_pair_limit": int(args.entity_pair_limit),
            "run_llm": bool(args.run_llm),
            "llm_limit": int(args.llm_limit),
            "llm_base_url": str(args.llm_base_url),
            "llm_model": str(args.llm_model),
        },
        "runtime": {
            **runtime,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "tests": [{key: value for key, value in test.items() if key not in {"rows", "pairs"}} for test in tests],
        "decision": decision,
        "outputs": {
            "json": str(json_path),
            "markdown": str(md_path),
            "extraction_samples": str(extraction_path),
            "program_samples": str(program_path),
            "entity_pairs": str(entity_path),
            "certificate_rows": str(certificate_path),
            "contrastive_rows": str(contrastive_path),
        },
    }
    write_json(report, json_path)
    write_markdown(report, md_path)
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({"decision": report["decision"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
