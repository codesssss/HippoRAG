#!/usr/bin/env python3
"""Build a query-obligation cache for closed-STO local PPR.

The selector in ``evaluate_obligation_closed_sto_local_ppr.py`` intentionally
abstains when no role-aligned query obligations are available. This script
creates that missing upstream object as a reusable cache:

    question -> query triples -> query_obligation_unit records

It does not read gold labels, QA outputs, SFB outputs, or retrieval outcomes.
The cache can be built either from an existing head-trace top-k-fact cache for
diagnosis, or from an LLM prompt that decomposes the question itself.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from build_query_obligation_units import build_query_obligation_units
from compare_graph_retrievers import install_qwen_disable_thinking
from evaluate_obligation_closed_sto_selector import DEFAULT_REPORT, load_json


DEFAULT_LLM_NAME = "gpt-4o-mini-2024-07-18"
DEFAULT_LLM_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_SAVE_DIR = "outputs_query_obligation_cache_20260428"


QUERY_OBLIGATION_SYSTEM_PROMPT = """You extract role-preserving query obligation triples from a question.

Return one JSON object only:
{"triples": [["subject", "relation", "object"], ...]}

Rules:
- Each triple is an obligation implied by the question, not an answer guess.
- Preserve relation direction. Do not reverse subject and object to make a fact look nicer.
- Use exact entity surface forms from the question when the entity is known.
- Use variables such as ?answer, ?x1, ?x2, ?date, or ?place for unknown answer or bridge nodes.
- Reuse the same variable when two obligations refer to the same unknown node.
- Keep relation phrases short and factual, for example "mother", "died in", "located in", "performed by".
- Decompose nested descriptions. Never use an endpoint like "the author of X", "the performer of X", "the city in Y", or "the state where Z is from"; introduce a variable and a separate triple instead.
- Prefer relation phrases that could appear in evidence text, for example "was born in" instead of only "is from" when the question asks where a person is from.
- Extract retrieval-support obligations, not the final unknown answer assertion. For a "Who won..." question, do not emit "?answer won ?race" unless the question itself gives one concrete endpoint for that fact. Stop at obligations that identify the answer-bearing evidence source.
- Do not include explanations or markdown.

Examples:
Question: When did Lothair II's mother die?
{"triples": [["Lothair II", "mother", "?x1"], ["?x1", "died in", "?date"]]}

Question: Who won the race in the city in the state where the author of Book X was born?
{"triples": [["Book X", "written by", "?x1"], ["?x1", "was born in", "?state"], ["?city", "located in", "?state"], ["?race", "held in", "?city"]]}

Question: Which city is in the state where the performer of Album Y is from?
{"triples": [["Album Y", "is an album by", "?x1"], ["?x1", "was born in", "?state"], ["?city", "located in", "?state"]]}
"""


QUERY_OBLIGATION_USER_PROMPT = """Question:
{question}

Extract the query obligation triples."""


def unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def coerce_triples(value: Any) -> List[List[str]]:
    triples: List[List[str]] = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return triples
    seen: Set[Tuple[str, str, str]] = set()
    for triple in value:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        subject, relation, obj = (str(item or "").strip() for item in triple)
        if not subject or not relation or not obj:
            continue
        key = (subject, relation, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append([subject, relation, obj])
    return triples


def strip_code_fence(text: str) -> str:
    stripped = str(text or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else stripped


def extract_json_like_payload(text: str) -> Any:
    """Parse a JSON object/list from a model response with minimal repair."""

    stripped = strip_code_fence(text)
    candidates = [stripped]
    first_obj = stripped.find("{")
    last_obj = stripped.rfind("}")
    if 0 <= first_obj < last_obj:
        candidates.append(stripped[first_obj : last_obj + 1])
    first_list = stripped.find("[")
    last_list = stripped.rfind("]")
    if 0 <= first_list < last_list:
        candidates.append(stripped[first_list : last_list + 1])

    errors: List[str] = []
    for candidate in unique_strings(candidates):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                errors.append(str(exc))
                continue
    raise ValueError("Unable to parse query obligation JSON response: " + " | ".join(errors[:3]))


def parse_query_obligation_response(response_text: str) -> List[List[str]]:
    payload = extract_json_like_payload(response_text)
    if isinstance(payload, Mapping):
        for key in ("triples", "query_triples", "obligations"):
            triples = coerce_triples(payload.get(key))
            if triples:
                return triples
        return []
    return coerce_triples(payload)


def render_query_obligation_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_OBLIGATION_SYSTEM_PROMPT},
        {"role": "user", "content": QUERY_OBLIGATION_USER_PROMPT.format(question=str(question))},
    ]


def parse_query_indices(spec: str) -> Set[int]:
    indices: Set[int] = set()
    for part in str(spec or "").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            indices.update(range(start, end + 1))
            continue
        indices.add(int(token))
    return indices


def parse_dataset_filter(spec: str) -> Set[str]:
    return {item.strip() for item in str(spec or "").split(",") if item.strip()}


def selected_rows_for_dataset(
    *,
    rows: Sequence[Mapping[str, Any]],
    max_queries: int,
    query_indices: Set[int],
) -> List[Mapping[str, Any]]:
    selected: List[Mapping[str, Any]] = []
    for row in rows:
        try:
            query_index = int(row.get("query_index", len(selected)))
        except (TypeError, ValueError):
            query_index = len(selected)
        if query_indices and query_index not in query_indices:
            continue
        selected.append(row)
        if max_queries > 0 and len(selected) >= max_queries:
            break
    return selected


def load_head_trace_topk_map(path: Path | None) -> Dict[str, Dict[int, List[List[str]]]]:
    if path is None:
        return {}
    raw_map = load_json(path)
    if not isinstance(raw_map, Mapping):
        raise ValueError("--head-traces-cache-map-json must point to a JSON object")
    result: Dict[str, Dict[int, List[List[str]]]] = {}
    for dataset, cache_path in raw_map.items():
        if not str(dataset).strip() or not str(cache_path).strip():
            continue
        payload = load_json(Path(str(cache_path)).resolve())
        traces = payload.get("head_traces", []) if isinstance(payload, Mapping) else []
        query_map: Dict[int, List[List[str]]] = {}
        if isinstance(traces, Sequence) and not isinstance(traces, (str, bytes)):
            for query_index, trace in enumerate(traces):
                top_k_facts = trace.get("top_k_facts", []) if isinstance(trace, Mapping) else []
                triples = coerce_triples(top_k_facts)
                if triples:
                    query_map[int(query_index)] = triples
        result[str(dataset)] = query_map
    return result


def build_cache_row(
    *,
    query_index: int,
    question: str,
    triples: Sequence[Sequence[Any]],
    source: str,
    status: str,
    metadata: Optional[Mapping[str, Any]] = None,
    raw_response: str | None = None,
) -> Dict[str, Any]:
    clean_triples = coerce_triples(triples)
    row: Dict[str, Any] = {
        "query_index": int(query_index),
        "question": str(question),
        "source": str(source),
        "status": str(status if status else ("ok" if clean_triples else "empty")),
        "query_triples": clean_triples,
        "query_obligations": build_query_obligation_units(query=str(question), query_triples=clean_triples),
    }
    if metadata:
        row["metadata"] = dict(metadata)
    if raw_response is not None:
        row["raw_response"] = str(raw_response)
    return row


def infer_query_obligation_triples(
    *,
    llm: Any,
    question: str,
    include_raw_response: bool,
) -> Tuple[List[List[str]], Dict[str, Any], str | None, str]:
    messages = render_query_obligation_messages(question)
    raw_response, metadata, cache_hit = llm.infer(messages=messages)
    metadata = dict(metadata or {})
    metadata["cache_hit"] = bool(cache_hit)
    triples = parse_query_obligation_response(str(raw_response))
    status = "ok" if triples else "empty"
    return triples, metadata, str(raw_response) if include_raw_response else None, status


def make_llm(
    *,
    llm_name: str,
    llm_base_url: str,
    save_dir: Path,
    max_new_tokens: int,
    max_retry_attempts: int,
) -> Any:
    from src.hipporag.llm.openai_gpt import CacheOpenAI
    from src.hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        llm_name=str(llm_name),
        llm_base_url=str(llm_base_url),
        save_dir=str(save_dir),
        max_new_tokens=int(max_new_tokens),
        max_retry_attempts=int(max_retry_attempts),
        temperature=0.0,
    )
    return CacheOpenAI.from_experiment_config(config)


def build_query_obligation_cache(
    *,
    report_path: Path,
    source: str,
    output_json_path: Path,
    datasets_filter: Set[str],
    query_indices: Set[int],
    max_queries: int,
    head_trace_topk_map: Mapping[str, Mapping[int, Sequence[Sequence[Any]]]],
    llm: Any | None = None,
    include_raw_response: bool = False,
) -> Dict[str, Any]:
    report = load_json(report_path)
    dataset_outputs: List[Dict[str, Any]] = []
    for dataset_payload in report.get("datasets", []) or []:
        dataset_name = str(dataset_payload.get("dataset") or "")
        if datasets_filter and dataset_name not in datasets_filter:
            continue
        rows = selected_rows_for_dataset(
            rows=list(dataset_payload.get("rows", []) or []),
            max_queries=max_queries,
            query_indices=query_indices,
        )
        cache_rows: List[Dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        for ordinal, row in enumerate(rows):
            try:
                query_index = int(row.get("query_index", ordinal))
            except (TypeError, ValueError):
                query_index = ordinal
            question = str(row.get("question") or row.get("query") or "")
            metadata: Dict[str, Any] = {}
            raw_response: str | None = None
            if source == "head_trace_top_k_facts":
                triples = coerce_triples((head_trace_topk_map.get(dataset_name, {}) or {}).get(query_index, []))
                status = "ok" if triples else "empty"
            elif source == "llm_query_obligation":
                if llm is None:
                    raise ValueError("llm is required when source=llm_query_obligation")
                try:
                    triples, metadata, raw_response, status = infer_query_obligation_triples(
                        llm=llm,
                        question=question,
                        include_raw_response=include_raw_response,
                    )
                except Exception as exc:
                    triples = []
                    metadata = {"error": str(exc)}
                    status = "error"
            else:
                raise ValueError(f"Unsupported source: {source}")
            cache_row = build_cache_row(
                query_index=query_index,
                question=question,
                triples=triples,
                source=source,
                status=status,
                metadata=metadata,
                raw_response=raw_response,
            )
            status_counts[cache_row["status"]] += 1
            cache_rows.append(cache_row)
        dataset_outputs.append(
            {
                "dataset": dataset_name,
                "rows": cache_rows,
                "metrics": {
                    "rows": len(cache_rows),
                    "ok_rows": int(status_counts.get("ok", 0)),
                    "empty_rows": int(status_counts.get("empty", 0)),
                    "error_rows": int(status_counts.get("error", 0)),
                    "triple_count": sum(len(row.get("query_triples", []) or []) for row in cache_rows),
                    "obligation_count": sum(len(row.get("query_obligations", []) or []) for row in cache_rows),
                    "cache_hit_rows": sum(
                        1
                        for row in cache_rows
                        if bool((row.get("metadata", {}) or {}).get("cache_hit", False))
                    ),
                },
            }
        )

    summary = {
        "method": "query_obligation_cache",
        "version": 1,
        "source_report_path": str(report_path),
        "source": source,
        "config": {
            "datasets": sorted(datasets_filter),
            "query_indices": sorted(query_indices),
            "max_queries": int(max_queries),
            "include_raw_response": bool(include_raw_response),
        },
        "datasets": dataset_outputs,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build query obligation cache from questions.")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--source",
        choices=("llm_query_obligation", "head_trace_top_k_facts"),
        default="llm_query_obligation",
    )
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset allowlist.")
    parser.add_argument("--query-indices", default="", help="Optional comma/range query indices, for example 714 or 0-4,714.")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--head-traces-cache-map-json", default="")
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--api-key-file", default="", help="Optional file containing OPENAI_API_KEY.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-retry-attempts", type=int, default=5)
    parser.add_argument(
        "--qwen-disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs.enable_thinking=false to Qwen OpenAI-compatible obligation-cache calls.",
    )
    parser.add_argument("--include-raw-response", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if str(args.api_key_file or "").strip():
        api_key_path = Path(str(args.api_key_file)).resolve()
        os.environ["OPENAI_API_KEY"] = api_key_path.read_text(encoding="utf-8").strip()

    head_trace_topk_map = load_head_trace_topk_map(
        Path(args.head_traces_cache_map_json).resolve()
        if str(args.head_traces_cache_map_json or "").strip()
        else None
    )
    llm = None
    if args.source == "llm_query_obligation":
        llm = make_llm(
            llm_name=str(args.llm_name),
            llm_base_url=str(args.llm_base_url),
            save_dir=Path(str(args.save_dir)).resolve(),
            max_new_tokens=max(int(args.max_new_tokens), 1),
            max_retry_attempts=max(int(args.max_retry_attempts), 1),
        )
        install_qwen_disable_thinking(
            llm,
            SimpleNamespace(qwen_disable_thinking=bool(args.qwen_disable_thinking)),
        )

    summary = build_query_obligation_cache(
        report_path=Path(args.report).resolve(),
        source=str(args.source),
        output_json_path=Path(args.output_json).resolve(),
        datasets_filter=parse_dataset_filter(args.datasets),
        query_indices=parse_query_indices(args.query_indices),
        max_queries=max(int(args.max_queries), 0),
        head_trace_topk_map=head_trace_topk_map,
        llm=llm,
        include_raw_response=bool(args.include_raw_response),
    )
    compact = {
        dataset["dataset"]: dataset.get("metrics", {})
        for dataset in summary.get("datasets", []) or []
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
