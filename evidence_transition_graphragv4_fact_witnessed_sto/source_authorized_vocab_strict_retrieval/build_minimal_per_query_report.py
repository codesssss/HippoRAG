"""Build minimal per-query reports from benchmark datasets.

The clean source-authorized pipeline only needs question text, gold document
indices, and candidate indices.  This utility reconstructs those fields from
the local benchmark JSON files without rerunning the legacy compare harness.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def unique_ints(values: Iterable[object]) -> List[int]:
    seen = set()
    output: List[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _title_to_local_indices(context: Sequence[Sequence[object]]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for local_index, item in enumerate(context):
        if not isinstance(item, Sequence) or not item:
            continue
        title = str(item[0])
        mapping.setdefault(title, []).append(int(local_index))
    return mapping


def _passage_title(text: object) -> str:
    raw = str(text or "")
    return raw.splitlines()[0].strip() if raw else ""


def _normalize_passage(text: object) -> str:
    return " ".join(str(text or "").split())


def _openie_indices(openie_path: Path) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    payload = load_json(openie_path)
    docs = payload.get("docs", []) if isinstance(payload, Mapping) else []
    title_to_indices: Dict[str, List[int]] = {}
    passage_to_index: Dict[str, int] = {}
    for doc_index, doc in enumerate(docs or []):
        if not isinstance(doc, Mapping):
            continue
        passage = str(doc.get("passage") or "")
        title = _passage_title(passage)
        if title:
            title_to_indices.setdefault(title, []).append(int(doc_index))
        normalized = _normalize_passage(passage)
        if normalized:
            passage_to_index.setdefault(normalized, int(doc_index))
    return title_to_indices, passage_to_index


def _context_passage(item: Sequence[object]) -> str:
    if not isinstance(item, Sequence) or len(item) < 2:
        return ""
    title = str(item[0])
    sentences = item[1]
    if isinstance(sentences, Sequence) and not isinstance(sentences, (str, bytes)):
        body = " ".join(str(sentence) for sentence in sentences)
    else:
        body = str(sentences)
    return f"{title}\n{body}".strip()


def _paragraph_passage(paragraph: Mapping[str, Any]) -> str:
    return f"{paragraph.get('title', '')}\n{paragraph.get('paragraph_text', '')}".strip()


def _dict_passage(item: Mapping[str, Any]) -> str:
    return f"{item.get('title', '')}\n{item.get('text', item.get('paragraph_text', ''))}".strip()


def _parse_answer_alias_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                return [cleaned]
            return _parse_answer_alias_values(parsed)
        return [cleaned]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases: List[str] = []
        for item in value:
            aliases.extend(_parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def _gold_from_marked_contexts(
    *,
    contexts: Sequence[Mapping[str, Any]],
    title_to_doc_indices: Mapping[str, Sequence[int]] | None = None,
    passage_to_doc_index: Mapping[str, int] | None = None,
    offset: int = 0,
) -> List[int]:
    title_to_doc_indices = title_to_doc_indices or {}
    passage_to_doc_index = passage_to_doc_index or {}
    output: List[int] = []
    for local_index, item in enumerate(contexts):
        if not isinstance(item, Mapping) or not item.get("is_supporting"):
            continue
        passage = _dict_passage(item)
        doc_index = passage_to_doc_index.get(_normalize_passage(passage))
        if doc_index is not None:
            output.append(int(doc_index))
            continue
        doc_indices = title_to_doc_indices.get(str(item.get("title") or ""), ())
        if len(doc_indices) == 1:
            output.append(int(doc_indices[0]))
            continue
        output.append(int(offset) + int(local_index))
    return unique_ints(output)


def _gold_from_supporting_facts(
    *,
    supporting_facts: Sequence[Sequence[object]],
    context: Sequence[Sequence[object]],
    title_to_doc_indices: Mapping[str, Sequence[int]] | None = None,
    passage_to_doc_index: Mapping[str, int] | None = None,
    offset: int = 0,
) -> List[int]:
    title_to_indices = _title_to_local_indices(context)
    title_to_doc_indices = title_to_doc_indices or {}
    passage_to_doc_index = passage_to_doc_index or {}
    output: List[int] = []
    for fact in supporting_facts:
        if not isinstance(fact, Sequence) or not fact:
            continue
        title = str(fact[0])
        matched_locals = title_to_indices.get(title, [])
        for local_index in matched_locals:
            passage = _context_passage(context[int(local_index)])
            doc_index = passage_to_doc_index.get(_normalize_passage(passage))
            if doc_index is not None:
                output.append(int(doc_index))
                continue
            doc_indices = title_to_doc_indices.get(title, ())
            if len(doc_indices) == 1:
                output.append(int(doc_indices[0]))
                continue
            output.append(int(offset) + int(local_index))
    return unique_ints(output)


def _rows_2wiki(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    title_to_doc_indices: Mapping[str, Sequence[int]] | None = None,
    passage_to_doc_index: Mapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    for query_index, row in enumerate(dataset_rows):
        context = list(row.get("context", []) or [])
        gold = _gold_from_supporting_facts(
            supporting_facts=list(row.get("supporting_facts", []) or []),
            context=context,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
            offset=offset,
        )
        rows.append(
            {
                "query_index": int(query_index),
                "question": str(row.get("question") or ""),
                "gold_answers": [str(row.get("answer") or "")],
                "gold_doc_indices": gold,
            }
        )
        offset += len(context)
    return rows


def _rows_hotpotqa(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    title_to_doc_indices: Mapping[str, Sequence[int]] | None = None,
    passage_to_doc_index: Mapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    for query_index, row in enumerate(dataset_rows):
        context = list(row.get("context", []) or [])
        gold = _gold_from_supporting_facts(
            supporting_facts=list(row.get("supporting_facts", []) or []),
            context=context,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
            offset=offset,
        )
        rows.append(
            {
                "query_index": int(query_index),
                "question": str(row.get("question") or ""),
                "gold_answers": [str(row.get("answer") or "")],
                "gold_doc_indices": gold,
            }
        )
        offset += len(context)
    return rows


def _rows_musique(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    title_to_doc_indices: Mapping[str, Sequence[int]] | None = None,
    passage_to_doc_index: Mapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    for query_index, row in enumerate(dataset_rows):
        paragraphs = list(row.get("paragraphs", []) or [])
        gold_values: List[int] = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping) or not paragraph.get("is_supporting"):
                continue
            passage = _paragraph_passage(paragraph)
            doc_index = (passage_to_doc_index or {}).get(_normalize_passage(passage))
            if doc_index is not None:
                gold_values.append(int(doc_index))
                continue
            doc_indices = (title_to_doc_indices or {}).get(str(paragraph.get("title") or ""), ())
            if len(doc_indices) == 1:
                gold_values.append(int(doc_indices[0]))
                continue
            gold_values.append(int(offset) + int(paragraph.get("idx")))
        gold = unique_ints(gold_values)
        answers = [str(row.get("answer") or "")]
        answers.extend(str(alias) for alias in row.get("answer_aliases", []) or [])
        rows.append(
            {
                "query_index": int(query_index),
                "question": str(row.get("question") or ""),
                "gold_answers": [answer for answer in dict.fromkeys(answers) if answer],
                "gold_doc_indices": gold,
            }
        )
        offset += len(paragraphs)
    return rows


def _rows_nq_rear(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    title_to_doc_indices: Mapping[str, Sequence[int]] | None = None,
    passage_to_doc_index: Mapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    for query_index, row in enumerate(dataset_rows):
        contexts = list(row.get("contexts", []) or [])
        gold = _gold_from_marked_contexts(
            contexts=contexts,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
            offset=offset,
        )
        answers = _parse_answer_alias_values(row.get("reference"))
        if "answer_aliases" in row:
            answers.extend(_parse_answer_alias_values(row.get("answer_aliases")))
        rows.append(
            {
                "query_index": int(query_index),
                "question": str(row.get("question") or ""),
                "gold_answers": [answer for answer in dict.fromkeys(answers) if answer],
                "gold_doc_indices": gold,
            }
        )
        offset += len(contexts)
    return rows


def _rows_popqa(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    title_to_doc_indices: Mapping[str, Sequence[int]] | None = None,
    passage_to_doc_index: Mapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    for query_index, row in enumerate(dataset_rows):
        paragraphs = list(row.get("paragraphs", []) or [])
        gold = _gold_from_marked_contexts(
            contexts=paragraphs,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
            offset=offset,
        )
        answers: List[str] = []
        for field in ("obj", "possible_answers", "o_wiki_title", "o_aliases", "answer_aliases"):
            answers.extend(_parse_answer_alias_values(row.get(field)))
        rows.append(
            {
                "query_index": int(query_index),
                "question": str(row.get("question") or ""),
                "gold_answers": [answer for answer in dict.fromkeys(answers) if answer],
                "gold_doc_indices": gold,
            }
        )
        offset += len(paragraphs)
    return rows


def build_minimal_report(
    *,
    dataset: str,
    dataset_path: Path,
    openie_path: Path,
    output_json: Path,
) -> Mapping[str, Any]:
    dataset_rows = load_json(dataset_path)
    if not isinstance(dataset_rows, list):
        raise ValueError(f"dataset file must contain a list: {dataset_path}")
    title_to_doc_indices, passage_to_doc_index = _openie_indices(openie_path)
    normalized_dataset = str(dataset)
    if normalized_dataset == "2wikimultihopqa":
        rows = _rows_2wiki(
            dataset_rows,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
        )
    elif normalized_dataset == "musique":
        rows = _rows_musique(
            dataset_rows,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
        )
    elif normalized_dataset == "hotpotqa":
        rows = _rows_hotpotqa(
            dataset_rows,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
        )
    elif normalized_dataset == "nq_rear":
        rows = _rows_nq_rear(
            dataset_rows,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
        )
    elif normalized_dataset == "popqa":
        rows = _rows_popqa(
            dataset_rows,
            title_to_doc_indices=title_to_doc_indices,
            passage_to_doc_index=passage_to_doc_index,
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    payload = {
        "dataset": normalized_dataset,
        "mode": "minimal_per_query",
        "num_queries": len(rows),
        "analysis_openie_path": str(openie_path.expanduser().resolve()),
        "variants": {
            "hipporag_v2": rows,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--openie-path", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_minimal_report(
        dataset=str(args.dataset),
        dataset_path=Path(args.dataset_path),
        openie_path=Path(args.openie_path),
        output_json=Path(args.output_json),
    )
    print(json.dumps({"dataset": payload["dataset"], "num_queries": payload["num_queries"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
