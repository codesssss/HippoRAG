#!/usr/bin/env python3
"""Reader-context diagnostic for frozen ETv3 retrieval reports.

This entrypoint does not change ETv3 retrieval.  It rewrites the reader context
for QA only: the first five documents remain the frozen ETv3 top5, then the
ETv3 candidate prefix is appended until ``--reader-context-k`` unique passages
are available.  The transformed report is passed to ETv3's reader-only QA
runner so the HippoRAG reader path stays unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence_transition_graphragv3_variable_flow.contract import (  # noqa: E402
    METHOD_NAME as ETV3_METHOD_NAME,
)
from evidence_transition_graphragv3_variable_flow.run_reader_qa import (  # noqa: E402
    main as run_etv3_reader_qa,
)
from evidence_transition_graphragv4_reader_context.contract import (  # noqa: E402
    DEFAULT_READER_CONTEXT_K,
    METHOD_CONTRACT,
    METHOD_NAME,
    QA_DOC_KEY,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def unique_ints(values: Iterable[Any], *, limit: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values or []:
        try:
            doc_index = int(value)
        except (TypeError, ValueError):
            continue
        if doc_index < 0 or doc_index in seen:
            continue
        seen.add(doc_index)
        output.append(doc_index)
        if len(output) >= max(int(limit), 1):
            break
    return output


def row_reader_context(row: Mapping[str, Any], *, reader_context_k: int) -> list[int]:
    route_trace = row.get("route_trace", {}) or {}
    candidate_universe = (
        route_trace.get("candidate_universe", {})
        if isinstance(route_trace, Mapping)
        else {}
    )
    candidate_docs = (
        candidate_universe.get("candidate_doc_indices", [])
        if isinstance(candidate_universe, Mapping)
        else []
    )
    frozen_top5 = row.get("retrieved_doc_indices_top5", []) or []
    return unique_ints(
        [*list(frozen_top5), *list(candidate_docs)],
        limit=max(int(reader_context_k), 1),
    )


def transform_retrieval_report(
    *,
    input_report: Path,
    output_report: Path,
    reader_context_k: int,
) -> Path:
    payload = load_json(input_report)
    rows = []
    for row in payload.get("rows", []) or []:
        copied = dict(row)
        context_docs = row_reader_context(
            copied,
            reader_context_k=max(int(reader_context_k), 1),
        )
        copied[QA_DOC_KEY] = context_docs
        # ETv3's reader runner reads this key; the first five remain ETv3 top5.
        copied["retrieved_doc_indices_top5"] = context_docs
        copied["reader_context_doc_indices"] = context_docs
        copied["reader_context_policy"] = "frozen_top5_then_candidate_prefix_unique"
        copied["reader_context_k"] = max(int(reader_context_k), 1)
        rows.append(copied)

    config = {
        **dict(payload.get("config", {}) or {}),
        **dict(METHOD_CONTRACT),
        "reader_context_k": max(int(reader_context_k), 1),
    }
    transformed = {
        **dict(payload),
        "method": METHOD_NAME,
        "base_method": str(payload.get("method") or ETV3_METHOD_NAME),
        "config": config,
        "reader_context_diagnostic": {
            "policy": "frozen_top5_then_candidate_prefix_unique",
            "reader_context_k": max(int(reader_context_k), 1),
            "top5_retrieval_is_frozen": True,
        },
        "rows": rows,
    }
    write_json(output_report, transformed)
    return output_report


def parse_reports(value: str) -> list[Path]:
    return [Path(item.strip()).expanduser() for item in str(value).split(",") if item.strip()]


def postprocess_qa_output(*, output_json: Path, output_md: Path | None) -> None:
    payload = load_json(output_json)
    payload["method"] = METHOD_NAME
    payload["format"] = "etv4_reader_context_qa"
    payload["qa_doc_key"] = QA_DOC_KEY
    payload["base_method"] = ETV3_METHOD_NAME
    for dataset in payload.get("datasets", []) or []:
        source = dataset.get("source", {}) or {}
        if isinstance(source, dict):
            source["doc_key"] = QA_DOC_KEY
            source["base_reader_runner"] = "etv3_reader_only"
            source["top5_retrieval_is_frozen"] = True
        methods = dataset.get("methods", {}) or {}
        if ETV3_METHOD_NAME in methods:
            method_payload = methods.pop(ETV3_METHOD_NAME)
            method_payload["method"] = METHOD_NAME
            methods[METHOD_NAME] = method_payload
    write_json(output_json, payload)
    if output_md is not None and output_md.exists():
        text = output_md.read_text(encoding="utf-8")
        text = text.replace(ETV3_METHOD_NAME, METHOD_NAME)
        output_md.write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-reports", required=True)
    parser.add_argument("--reader-context-k", type=int, default=DEFAULT_READER_CONTEXT_K)
    parser.add_argument("--qa-top-k", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--save-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, passthrough = parser.parse_known_args(argv)
    output_json = Path(args.output_json).expanduser()
    output_md = Path(args.output_md).expanduser() if str(args.output_md).strip() else None
    transformed_dir = output_json.parent / "v4_reader_context_inputs"
    transformed_reports = []
    for report_path in parse_reports(str(args.retrieval_reports)):
        transformed_reports.append(
            transform_retrieval_report(
                input_report=report_path,
                output_report=transformed_dir
                / f"{report_path.stem}_reader_context_k{max(int(args.reader_context_k), 1)}.json",
                reader_context_k=max(int(args.reader_context_k), 1),
            )
        )

    qa_top_k = int(args.qa_top_k) if int(args.qa_top_k) > 0 else max(int(args.reader_context_k), 1)
    etv3_args = [
        "--retrieval-reports",
        ",".join(str(path) for path in transformed_reports),
        "--qa-top-k",
        str(qa_top_k),
        "--output-json",
        str(output_json),
        "--save-dir",
        str(Path(args.save_dir).expanduser()),
    ]
    if output_md is not None:
        etv3_args.extend(["--output-md", str(output_md)])
    etv3_args.extend(passthrough)
    run_etv3_reader_qa(etv3_args)
    postprocess_qa_output(output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
