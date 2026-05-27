"""Export vocab-strict GraphRAG retrieval rows for the shared QA runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from evidence_transition_graphrag.contract import (
    LEGACY_METHOD_NAMES,
    LEGACY_QA_DOC_KEYS,
    METHOD_NAME as EVIDENCE_TRANSITION_QA_METHOD,
    METHOD_V2_NAME as EVIDENCE_TRANSITION_V2_QA_METHOD,
    QA_DOC_KEY as EVIDENCE_TRANSITION_QA_DOC_KEY,
    QA_V2_DOC_KEY as EVIDENCE_TRANSITION_V2_QA_DOC_KEY,
)
from .contract import CANONICAL_CLEAN_METHOD_NAME

QA_RUNNER_PUBLIC_METHOD = "source_authorized_vocab_strict_graphrag"
QUERY_GROUNDED_STO_QA_METHOD = EVIDENCE_TRANSITION_QA_METHOD
QUERY_GROUNDED_STO_V2_QA_METHOD = EVIDENCE_TRANSITION_V2_QA_METHOD
QUERY_GROUNDED_STO_LEGACY_QA_METHOD = "query_grounded_sto_graphrag"
LEGACY_QA_DOC_KEY = "source_authorized_vocab_strict_doc_indices_top5"
QUERY_GROUNDED_STO_QA_DOC_KEY = "query_grounded_sto_doc_indices_top5"


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def export_qa_transition_report(
    *,
    input_jsons: Sequence[Path],
    output_json: Path,
    public_method: str = QA_RUNNER_PUBLIC_METHOD,
) -> Dict[str, Any]:
    datasets = []
    for input_json in input_jsons:
        payload = load_json(input_json)
        rows = []
        for row in payload.get("rows", []) or []:
            copied = dict(row)
            top5 = list(row.get("retrieved_doc_indices_top5", []) or [])[:5]
            copied[EVIDENCE_TRANSITION_QA_DOC_KEY] = top5
            copied[EVIDENCE_TRANSITION_V2_QA_DOC_KEY] = top5
            copied[QUERY_GROUNDED_STO_QA_DOC_KEY] = top5
            copied[LEGACY_QA_DOC_KEY] = top5
            rows.append(copied)
        datasets.append(
            {
                "dataset": str(payload.get("dataset") or ""),
                "report_path": str(payload.get("input_report") or ""),
                "openie_path": str(payload.get("openie_path") or ""),
                "rows": rows,
                "source_retrieval_json": str(input_json),
                "metrics": dict(payload.get("metrics", {}) or {}),
            }
        )

    clean_public_method = str(public_method)
    canonical_method = (
        clean_public_method
        if clean_public_method
        in {EVIDENCE_TRANSITION_QA_METHOD, EVIDENCE_TRANSITION_V2_QA_METHOD}
        else CANONICAL_CLEAN_METHOD_NAME
    )
    public_doc_key = (
        EVIDENCE_TRANSITION_V2_QA_DOC_KEY
        if clean_public_method == EVIDENCE_TRANSITION_V2_QA_METHOD
        else EVIDENCE_TRANSITION_QA_DOC_KEY
    )
    output = {
        "method": clean_public_method,
        "legacy_method_alias": QA_RUNNER_PUBLIC_METHOD,
        "legacy_method_aliases": [QA_RUNNER_PUBLIC_METHOD, *LEGACY_METHOD_NAMES],
        "legacy_doc_key_aliases": list(LEGACY_QA_DOC_KEYS),
        "public_doc_key": public_doc_key,
        "canonical_method": canonical_method,
        "legacy_canonical_method_alias": CANONICAL_CLEAN_METHOD_NAME,
        "format": "transition_top5_qa_input",
        "datasets": datasets,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = export_qa_transition_report(
        input_jsons=[Path(path) for path in args.input_json],
        output_json=Path(args.output_json),
    )
    print(json.dumps({"datasets": [row["dataset"] for row in payload["datasets"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
