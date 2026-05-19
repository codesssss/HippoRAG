"""Requirement providers for native PCEC readout."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dtc_embed_utils import DTCRequirement  # noqa: E402

from evidenceflow.readout import read_json
from evidenceflow.pcec_types import PCECQueryState


def requirement_from_trace(payload: Mapping[str, Any]) -> DTCRequirement:
    return DTCRequirement(
        unit_id=str(payload.get("unit_id") or payload.get("id") or ""),
        subquery=str(payload.get("subquery") or ""),
        depends_on=tuple(str(item) for item in list(payload.get("depends_on") or [])),
        expected_answer_type=str(payload.get("expected_answer_type") or "unknown"),
        anchor_mentions=tuple(str(item) for item in list(payload.get("anchor_mentions") or [])),
        role=str(payload.get("role") or "support"),
        satisfiable_by=str(payload.get("satisfiable_by") or "unknown"),
    )


class FrozenReportRequirementProvider:
    """Load DBEC decomposition requirements from an existing selector report."""

    def __init__(self, report_path: str | Path, *, allow_missing: bool = False) -> None:
        self.report_path = Path(report_path)
        self.allow_missing = bool(allow_missing)
        self.requirements_by_question = self._load_requirements(self.report_path)

    @staticmethod
    def _load_requirements(report_path: Path) -> dict[str, list[DTCRequirement]]:
        payload = read_json(report_path)
        cache: dict[str, list[DTCRequirement]] = {}
        for trace in list(payload.get("setwise_selector_query_traces") or []):
            if not isinstance(trace, Mapping):
                continue
            question = str(trace.get("question") or "").strip()
            selector_trace = trace.get("selector_trace")
            if not isinstance(selector_trace, Mapping):
                continue
            requirements = [
                requirement_from_trace(item)
                for item in list(selector_trace.get("requirements") or [])
                if isinstance(item, Mapping) and str(item.get("subquery") or "").strip()
            ]
            if question:
                cache[question] = requirements
        return cache

    def get_requirements(self, state: PCECQueryState) -> list[DTCRequirement]:
        key = str(state.question or "").strip()
        if key in self.requirements_by_question:
            return list(self.requirements_by_question[key])
        if self.allow_missing:
            return []
        raise KeyError(f"No frozen requirements for query_idx={state.query_idx} question={key!r}")

    def summary(self) -> dict[str, Any]:
        return {
            "provider": "frozen_report",
            "report_path": str(self.report_path),
            "query_count": int(len(self.requirements_by_question)),
            "allow_missing": bool(self.allow_missing),
        }


def requirement_subqueries(requirements: Sequence[DTCRequirement]) -> list[str]:
    return [str(req.subquery or "") for req in requirements if str(req.subquery or "").strip()]
