"""Compatibility import for source-layout AG-STO proposal role metadata."""

from src.agsto.proposal_roles import (
    build_proposal_role_records,
    build_proposal_role_units,
    covered_role_unit_counts,
    proposal_role_summary_by_doc,
)

__all__ = [
    "build_proposal_role_records",
    "build_proposal_role_units",
    "covered_role_unit_counts",
    "proposal_role_summary_by_doc",
]
