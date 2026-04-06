import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hipporag.utils.causal_utils import derive_directed_structure_edge


def test_general_factual_v2_keeps_legacy_mode_unchanged_for_located_in():
    assert derive_directed_structure_edge(
        "Drexel Heights",
        "is located in",
        "Pima County",
        relation_probe_mode="general_factual",
    ) is None


def test_general_factual_v2_adds_containment_and_reference_edges():
    assert derive_directed_structure_edge(
        "Drexel Heights",
        "is located in",
        "Pima County",
        relation_probe_mode="general_factual_v2",
    ) == (
        "drexel heights",
        "pima county",
        "factual_part_of_or_contains",
        0.8,
    )
    assert derive_directed_structure_edge(
        "ISO 3166-2:CV",
        "is the entry for",
        "Cabo Verde",
        relation_probe_mode="general_factual_v2",
    ) == (
        "iso 3166 2 cv",
        "cabo verde",
        "factual_part_of_or_contains",
        0.8,
    )


def test_general_factual_v2_adds_alias_edges():
    assert derive_directed_structure_edge(
        "Cape Verde",
        "is also known as",
        "Cabo Verde",
        relation_probe_mode="general_factual_v2",
    ) == (
        "cape verde",
        "cabo verde",
        "factual_alias_or_name",
        0.78,
    )
