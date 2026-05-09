"""Non-overwriting QA adapter for Evidence Transition GraphRAG.

Retrieval is owned by the portable overlay. Reader QA should stay aligned with
the target repository, so this module loads the target repo's existing
``run_transition_top5_qa.py`` and only registers the Evidence Transition method
key before delegating to its ``main``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from evidence_transition_graphrag.contract import METHOD_NAME, QA_DOC_KEY


def _candidate_roots() -> list[Path]:
    this_file = Path(__file__).resolve()
    roots: list[Path] = [this_file.parent]
    roots.extend(Path(entry or ".").resolve() for entry in sys.path)

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _load_target_runner() -> ModuleType:
    this_file = Path(__file__).resolve()
    for root in _candidate_roots():
        candidate = (root / "run_transition_top5_qa.py").resolve()
        if not candidate.exists() or candidate == this_file:
            continue
        spec = importlib.util.spec_from_file_location(
            "_evidence_transition_target_run_transition_top5_qa",
            candidate,
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError(
        "Could not find target run_transition_top5_qa.py. "
        "Keep the target repository's QA runner, or run retrieval without --run-qa."
    )


_TARGET = _load_target_runner()
if not hasattr(_TARGET, "METHOD_DOC_KEYS"):
    raise AttributeError(
        "Target run_transition_top5_qa.py does not expose METHOD_DOC_KEYS, "
        "so the Evidence Transition QA method cannot be registered safely."
    )

_TARGET.METHOD_DOC_KEYS[METHOD_NAME] = QA_DOC_KEY

main = _TARGET.main


if __name__ == "__main__":
    raise SystemExit(main())
