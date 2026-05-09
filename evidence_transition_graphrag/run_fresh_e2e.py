#!/usr/bin/env python3
"""Run Evidence Transition GraphRAG from fresh indexing artifacts.

This is a thin public wrapper around the current fresh end-to-end runner. It
keeps the method package self-contained while avoiding a risky code fork.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

_PACKAGE_DIR = Path(__file__).resolve().parent
_OVERLAY_DIR = _PACKAGE_DIR / "portable_overlay"
if _OVERLAY_DIR.exists() and str(_OVERLAY_DIR) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_DIR))

from run_query_grounded_sto_fresh_e2e import main as _run_fresh_e2e


def main(argv: Sequence[str] | None = None) -> int:
    return _run_fresh_e2e(argv)


if __name__ == "__main__":
    raise SystemExit(main())
