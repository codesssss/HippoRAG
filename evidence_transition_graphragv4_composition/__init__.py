"""Compatibility shim for the renamed :mod:`evidenceflow` package.

New code should import :mod:`evidenceflow` directly.  This package keeps legacy
imports such as ``evidence_transition_graphragv4_composition.expander`` and
deep frozen-snapshot imports working while experiment scripts are migrated.
"""

from __future__ import annotations

from pathlib import Path

import evidenceflow as _evidenceflow

__path__ = [str(Path(__file__).resolve().parent), *list(_evidenceflow.__path__)]
