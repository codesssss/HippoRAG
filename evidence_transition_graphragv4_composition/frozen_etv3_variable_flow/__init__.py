"""Compatibility shim for :mod:`evidenceflow.frozen_etv3_variable_flow`."""

from __future__ import annotations

from pathlib import Path

import evidenceflow.frozen_etv3_variable_flow as _frozen

__path__ = [str(Path(__file__).resolve().parent), *list(_frozen.__path__)]
