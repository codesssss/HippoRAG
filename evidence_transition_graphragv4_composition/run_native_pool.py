"""Compatibility runner for :mod:`evidenceflow.run_native_pool`."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidenceflow.run_native_pool import *  # noqa: F401,F403
from evidenceflow.run_native_pool import main


if __name__ == "__main__":
    raise SystemExit(main())
