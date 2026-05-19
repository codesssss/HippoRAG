"""Compatibility runner for :mod:`evidenceflow.frozen_etv3_variable_flow.run_reader_qa`."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidenceflow.frozen_etv3_variable_flow.run_reader_qa import *  # noqa: F401,F403
from evidenceflow.frozen_etv3_variable_flow.run_reader_qa import main


if __name__ == "__main__":
    raise SystemExit(main())
