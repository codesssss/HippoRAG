"""BSGS: Belief-State Graph Scratchpad utilities.

This package is intentionally additive and independent from the frozen DAEC
path.  DAEC composes a final evidence set posterior in one step; BSGS tracks a
tractable node-level marginal filtering approximation across oracle slots.
"""

from .slots import Slot
from .state import BeliefStateGraph, PropositionNode

__all__ = ["BeliefStateGraph", "PropositionNode", "Slot"]
