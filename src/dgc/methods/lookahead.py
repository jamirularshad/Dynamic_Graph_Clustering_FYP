"""Explicit interface for methods that need snapshot t+1 to produce output for t.

This is not a causal method: the output for t becomes available only when t+1 has
arrived (or when the stream ends). The adapter records that availability time so
operational delay can be measured (E2).
"""

from dataclasses import dataclass
from typing import Protocol

from dgc.data.types import GraphSnapshot
from dgc.methods.base import StepResult


class LookaheadCore(Protocol):
    def process(self, cur: GraphSnapshot, nxt: GraphSnapshot | None) -> StepResult: ...


@dataclass(frozen=True)
class DelayedOutput:
    """Output for `snapshot`, available once snapshot `available_at` has arrived."""

    snapshot: GraphSnapshot
    result: StepResult
    available_at: int


class LookaheadAdapter:
    """Buffers one snapshot; `step` returns the output for the previous one, if any.

    Call `finish` after the last snapshot: the final snapshot is processed without
    t+1, as the DyG-MF paper specifies for t = tau.
    """

    def __init__(self, core: LookaheadCore):
        self.core = core
        self._pending: GraphSnapshot | None = None

    def step(self, snapshot: GraphSnapshot) -> DelayedOutput | None:
        pending, self._pending = self._pending, snapshot
        if pending is None:
            return None
        return DelayedOutput(pending, self.core.process(pending, snapshot), snapshot.t)

    def finish(self) -> DelayedOutput | None:
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        return DelayedOutput(pending, self.core.process(pending, None), pending.t)
