"""Interface shared by all online clustering methods."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from dgc.data.types import GraphSnapshot, Partition


@dataclass
class StepResult:
    """Output of one online step: the partition at t plus update accounting.

    `updated_nodes` holds every node whose state was recomputed at t, whether or not
    its label changed. `selected` is the subset chosen by a change score (empty for
    methods that do not select). `timings` maps phase name to elapsed seconds.
    """

    labels: Partition
    updated_nodes: frozenset[int]
    timings: dict[str, float] = field(default_factory=dict)
    selected: frozenset[int] = frozenset()


class OnlineClusterer(Protocol):
    """A causal clustering method: `step` sees only the current snapshot and its own state.

    Historical state must be updated only after the output for t is computed.
    `get_state`/`set_state` capture everything needed to resume exactly, including
    any random-generator state.
    """

    def step(self, snapshot: GraphSnapshot) -> StepResult: ...

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, state: dict[str, Any]) -> None: ...
