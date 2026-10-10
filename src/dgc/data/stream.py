"""Online access to a snapshot sequence.

Node lifecycle policy:
- A node is active at t iff it is in `GraphSnapshot.nodes`; isolated active nodes are kept.
- A node absent at t has vanished; it keeps its stable ID and may reappear later.
- `lifecycle` reports, for each transition, which nodes arrived (new or reappearing)
  and which departed, so methods can count new-node initialisation as mandatory work.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from dgc.data.types import GraphSnapshot


@dataclass(frozen=True)
class Lifecycle:
    """Node set changes on the transition into snapshot t."""

    arrived: frozenset[int]
    departed: frozenset[int]


def lifecycle(prev: GraphSnapshot | None, cur: GraphSnapshot) -> Lifecycle:
    """Nodes arriving and departing between `prev` and `cur`. `prev` is None at t = 0."""
    if prev is None:
        return Lifecycle(arrived=cur.nodes, departed=frozenset())
    return Lifecycle(arrived=cur.nodes - prev.nodes, departed=prev.nodes - cur.nodes)


class OnlineStream:
    """Yields snapshots one at a time, in time order.

    The stream itself is never handed to a method: the runner passes each yielded
    snapshot to `step`, so a method can only see the present and what it stored.
    """

    def __init__(self, snapshots: Sequence[GraphSnapshot]):
        times = [s.t for s in snapshots]
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError("snapshots must have strictly increasing t")
        self._snapshots = list(snapshots)

    def __iter__(self) -> Iterator[GraphSnapshot]:
        return iter(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)
