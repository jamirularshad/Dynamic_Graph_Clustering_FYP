"""Core data types.

Methods receive only `GraphSnapshot`s. `EvalAnnotations` holds reference labels and
planted events and is passed only to evaluation code, so ground truth cannot reach
a clustering method through its input.
"""

from dataclasses import dataclass, field

EVENT_KINDS = ("continuation", "birth", "death", "merge", "split")
"""The five tracked event classes. The tracker may also emit "complex" for
many-to-many transitions, which is reported separately from these five."""

Partition = dict[int, int]
"""Node ID -> community label. Labels are arbitrary integers; only grouping matters."""


@dataclass(frozen=True)
class GraphSnapshot:
    """Undirected, unweighted graph at time t over stable integer node IDs.

    `nodes` is the active node set (it may include isolated nodes). Every edge is a
    canonical pair (u, v) with u < v and both endpoints in `nodes`.
    """

    t: int
    nodes: frozenset[int]
    edges: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class CommunityEvent:
    """A community life-cycle event on the transition from t-1 to t.

    `before` holds the communities at t-1 involved in the event, `after` those at t.
    A birth has an empty `before`; a death has an empty `after`. The same type is
    used for planted (ground-truth) and detected events; callers keep the roles apart.
    """

    t: int
    kind: str
    before: tuple[frozenset[int], ...] = ()
    after: tuple[frozenset[int], ...] = ()


@dataclass(frozen=True)
class EvalAnnotations:
    """Reference membership per snapshot and planted events for one sequence."""

    labels: dict[int, Partition]
    events: list[CommunityEvent] = field(default_factory=list)


def canonical_edges(pairs, nodes: frozenset[int]) -> frozenset[tuple[int, int]]:
    """Canonicalise raw pairs: order each as (min, max), drop self-loops and duplicates.

    Raises ValueError if an endpoint is not in `nodes`.
    """
    edges = set()
    for u, v in pairs:
        if u == v:
            continue
        if u not in nodes or v not in nodes:
            raise ValueError(f"edge ({u}, {v}) has an endpoint outside the active node set")
        edges.add((u, v) if u < v else (v, u))
    return frozenset(edges)
