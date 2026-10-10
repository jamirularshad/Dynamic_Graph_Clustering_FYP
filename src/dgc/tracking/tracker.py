"""Community tracking across snapshots.

Successive partitions are linked by a many-to-many overlap graph: community A at t-1
and community B at t are linked when jaccard(A, B) >= threshold. Each connected
component of that bipartite graph is one transition, classified by how many old and
new communities it contains. Identities are tracker-assigned integers, independent
of the arbitrary labels a clustering method emits: a continuation keeps its identity,
every other new community receives a fresh one.
"""

from typing import Any

from dgc.data.types import CommunityEvent, Partition
from dgc.metrics.partition import communities


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """|a & b| / |a | b|; 0.0 when both are empty."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def classify_transition(n_before: int, n_after: int) -> str:
    """Event kind for a component with `n_before` old and `n_after` new communities."""
    if n_before == 1 and n_after == 1:
        return "continuation"
    if n_before == 0 and n_after == 1:
        return "birth"
    if n_before == 1 and n_after == 0:
        return "death"
    if n_before >= 2 and n_after == 1:
        return "merge"
    if n_before == 1 and n_after >= 2:
        return "split"
    if n_before >= 2 and n_after >= 2:
        return "complex"
    raise ValueError(f"no transition with {n_before} before and {n_after} after")


class Tracker:
    """Online community tracker; call `step` once per snapshot, in time order."""

    def __init__(self, threshold: float):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self._prev: dict[int, frozenset[int]] = {}
        self._next_id = 0
        self._t: int | None = None

    def _fresh_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def step(self, t: int, partition: Partition) -> tuple[list[CommunityEvent], dict[int, int]]:
        """Match `partition` at t to the previous one.

        Returns the events on the transition into t (none at the first call) and the
        tracked identity of every node at t.
        """
        current = communities(partition)
        if self._t is None:
            ids = {self._fresh_id(): members for members in current}
            self._prev, self._t = ids, t
            return [], _node_identities(ids)

        old = sorted(self._prev.items())
        components = _components(
            [m for _, m in old], current, lambda a, b: jaccard(a, b) >= self.threshold
        )
        events: list[CommunityEvent] = []
        ids: dict[int, frozenset[int]] = {}
        for old_idx, new_idx in components:
            kind = classify_transition(len(old_idx), len(new_idx))
            if kind == "continuation":
                ids[old[old_idx[0]][0]] = current[new_idx[0]]
            else:
                for j in new_idx:
                    ids[self._fresh_id()] = current[j]
            events.append(
                CommunityEvent(
                    t,
                    kind,
                    before=tuple(old[i][1] for i in old_idx),
                    after=tuple(current[j] for j in new_idx),
                )
            )
        self._prev, self._t = ids, t
        return events, _node_identities(ids)

    def get_state(self) -> dict[str, Any]:
        return {"prev": dict(self._prev), "next_id": self._next_id, "t": self._t}

    def set_state(self, state: dict[str, Any]) -> None:
        self._prev = dict(state["prev"])
        self._next_id = state["next_id"]
        self._t = state["t"]


def _node_identities(ids: dict[int, frozenset[int]]) -> dict[int, int]:
    return {n: cid for cid, members in ids.items() for n in members}


def _components(old, new, linked) -> list[tuple[list[int], list[int]]]:
    """Connected components of the bipartite overlap graph, as (old indices, new indices).

    Components are ordered by their smallest old index, then smallest new index.
    """
    parent = list(range(len(old) + len(new)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(old):
        for j, b in enumerate(new):
            if a & b and linked(a, b):
                parent[find(i)] = find(len(old) + j)

    groups: dict[int, tuple[list[int], list[int]]] = {}
    for x in range(len(old) + len(new)):
        root = find(x)
        olds, news = groups.setdefault(root, ([], []))
        (olds if x < len(old) else news).append(x if x < len(old) else x - len(old))
    return sorted(
        groups.values(), key=lambda g: (min(g[0], default=1 << 62), min(g[1], default=1 << 62))
    )
