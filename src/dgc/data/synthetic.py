"""Planted-partition dynamic graph generator with a scheduled event log.

Each snapshot is a stochastic block model over the communities alive at t: an edge
appears with probability `p_in` inside a community and `p_out` across communities.
Between snapshots the generator applies scheduled events, then moves a fraction of
the nodes in uninvolved communities to another uninvolved community (gradual change).
Births bring new nodes; deaths remove their nodes from the graph.
"""

from dataclasses import dataclass, field

import numpy as np

from dgc.data.types import CommunityEvent, EvalAnnotations, GraphSnapshot, Partition

SCHEDULABLE_KINDS = ("birth", "death", "merge", "split")


@dataclass(frozen=True)
class ScheduledEvent:
    """An event to plant on the transition into snapshot t (t >= 1)."""

    t: int
    kind: str


@dataclass(frozen=True)
class SyntheticConfig:
    """Generator settings. `move_fraction` is the share of eligible nodes moved per step."""

    n_communities: int
    community_size: int
    n_snapshots: int
    p_in: float
    p_out: float
    move_fraction: float = 0.0
    events: list[ScheduledEvent] = field(default_factory=list)


def generate(config: SyntheticConfig, seed: int) -> tuple[list[GraphSnapshot], EvalAnnotations]:
    """Generate snapshots and the matching reference labels and planted events."""
    for ev in config.events:
        if ev.kind not in SCHEDULABLE_KINDS:
            raise ValueError(f"unknown event kind {ev.kind!r}")
        if not 1 <= ev.t < config.n_snapshots:
            raise ValueError(f"event time {ev.t} outside 1..{config.n_snapshots - 1}")

    rng = np.random.default_rng(seed)
    gen = _Generator(config, rng)
    snapshots = [gen.snapshot(0)]
    labels = {0: gen.partition()}
    events: list[CommunityEvent] = []
    for t in range(1, config.n_snapshots):
        events.extend(gen.advance(t))
        snapshots.append(gen.snapshot(t))
        labels[t] = gen.partition()
    return snapshots, EvalAnnotations(labels=labels, events=events)


class _Generator:
    def __init__(self, config: SyntheticConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.communities: dict[int, set[int]] = {}
        self.next_node = 0
        self.next_cid = 0
        for _ in range(config.n_communities):
            self._new_community(self._new_nodes(config.community_size))

    def _new_nodes(self, k: int) -> set[int]:
        nodes = set(range(self.next_node, self.next_node + k))
        self.next_node += k
        return nodes

    def _new_community(self, members: set[int]) -> int:
        cid = self.next_cid
        self.next_cid += 1
        self.communities[cid] = members
        return cid

    def _pick(self, eligible: list[int], k: int, kind: str) -> list[int]:
        if len(eligible) < k:
            raise ValueError(f"not enough uninvolved communities for a {kind}")
        return [int(c) for c in self.rng.choice(eligible, size=k, replace=False)]

    def partition(self) -> Partition:
        return {n: cid for cid, members in self.communities.items() for n in members}

    def advance(self, t: int) -> list[CommunityEvent]:
        prev = {cid: frozenset(m) for cid, m in self.communities.items()}
        involved: set[int] = set()
        events: list[CommunityEvent] = []

        for ev in (e for e in self.config.events if e.t == t):
            eligible = sorted(set(prev) & set(self.communities) - involved)
            if ev.kind == "birth":
                cid = self._new_community(self._new_nodes(self.config.community_size))
                involved.add(cid)
                events.append(CommunityEvent(t, "birth", after=(frozenset(self.communities[cid]),)))
            elif ev.kind == "death":
                (cid,) = self._pick(eligible, 1, "death")
                involved.add(cid)
                events.append(CommunityEvent(t, "death", before=(prev[cid],)))
                del self.communities[cid]
            elif ev.kind == "merge":
                a, b = self._pick(eligible, 2, "merge")
                involved.update((a, b))
                merged = self.communities.pop(a) | self.communities.pop(b)
                new = self._new_community(merged)
                involved.add(new)
                events.append(
                    CommunityEvent(
                        t, "merge", before=(prev[a], prev[b]), after=(frozenset(merged),)
                    )
                )
            else:  # split
                eligible = [c for c in eligible if len(self.communities[c]) >= 2]
                (cid,) = self._pick(eligible, 1, "split")
                involved.add(cid)
                members = sorted(self.communities.pop(cid))
                self.rng.shuffle(members)
                half = len(members) // 2
                parts = (set(members[:half]), set(members[half:]))
                for part in parts:
                    involved.add(self._new_community(part))
                events.append(
                    CommunityEvent(
                        t, "split", before=(prev[cid],), after=tuple(frozenset(p) for p in parts)
                    )
                )

        continuing = sorted(set(prev) & set(self.communities) - involved)
        self._move(continuing)
        for cid in continuing:
            events.append(
                CommunityEvent(
                    t,
                    "continuation",
                    before=(prev[cid],),
                    after=(frozenset(self.communities[cid]),),
                )
            )
        return events

    def _move(self, continuing: list[int]) -> None:
        """Move a fraction of nodes between continuing communities, keeping each non-empty."""
        if self.config.move_fraction <= 0 or len(continuing) < 2:
            return
        pool = sorted(n for cid in continuing for n in self.communities[cid])
        k = round(self.config.move_fraction * len(pool))
        owner = {n: cid for cid in continuing for n in self.communities[cid]}
        for n in self.rng.choice(pool, size=k, replace=False):
            n = int(n)
            src = owner[n]
            if len(self.communities[src]) <= 1:
                continue
            dst = int(self.rng.choice([c for c in continuing if c != src]))
            self.communities[src].remove(n)
            self.communities[dst].add(n)
            owner[n] = dst

    def snapshot(self, t: int) -> GraphSnapshot:
        nodes = np.array(sorted(n for m in self.communities.values() for n in m))
        part = self.partition()
        labels = np.array([part[int(n)] for n in nodes])
        same = labels[:, None] == labels[None, :]
        prob = np.where(same, self.config.p_in, self.config.p_out)
        draw = self.rng.random(prob.shape) < prob
        iu, ju = np.nonzero(np.triu(draw, k=1))
        edges = frozenset((int(nodes[i]), int(nodes[j])) for i, j in zip(iu, ju, strict=True))
        return GraphSnapshot(t=t, nodes=frozenset(int(n) for n in nodes), edges=edges)
