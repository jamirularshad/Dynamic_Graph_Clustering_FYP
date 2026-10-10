"""Loaders for external benchmarks."""

from pathlib import Path

import numpy as np
import pandas as pd

from dgc.data.types import CommunityEvent, EvalAnnotations, GraphSnapshot, canonical_edges


def read_timeline(path: Path) -> dict[str, dict[int, int]]:
    """Parse a Green-benchmark timeline: dynamic community -> {snapshot: line in .comm}.

    Each line looks like "M500:2=499,3=498,..." (1-based snapshots and lines).
    """
    timeline: dict[str, dict[int, int]] = {}
    for line in path.read_text().splitlines():
        name, _, body = line.partition(":")
        entries = [e for e in body.strip().rstrip(",").split(",") if e]
        timeline[name] = {int(t): int(idx) for t, idx in (e.split("=") for e in entries)}
    return timeline


def load_birthdeath(
    directory: Path, n_communities: int, seed: int, prefix: str = "birthdeath"
) -> tuple[list[GraphSnapshot], EvalAnnotations]:
    """Induced subsample of a Green Birth-Death sequence over random tracked communities.

    `n_communities` dynamic communities are drawn (seeded) from the timeline. At each
    snapshot the node set is the members of the drawn communities alive then, edges
    are those of the original snapshot with both endpoints in that set, and reference
    labels are the dynamic community identities. Raw node IDs are remapped once to
    contiguous integers. Snapshot t in the files (1-based) becomes t - 1.
    Planted events: continuation between consecutive alive snapshots, birth at the
    first snapshot after the first one, death after the last alive snapshot.
    """
    timeline = read_timeline(directory / f"{prefix}.timeline")
    times = sorted({t for spans in timeline.values() for t in spans})
    rng = np.random.default_rng(seed)
    names = sorted(timeline)
    chosen = sorted(
        str(n) for n in rng.choice(names, size=min(n_communities, len(names)), replace=False)
    )

    raw_members: dict[int, dict[str, frozenset[int]]] = {}
    for t in times:
        lines = (directory / f"{prefix}.t{t:02d}.comm").read_text().splitlines()
        raw_members[t] = {
            name: frozenset(int(x) for x in lines[timeline[name][t] - 1].split())
            for name in chosen
            if t in timeline[name]
        }

    all_raw = sorted({n for per_t in raw_members.values() for m in per_t.values() for n in m})
    remap = {raw: i for i, raw in enumerate(all_raw)}
    community_id = {name: i for i, name in enumerate(chosen)}
    members = {
        t: {name: frozenset(remap[n] for n in m) for name, m in per_t.items()}
        for t, per_t in raw_members.items()
    }

    snapshots, labels = [], {}
    for t in times:
        alive = members[t]
        raw_nodes = {n for m in raw_members[t].values() for n in m}
        nodes = frozenset(remap[n] for n in raw_nodes)
        frame = pd.read_csv(
            directory / f"{prefix}.t{t:02d}.edges", sep=r"\s+", header=None, dtype=np.int64
        )
        keep = sorted(raw_nodes)
        mask = frame[0].isin(keep) & frame[1].isin(keep)
        pairs = ((remap[u], remap[v]) for u, v in frame.loc[mask, [0, 1]].itertuples(index=False))
        snapshots.append(GraphSnapshot(t=t - 1, nodes=nodes, edges=canonical_edges(pairs, nodes)))
        labels[t - 1] = {n: community_id[name] for name, m in alive.items() for n in m}

    events: list[CommunityEvent] = []
    for name in chosen:
        alive_times = [t for t in times if t in members and name in members[t]]
        first, last = alive_times[0], alive_times[-1]
        if first != times[0]:
            events.append(CommunityEvent(first - 1, "birth", after=(members[first][name],)))
        for a, b in zip(alive_times, alive_times[1:], strict=False):
            if b == a + 1:
                events.append(
                    CommunityEvent(
                        b - 1, "continuation", before=(members[a][name],), after=(members[b][name],)
                    )
                )
        if last != times[-1]:
            events.append(CommunityEvent(last, "death", before=(members[last][name],)))
    events.sort(key=lambda e: (e.t, e.kind))
    return snapshots, EvalAnnotations(labels=labels, events=events)
