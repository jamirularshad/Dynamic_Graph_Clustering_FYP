import pytest

from dgc.data.stream import OnlineStream, lifecycle
from dgc.data.synthetic import ScheduledEvent, SyntheticConfig, generate
from dgc.data.types import GraphSnapshot, canonical_edges
from dgc.metrics.partition import communities


def snap(t, nodes, edges=()):
    nodes = frozenset(nodes)
    return GraphSnapshot(t=t, nodes=nodes, edges=canonical_edges(edges, nodes))


def test_canonical_edges_orders_and_deduplicates():
    nodes = frozenset({1, 2, 3})
    assert canonical_edges([(2, 1), (1, 2), (3, 3), (3, 2)], nodes) == {(1, 2), (2, 3)}


def test_canonical_edges_rejects_inactive_endpoint():
    with pytest.raises(ValueError):
        canonical_edges([(1, 9)], frozenset({1, 2}))


def test_lifecycle_tracks_arrival_departure_and_reappearance():
    s0, s1, s2 = snap(0, {1, 2, 3}), snap(1, {2, 3, 4}), snap(2, {1, 2, 3, 4})
    assert lifecycle(None, s0).arrived == {1, 2, 3}
    first = lifecycle(s0, s1)
    assert (first.arrived, first.departed) == ({4}, {1})
    assert lifecycle(s1, s2).arrived == {1}  # reappearing node keeps its ID


def test_isolated_active_node_is_kept():
    s = snap(0, {1, 2, 3}, [(1, 2)])
    assert 3 in s.nodes


def test_stream_requires_increasing_time():
    with pytest.raises(ValueError):
        OnlineStream([snap(1, {1}), snap(0, {1})])


CONFIG = SyntheticConfig(
    n_communities=4,
    community_size=20,
    n_snapshots=6,
    p_in=0.4,
    p_out=0.02,
    move_fraction=0.05,
    events=[
        ScheduledEvent(1, "birth"),
        ScheduledEvent(2, "merge"),
        ScheduledEvent(3, "split"),
        ScheduledEvent(4, "death"),
    ],
)


def test_generator_is_deterministic_per_seed():
    a = generate(CONFIG, seed=7)
    b = generate(CONFIG, seed=7)
    c = generate(CONFIG, seed=8)
    assert a == b
    assert a[0] != c[0]


def test_event_log_matches_reference_labels():
    snapshots, ann = generate(CONFIG, seed=7)
    for s in snapshots:
        assert set(ann.labels[s.t]) == s.nodes
    for e in ann.events:
        prev = set(communities(ann.labels[e.t - 1]))
        cur = set(communities(ann.labels[e.t]))
        assert all(c in prev for c in e.before)
        assert all(c in cur for c in e.after)
    kinds = [e.kind for e in ann.events if e.kind != "continuation"]
    assert kinds == ["birth", "merge", "split", "death"]


def test_birth_adds_new_nodes_and_death_removes_them():
    snapshots, ann = generate(CONFIG, seed=7)
    birth = next(e for e in ann.events if e.kind == "birth")
    assert not (birth.after[0] & snapshots[0].nodes)
    death = next(e for e in ann.events if e.kind == "death")
    assert not (death.before[0] & snapshots[death.t].nodes)


def test_generator_rejects_bad_schedule():
    bad = SyntheticConfig(2, 5, 3, 0.5, 0.0, events=[ScheduledEvent(0, "birth")])
    with pytest.raises(ValueError):
        generate(bad, seed=0)
