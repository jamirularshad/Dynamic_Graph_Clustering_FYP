import pytest

from dgc.tracking.tracker import Tracker, classify_transition, jaccard

A = frozenset(range(0, 10))
B = frozenset(range(10, 20))
C = frozenset(range(20, 30))


def part(*groups, offset=0):
    """Partition with arbitrary labels starting at `offset`."""
    return {n: offset + i for i, g in enumerate(groups) for n in g}


def kinds(events):
    return sorted(e.kind for e in events)


def run(prev, cur):
    tracker = Tracker(threshold=0.3)
    _, ids_prev = tracker.step(0, prev)
    events, ids_cur = tracker.step(1, cur)
    return events, ids_prev, ids_cur


def test_jaccard():
    assert jaccard(A, A) == 1.0
    assert jaccard(A, B) == 0.0
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_classify_transition():
    assert classify_transition(1, 1) == "continuation"
    assert classify_transition(0, 1) == "birth"
    assert classify_transition(1, 0) == "death"
    assert classify_transition(3, 1) == "merge"
    assert classify_transition(1, 2) == "split"
    assert classify_transition(2, 2) == "complex"
    with pytest.raises(ValueError):
        classify_transition(0, 0)


def test_continuation_keeps_identity_despite_relabelling_and_small_change():
    moved = (A - {9}) | {10}
    events, ids0, ids1 = run(part(A, B), part(B - {10}, moved, offset=50))
    assert kinds(events) == ["continuation", "continuation"]
    assert ids1[0] == ids0[0] and ids1[15] == ids0[15]


def test_birth():
    events, _, _ = run(part(A), part(A, C))
    assert kinds(events) == ["birth", "continuation"]


def test_death():
    events, _, _ = run(part(A, B), part(A))
    assert kinds(events) == ["continuation", "death"]


def test_merge_gets_fresh_identity():
    events, ids0, ids1 = run(part(A, B), part(A | B))
    (merge,) = events
    assert merge.kind == "merge" and set(merge.before) == {A, B}
    assert ids1[0] not in {ids0[0], ids0[10]}


def test_split():
    halves = (frozenset(range(5)), frozenset(range(5, 10)))
    events, _, _ = run(part(A), part(*halves))
    (split,) = events
    assert split.kind == "split" and set(split.after) == set(halves)


def test_many_to_many_is_complex():
    c = frozenset(range(0, 5)) | frozenset(range(10, 15))
    d = frozenset(range(5, 10)) | frozenset(range(15, 20))
    events, _, _ = run(part(A, B), part(c, d))
    assert kinds(events) == ["complex"]


def test_state_round_trip():
    tracker = Tracker(threshold=0.3)
    tracker.step(0, part(A, B))
    clone = Tracker(threshold=0.3)
    clone.set_state(tracker.get_state())
    assert tracker.step(1, part(A | B)) == clone.step(1, part(A | B))
