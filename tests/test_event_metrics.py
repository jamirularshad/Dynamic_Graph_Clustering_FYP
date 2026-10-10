import pytest

from dgc.data.types import CommunityEvent
from dgc.metrics.events import false_events_per_transition, match_events

A = frozenset(range(10))
B = frozenset(range(10, 20))


def scores_for(kind, detected, planted, tolerance=1, min_overlap=0.5):
    return next(
        s for s in match_events(detected, planted, tolerance, min_overlap) if s.kind == kind
    )


def test_exact_match():
    ev = CommunityEvent(3, "merge", before=(A, B), after=(A | B,))
    s = scores_for("merge", [ev], [ev])
    assert (s.precision, s.recall, s.f1, s.mean_delay) == (1.0, 1.0, 1.0, 0.0)


def test_late_detection_within_tolerance_records_delay():
    planted = CommunityEvent(3, "death", before=(A,))
    detected = CommunityEvent(4, "death", before=(A,))
    assert scores_for("death", [detected], [planted]).mean_delay == 1.0


def test_detection_outside_tolerance_or_early_is_missed():
    planted = CommunityEvent(3, "death", before=(A,))
    for t in (2, 5):
        s = scores_for("death", [CommunityEvent(t, "death", before=(A,))], [planted])
        assert s.true_positives == 0 and s.f1 == 0.0


def test_low_overlap_does_not_match():
    planted = CommunityEvent(3, "birth", after=(A,))
    detected = CommunityEvent(3, "birth", after=(B,))
    assert scores_for("birth", [detected], [planted]).true_positives == 0


def test_matching_is_one_to_one():
    planted = CommunityEvent(3, "birth", after=(A,))
    detected = [CommunityEvent(3, "birth", after=(A,)), CommunityEvent(3, "birth", after=(A,))]
    s = scores_for("birth", detected, [planted])
    assert (s.true_positives, s.precision, s.recall) == (1, 0.5, 1.0)


def test_undefined_scores_are_none():
    s = scores_for("split", [], [])
    assert s.precision is None and s.recall is None and s.f1 is None
    s = scores_for("split", [], [CommunityEvent(1, "split", before=(A,))])
    assert s.precision is None and s.recall == 0.0 and s.f1 is None


def test_false_events_per_transition():
    detected = [
        CommunityEvent(1, "continuation", before=(A,), after=(A,)),
        CommunityEvent(2, "birth", after=(B,)),
        CommunityEvent(2, "complex"),
    ]
    assert false_events_per_transition(detected, n_transitions=4) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        false_events_per_transition(detected, 0)
