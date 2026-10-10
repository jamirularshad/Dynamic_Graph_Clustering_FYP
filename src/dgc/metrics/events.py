"""Event recovery metrics.

A detected event matches a planted event when they have the same kind, the detection
is not earlier than the planted time and at most `tolerance` snapshots later, and the
Jaccard overlap of their involved node sets (union of `before` and `after`) is at
least `min_overlap`. Matching is one-to-one and greedy: smallest delay first, then
largest overlap. Precision is undefined (None) with no detections of a kind, recall
with no planted events of a kind; F1 is None when either is.
"""

from dataclasses import dataclass

from dgc.data.types import EVENT_KINDS, CommunityEvent
from dgc.tracking.tracker import jaccard


@dataclass(frozen=True)
class KindScores:
    """Matching outcome for one event kind."""

    kind: str
    planted: int
    detected: int
    true_positives: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_delay: float | None


def involved_nodes(event: CommunityEvent) -> frozenset[int]:
    """All nodes in the communities before and after the event."""
    return frozenset().union(*event.before, *event.after)


def match_events(
    detected: list[CommunityEvent],
    planted: list[CommunityEvent],
    tolerance: int,
    min_overlap: float,
) -> list[KindScores]:
    """Per-kind scores for the five tracked kinds. "complex" detections never match."""
    return [
        _score_kind(
            kind,
            [e for e in detected if e.kind == kind],
            [e for e in planted if e.kind == kind],
            tolerance,
            min_overlap,
        )
        for kind in EVENT_KINDS
    ]


def _score_kind(kind, detected, planted, tolerance, min_overlap) -> KindScores:
    det_nodes = [involved_nodes(e) for e in detected]
    pl_nodes = [involved_nodes(e) for e in planted]
    candidates = []
    for i, d in enumerate(detected):
        for j, p in enumerate(planted):
            delay = d.t - p.t
            if not 0 <= delay <= tolerance:
                continue
            overlap = jaccard(det_nodes[i], pl_nodes[j])
            if overlap >= min_overlap:
                candidates.append((delay, -overlap, i, j))

    used_d: set[int] = set()
    used_p: set[int] = set()
    delays = []
    for delay, _, i, j in sorted(candidates):
        if i in used_d or j in used_p:
            continue
        used_d.add(i)
        used_p.add(j)
        delays.append(delay)

    tp = len(delays)
    precision = tp / len(detected) if detected else None
    recall = tp / len(planted) if planted else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return KindScores(
        kind=kind,
        planted=len(planted),
        detected=len(detected),
        true_positives=tp,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_delay=sum(delays) / tp if tp else None,
    )


def false_events_per_transition(detected: list[CommunityEvent], n_transitions: int) -> float:
    """Non-continuation detections per transition; use on sequences with no planted events."""
    if n_transitions <= 0:
        raise ValueError("need at least one transition")
    return sum(e.kind != "continuation" for e in detected) / n_transitions
