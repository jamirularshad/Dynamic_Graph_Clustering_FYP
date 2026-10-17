"""Causal node change scores (E2, E3).

Every function takes matrices already aligned to the current node index (rows and
columns = nodes at t, missing entries zero) and returns one score per node; higher
means more change. None reads a future snapshot. Historical state (the EWMA) is
updated by the caller only after the output for t has been produced.
"""

import numpy as np

UNKNOWN = -1
"""Label for a neighbour that had no community at t-1 (new or reappearing node)."""


def one_step(w_t: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
    """||w_t - w_{t-1}||^2 per node: change of the structural (adjacency) row."""
    return ((w_t - w_prev) ** 2).sum(axis=1)


def ewma_score(w_t: np.ndarray, history: np.ndarray) -> np.ndarray:
    """||w_t - h_{t-1}||^2 per node, against an EWMA of earlier rows."""
    return ((w_t - history) ** 2).sum(axis=1)


def ewma_update(history: np.ndarray, w_t: np.ndarray, gamma: float) -> np.ndarray:
    """h_t = (1 - gamma) h_{t-1} + gamma w_t."""
    return (1 - gamma) * history + gamma * w_t


def community_profile(
    w_t: np.ndarray, w_prev: np.ndarray, prev_labels: np.ndarray, n_communities: int
) -> np.ndarray:
    """Jensen-Shannon divergence (base 2) between each node's neighbour distribution
    over the t-1 communities, computed from its t-1 and its t neighbours.

    `prev_labels[j]` is neighbour j's community at t-1 in 0..n_communities-1, or
    UNKNOWN. Profiles have one bucket per community, one for unknown neighbours and
    one for "no neighbours", so an isolated node has a well-defined profile.
    """
    buckets = np.where(prev_labels == UNKNOWN, n_communities, prev_labels)
    onehot = np.zeros((len(prev_labels), n_communities + 2))
    onehot[np.arange(len(prev_labels)), buckets] = 1.0
    return _jsd(_profile(w_t, onehot), _profile(w_prev, onehot))


def _profile(w: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    counts = w @ onehot
    totals = counts.sum(axis=1)
    counts[totals == 0, -1] = 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2

    def kl(a: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(a > 0, a * np.log2(a / m), 0.0)
        return terms.sum(axis=1)

    return np.clip((kl(p) + kl(q)) / 2, 0.0, 1.0)


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Average ranks scaled to [0, 1]; ties share their mean rank."""
    order = scores.argsort(kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return ranks / max(len(scores) - 1, 1)


def combined(score_sets: list[np.ndarray]) -> np.ndarray:
    """Mean of rank-normalised scores, so estimators on different scales mix fairly."""
    return np.mean([rank_normalise(s) for s in score_sets], axis=0)
