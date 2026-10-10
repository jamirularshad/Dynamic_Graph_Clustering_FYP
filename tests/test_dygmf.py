import math
import random

import numpy as np
import pytest

from dgc.data.synthetic import ScheduledEvent, SyntheticConfig, generate
from dgc.data.types import EvalAnnotations
from dgc.methods import dygmf as D
from dgc.methods.lookahead import LookaheadAdapter
from dgc.metrics import partition as pm
from dgc.runner.run import run_lookahead_sequence
from dgc.tracking.tracker import Tracker

PLANTED = SyntheticConfig(5, 30, 5, 0.35, 0.02, 0.02, [ScheduledEvent(2, "merge")])
SMALL = D.DyGMFConfig(r=10, n_iter=40, k_max=8)


def oracle(ann):
    return {t: len(set(p.values())) for t, p in ann.labels.items()}


@pytest.fixture(scope="module")
def data():
    return generate(PLANTED, seed=4)


def test_ppmi_window_one_matches_edge_formula():
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)  # path 0-1-2, vol 4
    m = D.ppmi(adj, window=1)
    assert m[0, 1] == pytest.approx(math.log(4 / 2)) and m[1, 2] == pytest.approx(math.log(2))
    assert m[0, 2] == 0.0 and (m >= 0).all()
    assert not D.ppmi(np.zeros((3, 3))).any()


def test_ppmi_window_two_matches_direct_netmf():
    rng = np.random.default_rng(0)
    adj = np.triu((rng.random((8, 8)) < 0.4).astype(float), 1)
    adj = adj + adj.T
    deg = adj.sum(1)
    p = adj / deg[:, None]
    direct = adj.sum() / 2 * (p + p @ p) / deg[None, :]
    expected = np.where(direct > 1, np.log(np.maximum(direct, 1e-300)), 0.0)
    assert np.allclose(D.ppmi(adj, window=2), expected)
    assert (D.ppmi(adj, 2) > 0).sum() >= (D.ppmi(adj, 1) > 0).sum()


def test_reindex_maps_rows_and_columns_and_zero_fills():
    old, new = np.array([1, 2, 3]), np.array([2, 3, 9])
    mat = np.arange(9.0).reshape(3, 3)
    out = D.reindex(mat, old, new)
    assert out[0, 1] == mat[1, 2] and not out[2].any() and not out[:, 2].any()
    cols = D.reindex(np.arange(6.0).reshape(2, 3), old, new, rows=False)
    assert cols.shape == (2, 3) and cols[1, 0] == 4.0 and not cols[:, 2].any()


def test_centre_update_is_the_corrected_eq6_minimiser():
    m_t = np.array([[1.0, 0.0], [3.0, 0.0]])
    m_prev = np.array([[0.0, 2.0], [0.0, 0.0]])
    lam = np.array([0.5, 0.0])  # node 1 was absent at t-1
    centres, _ = D.temporal_centres(m_t, m_prev, lam, k=1, seed=0, max_iter=10)
    expected = (m_t[0] + 0.5 * m_prev[0] + m_t[1]) / (1.5 + 1.0)
    assert centres[0] == pytest.approx(expected)


def test_nearest_landmarks_respects_quotas():
    dist = np.array([[0, 9], [1, 9], [2, 0], [9, 1], [9, 2]], dtype=float)
    assert list(D.nearest_landmarks(dist, 4)) == [0, 1, 2, 3]


def test_eq14_zero_for_unchanged_nodes_positive_for_changed():
    w = np.eye(3)
    centres = np.ones((1, 3))
    assert not D.eq14_scores([w, w, w], [centres] * 3, current=1).any()
    changed = w.copy()
    changed[0] = [0, 1, 1]
    scores = D.eq14_scores([w, changed, w], [centres] * 3, current=1)
    assert scores[0] > 0 and scores[1] == scores[2] == 0


def test_biclustering_penalty_shape_and_sign():
    x = D.biclustering_penalty(np.random.default_rng(0).random((6, 3)), k=2)
    assert x.shape == (6, 3) and (x >= 0).all()


def test_elbow_finds_well_separated_blobs():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 0.05, size=(30, 2)) for c in ((0, 0), (5, 0), (0, 5))])
    assert D.elbow_k(blobs, 2, 8, seed=0) == 3


def run_dygmf(snapshots, ann, cfg=SMALL, seed=1, k_schedule=None):
    adapter = LookaheadAdapter(D.DyGMFCore(cfg, seed=seed, k_schedule=k_schedule))
    return run_lookahead_sequence(adapter, Tracker(0.3), snapshots, ann)


def test_first_snapshot_recovers_planted_communities(data):
    snapshots, ann = data
    out = run_dygmf(snapshots, ann, k_schedule=oracle(ann))
    assert out.rows[0]["nmi"] >= 0.95


def test_availability_times(data):
    snapshots, ann = data
    out = run_dygmf(snapshots, ann, k_schedule=oracle(ann))
    assert [r["available_at"] for r in out.rows] == [1, 2, 3, 4, 4]


def test_output_depends_on_at_most_one_snapshot_ahead(data):
    snapshots, ann = data
    other, _ = generate(PLANTED, seed=99)
    k = 2  # outputs for t <= k may use t + 1 <= 3, but nothing later
    full = run_dygmf(snapshots, ann, k_schedule=oracle(ann))
    altered = run_dygmf(
        snapshots[: k + 2] + other[k + 2 :], EvalAnnotations(labels={}), k_schedule=oracle(ann)
    )
    for t in range(k + 1):
        assert full.predictions[t] == altered.predictions[t]


def test_evaluation_labels_do_not_reach_method_in_elbow_mode(data):
    snapshots, ann = data
    rng = random.Random(0)
    scrambled = EvalAnnotations(
        labels={t: {n: rng.randrange(50) for n in p} for t, p in ann.labels.items()}
    )
    assert run_dygmf(snapshots, ann).predictions == run_dygmf(snapshots, scrambled).predictions


def test_full_update_marks_every_node(data):
    snapshots, ann = data
    cfg = D.DyGMFConfig(r=10, n_iter=20, k_max=8, mu=1.0)
    out = run_dygmf(snapshots, ann, cfg=cfg, k_schedule=oracle(ann))
    assert all(r["update_fraction"] == 1.0 for r in out.rows)


def test_static_nodes_keep_their_embedding(data):
    snapshots, ann = data
    core = D.DyGMFCore(SMALL, seed=1, k_schedule=oracle(ann))
    core.process(snapshots[0], snapshots[1])
    before = dict(core.emb)
    result = core.process(snapshots[1], snapshots[2])
    n = len(snapshots[1].nodes)
    assert len(result.updated_nodes) >= math.ceil(SMALL.mu * n)
    static = snapshots[1].nodes - result.updated_nodes
    assert static and all(np.array_equal(core.emb[v], before[v]) for v in static)


def test_state_round_trip(data):
    snapshots, ann = data
    a = D.DyGMFCore(SMALL, seed=1, k_schedule=oracle(ann))
    a.process(snapshots[0], snapshots[1])
    b = D.DyGMFCore(SMALL, seed=999, k_schedule=oracle(ann))
    b.set_state(a.get_state())
    assert (
        a.process(snapshots[1], snapshots[2]).labels == b.process(snapshots[1], snapshots[2]).labels
    )


def test_deterministic_per_seed(data):
    snapshots, ann = data
    assert run_dygmf(snapshots, ann).predictions == run_dygmf(snapshots, ann).predictions
    nmis = [pm.nmi(ann.labels[t], p) for t, p in run_dygmf(snapshots, ann).predictions.items()]
    assert all(0.0 <= v <= 1.0 for v in nmis)
