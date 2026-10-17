import numpy as np
import pytest

from dgc.methods import scores as sc


def test_one_step_hand_computed():
    w_prev = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    w_t = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
    assert list(sc.one_step(w_t, w_prev)) == [2.0, 1.0, 1.0]


def test_ewma_update_and_score():
    h = np.array([[1.0, 0.0]])
    w = np.array([[0.0, 1.0]])
    assert np.allclose(sc.ewma_update(h, w, gamma=0.25), [[0.75, 0.25]])
    assert sc.ewma_score(w, h)[0] == pytest.approx(2.0)


def test_original_community_relative_formula_reduces_to_one_step():
    """Our original proposal subtracted the same previous community centre c from the
    current and previous rows: ||(w_t - c) - (w_prev - c)||^2. The centres cancel."""
    rng = np.random.default_rng(0)
    w_t, w_prev = rng.random((20, 20)), rng.random((20, 20))
    centre = rng.random((20, 20))
    original = (((w_t - centre) - (w_prev - centre)) ** 2).sum(axis=1)
    assert np.allclose(original, sc.one_step(w_t, w_prev))


def test_community_profile_differs_from_one_step():
    # Communities at t-1: {0,1,2} -> 0 and {3,4,5} -> 1. Node 0 swaps neighbour 1 for 2
    # (same community): one-step change, no profile change. Node 3 swaps neighbour 4
    # for 1 (other community): same one-step change, large profile change.
    labels = np.array([0, 0, 0, 1, 1, 1])
    w_prev = np.zeros((6, 6))
    w_t = np.zeros((6, 6))
    w_prev[0, 1] = w_t[0, 2] = 1
    w_prev[3, 4] = w_t[3, 1] = 1
    one = sc.one_step(w_t, w_prev)
    prof = sc.community_profile(w_t, w_prev, labels, n_communities=2)
    assert one[0] == one[3] == 2.0
    assert prof[0] == pytest.approx(0.0) and prof[3] == pytest.approx(1.0)


def test_community_profile_unknown_and_isolated_buckets():
    labels = np.array([0, sc.UNKNOWN, 0])
    w_prev = np.zeros((3, 3))
    w_t = np.zeros((3, 3))
    w_prev[0, 2] = 1  # neighbour in community 0
    w_t[0, 1] = 1  # neighbour with no previous community
    prof = sc.community_profile(w_t, w_prev, labels, n_communities=1)
    assert prof[0] == pytest.approx(1.0)
    assert prof[1] == pytest.approx(0.0)  # isolated at both times
    assert ((prof >= 0) & (prof <= 1)).all()


def test_rank_normalise_handles_ties():
    assert list(sc.rank_normalise(np.array([5.0, 1.0, 5.0, 3.0]))) == pytest.approx(
        [5 / 6, 0.0, 5 / 6, 1 / 3]
    )


def test_combined_mixes_scales_fairly():
    small = np.array([0.1, 0.2, 0.3])
    large = np.array([100.0, 300.0, 200.0])
    # ranks: small -> [0, .5, 1], large -> [0, 1, .5]; the raw scale of `large` is ignored
    assert list(sc.combined([small, large])) == pytest.approx([0.0, 0.75, 0.75])
