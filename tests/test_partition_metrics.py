import math

import pytest

from dgc.data.types import GraphSnapshot, canonical_edges
from dgc.metrics import partition as pm

TRUE = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1}


def test_nmi_and_ari_ignore_label_names():
    relabelled = {n: 10 - c for n, c in TRUE.items()}
    assert pm.nmi(TRUE, relabelled) == pytest.approx(1.0)
    assert pm.ari(TRUE, relabelled) == pytest.approx(1.0)


def test_prediction_must_cover_reference_nodes():
    with pytest.raises(ValueError):
        pm.nmi(TRUE, {1: 0})


def test_nf1_hand_computed():
    # pred {1,2,3}: P=1, R=3/4 -> F1 0.857 -> 0.86; pred {4..8}: P=4/5, R=1 -> 0.889 -> 0.89
    true = [frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})]
    pred = [frozenset({1, 2, 3}), frozenset({4, 5, 6, 7, 8})]
    assert pm.nf1(pred, true) == pytest.approx((0.86 + 0.89) / 2)


def test_nf1_penalises_redundancy():
    # F1s 0.67, 0.67, 1.0; coverage 1; redundancy 3 / 2
    true = [frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})]
    pred = [frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6, 7, 8})]
    assert pm.nf1(pred, true) == pytest.approx(((0.67 + 0.67 + 1.0) / 3) / 1.5)


def test_nf1_perfect_and_no_match():
    true = [frozenset({1, 2}), frozenset({3, 4})]
    assert pm.nf1(true, true) == pytest.approx(1.0)
    assert pm.nf1([frozenset({9})], true) == 0.0


def two_triangles():
    nodes = frozenset(range(6))
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    return pm.to_networkx(GraphSnapshot(0, nodes, canonical_edges(edges, nodes)))


def test_modularity_hand_computed():
    # m = 7; each community: 3 internal edges, degree sum 7 -> Q = 2 * (3/7 - 1/4) = 5/14
    part = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    assert pm.modularity(two_triangles(), part) == pytest.approx(5 / 14)


def test_conductance_hand_computed():
    part = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    assert pm.mean_conductance(two_triangles(), part) == pytest.approx(1 / 7)


def test_conductance_undefined_for_single_community():
    assert math.isnan(pm.mean_conductance(two_triangles(), dict.fromkeys(range(6), 0)))
