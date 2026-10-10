"""Partition quality metrics.

NMI and ARI measure agreement with reference labels; modularity and conductance
describe structure only. All functions evaluate on the nodes of the reference
partition, which the predicted partition must cover.
"""

from collections import defaultdict

import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from dgc.data.types import GraphSnapshot, Partition


def communities(partition: Partition) -> list[frozenset[int]]:
    """Group a node -> label mapping into community node sets, ordered by smallest node."""
    groups: dict[int, set[int]] = defaultdict(set)
    for node, label in partition.items():
        groups[label].add(node)
    return sorted((frozenset(g) for g in groups.values()), key=min)


def _aligned(true: Partition, pred: Partition) -> tuple[list[int], list[int]]:
    missing = true.keys() - pred.keys()
    if missing:
        raise ValueError(f"prediction is missing {len(missing)} reference nodes")
    nodes = sorted(true)
    return [true[n] for n in nodes], [pred[n] for n in nodes]


def nmi(true: Partition, pred: Partition) -> float:
    """Normalised mutual information, arithmetic-mean normalisation (scikit-learn default)."""
    y_true, y_pred = _aligned(true, pred)
    return float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))


def ari(true: Partition, pred: Partition) -> float:
    """Adjusted Rand index."""
    y_true, y_pred = _aligned(true, pred)
    return float(adjusted_rand_score(y_true, y_pred))


def nf1(pred: list[frozenset[int]], true: list[frozenset[int]]) -> float:
    """Normalised F1 of Rossetti, Pappalardo & Rinzivillo (2016), ported from CDLIB's NF1.

    Each predicted community is matched to every reference community sharing the
    maximal number of its (first-seen) nodes; each match yields precision
    p / |pred community| and recall p / |reference community|. Per-match F1 is rounded
    to two decimals, as in the reference implementation. Then
    NF1 = mean(F1) * coverage / redundancy, with coverage = matched reference
    communities / all reference communities and redundancy = predicted communities /
    matched reference communities. Returns 0.0 when nothing matches.
    """
    node_to_true: dict[int, list[int]] = defaultdict(list)
    for label, members in enumerate(true):
        for n in members:
            node_to_true[n].append(label)

    seen: set[int] = set()
    matched: set[int] = set()
    f1s: list[float] = []
    for members in pred:
        counts: dict[int, int] = defaultdict(int)
        for n in sorted(members):
            if n in seen:
                continue
            seen.add(n)
            for label in node_to_true[n]:
                counts[label] += 1
        if not counts:
            continue
        best = max(counts.values())
        for label, p in counts.items():
            if p != best:
                continue
            matched.add(label)
            precision, recall = p / len(members), p / len(true[label])
            f1s.append(round(2 * precision * recall / (precision + recall), 2))

    if not matched:
        return 0.0
    coverage = len(matched) / len(true)
    redundancy = len(pred) / len(matched)
    return (sum(f1s) / len(f1s)) * coverage / redundancy


def to_networkx(snapshot: GraphSnapshot) -> nx.Graph:
    """Build an undirected networkx graph over the snapshot's active nodes."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted(snapshot.nodes))
    graph.add_edges_from(sorted(snapshot.edges))
    return graph


def modularity(graph: nx.Graph, partition: Partition) -> float:
    """Newman modularity of the partition on `graph`; 0.0 for an edgeless graph."""
    if graph.number_of_edges() == 0:
        return 0.0
    return float(nx.community.modularity(graph, communities(partition)))


def mean_conductance(graph: nx.Graph, partition: Partition) -> float:
    """Mean conductance over communities with defined conductance (0 < volume < total).

    Returns NaN if no community qualifies (for example, a single community).
    """
    total = 2 * graph.number_of_edges()
    values = []
    for members in communities(partition):
        volume = nx.volume(graph, members)
        if 0 < volume < total:
            values.append(nx.conductance(graph, members))
    return sum(values) / len(values) if values else float("nan")
