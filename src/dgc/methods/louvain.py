"""Per-snapshot Louvain baseline: clusters every snapshot from scratch."""

import time
from typing import Any

import networkx as nx
import numpy as np

from dgc.data.types import GraphSnapshot
from dgc.methods.base import StepResult
from dgc.metrics.partition import to_networkx


class LouvainBaseline:
    """Static Louvain on each snapshot; every active node counts as updated.

    The seed for snapshot t is derived from (seed, t), so the output at t does not
    depend on how many snapshots came before or after it.
    """

    def __init__(self, seed: int, resolution: float = 1.0):
        self.seed = seed
        self.resolution = resolution

    def step(self, snapshot: GraphSnapshot) -> StepResult:
        start = time.perf_counter()
        step_seed = int(np.random.SeedSequence([self.seed, snapshot.t]).generate_state(1)[0])
        groups = nx.community.louvain_communities(
            to_networkx(snapshot), resolution=self.resolution, seed=step_seed
        )
        labels = {n: label for label, members in enumerate(groups) for n in members}
        return StepResult(
            labels=labels,
            updated_nodes=snapshot.nodes,
            timings={"cluster": time.perf_counter() - start},
        )

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        pass
