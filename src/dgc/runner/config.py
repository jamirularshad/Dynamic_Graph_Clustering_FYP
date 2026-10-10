"""Experiment configuration loaded from YAML."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dgc.data.synthetic import ScheduledEvent, SyntheticConfig
from dgc.methods.base import OnlineClusterer
from dgc.methods.louvain import LouvainBaseline

METHODS = {"louvain": LouvainBaseline}


@dataclass(frozen=True)
class RunConfig:
    """One experiment run: data, method, seeds, tracker and event-matching settings."""

    name: str
    data: SyntheticConfig
    method: str
    graph_seed: int
    algo_seed: int
    tracker_threshold: float
    event_tolerance: int
    event_min_overlap: float
    method_params: dict[str, Any] = field(default_factory=dict)
    checkpoint_every: int = 0

    def resolved(self) -> dict[str, Any]:
        """The full configuration with all defaults filled in, for the manifest."""
        return asdict(self)


def load_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    data = dict(raw["data"])
    data["events"] = [ScheduledEvent(**e) for e in data.get("events", [])]
    raw["data"] = SyntheticConfig(**data)
    if raw["method"] not in METHODS:
        raise ValueError(f"unknown method {raw['method']!r}; known: {sorted(METHODS)}")
    return RunConfig(**raw)


def build_method(config: RunConfig) -> OnlineClusterer:
    return METHODS[config.method](seed=config.algo_seed, **config.method_params)
