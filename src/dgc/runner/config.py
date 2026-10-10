"""Experiment configuration loaded from YAML."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dgc.data.loaders import load_birthdeath
from dgc.data.synthetic import ScheduledEvent, SyntheticConfig, generate
from dgc.data.types import EvalAnnotations, GraphSnapshot
from dgc.methods.base import OnlineClusterer
from dgc.methods.dygmf import DyGMFConfig, DyGMFCore
from dgc.methods.lookahead import LookaheadAdapter
from dgc.methods.louvain import LouvainBaseline
from dgc.runner.manifest import REPO_ROOT

METHODS = ("louvain", "dygmf_lookahead")


@dataclass(frozen=True)
class BirthDeathConfig:
    """Seeded induced subsample of a Green Birth-Death benchmark (see `load_birthdeath`)."""

    path: str
    n_communities: int
    prefix: str = "birthdeath"


@dataclass(frozen=True)
class RunConfig:
    """One experiment run: data, method, seeds, tracker and event-matching settings.

    `k_mode: oracle` in `method_params` gives the method the reference number of
    communities per snapshot: an explicit, labelled exception to the rule that ground
    truth never reaches a method, for diagnostics only.
    """

    name: str
    data: SyntheticConfig | BirthDeathConfig
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
        return asdict(self) | {"data_source": type(self.data).__name__}


def load_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    data = dict(raw["data"])
    source = data.pop("source", "synthetic")
    if source == "synthetic":
        data["events"] = [ScheduledEvent(**e) for e in data.get("events", [])]
        raw["data"] = SyntheticConfig(**data)
    elif source == "birthdeath":
        raw["data"] = BirthDeathConfig(**data)
    else:
        raise ValueError(f"unknown data source {source!r}")
    if raw["method"] not in METHODS:
        raise ValueError(f"unknown method {raw['method']!r}; known: {list(METHODS)}")
    return RunConfig(**raw)


def build_data(config: RunConfig) -> tuple[list[GraphSnapshot], EvalAnnotations]:
    if isinstance(config.data, SyntheticConfig):
        return generate(config.data, seed=config.graph_seed)
    return load_birthdeath(
        REPO_ROOT / config.data.path,
        config.data.n_communities,
        seed=config.graph_seed,
        prefix=config.data.prefix,
    )


def build_method(
    config: RunConfig, annotations: EvalAnnotations
) -> OnlineClusterer | LookaheadAdapter:
    params = dict(config.method_params)
    if config.method == "louvain":
        return LouvainBaseline(seed=config.algo_seed, **params)
    k_mode = params.pop("k_mode", "elbow")
    if k_mode not in ("elbow", "oracle"):
        raise ValueError(f"unknown k_mode {k_mode!r}")
    k_schedule = (
        {t: len(set(p.values())) for t, p in annotations.labels.items()}
        if k_mode == "oracle"
        else None
    )
    core = DyGMFCore(DyGMFConfig(**params), seed=config.algo_seed, k_schedule=k_schedule)
    return LookaheadAdapter(core)
