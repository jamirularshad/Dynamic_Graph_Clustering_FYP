"""Checkpoints: complete run state after a snapshot, so a run can resume exactly."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dgc.data.types import CommunityEvent, GraphSnapshot, Partition


@dataclass
class Checkpoint:
    """Run state after snapshot `t`: method, tracker and outputs produced so far."""

    t: int
    data_hash: str
    method_state: dict[str, Any]
    tracker_state: dict[str, Any]
    prev_snapshot: GraphSnapshot
    rows: list[dict[str, Any]]
    predictions: dict[int, Partition]
    identities: dict[int, dict[int, int]]
    events: list[CommunityEvent]


def save(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(checkpoint))
    tmp.replace(path)


def load(path: Path) -> Checkpoint:
    return pickle.loads(path.read_bytes())
