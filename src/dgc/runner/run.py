"""Run one experiment: python -m dgc.runner.run CONFIG [--out DIR] [--resume CHECKPOINT].

Methods receive only `GraphSnapshot`s through `OnlineStream`; reference labels and
planted events go to the evaluation code alone.
"""

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from dgc.data.stream import OnlineStream, lifecycle
from dgc.data.types import CommunityEvent, EvalAnnotations, GraphSnapshot, Partition
from dgc.methods.base import OnlineClusterer, StepResult
from dgc.methods.lookahead import LookaheadAdapter
from dgc.metrics import partition as pm
from dgc.metrics.accounting import step_accounting
from dgc.metrics.events import match_events
from dgc.runner import checkpoint as ckpt
from dgc.runner.config import RunConfig, build_data, build_method, load_config
from dgc.runner.manifest import build_manifest, dataset_hash
from dgc.tracking.tracker import Tracker


@dataclass
class SequenceOutput:
    """Everything a run produces, before it is written to disk."""

    rows: list[dict[str, Any]]
    predictions: dict[int, Partition]
    identities: dict[int, dict[int, int]]
    events: list[CommunityEvent]


def run_sequence(
    method: OnlineClusterer,
    tracker: Tracker,
    snapshots: list[GraphSnapshot],
    annotations: EvalAnnotations,
    data_hash: str = "",
    on_checkpoint: Callable[[ckpt.Checkpoint], None] | None = None,
    checkpoint_every: int = 0,
    resume: ckpt.Checkpoint | None = None,
    stop_after: int | None = None,
) -> SequenceOutput:
    """Process snapshots online and evaluate each step against `annotations`.

    `resume` restores state saved after snapshot `resume.t` and continues from the
    next one. `stop_after` ends the run after that snapshot (used to test resuming).
    """
    out = SequenceOutput(rows=[], predictions={}, identities={}, events=[])
    prev: GraphSnapshot | None = None
    if resume is not None:
        if resume.data_hash != data_hash:
            raise ValueError("checkpoint was written for a different dataset")
        method.set_state(resume.method_state)
        tracker.set_state(resume.tracker_state)
        prev = resume.prev_snapshot
        out = SequenceOutput(
            rows=list(resume.rows),
            predictions=dict(resume.predictions),
            identities=dict(resume.identities),
            events=list(resume.events),
        )

    for snap in OnlineStream(snapshots):
        if resume is not None and snap.t <= resume.t:
            continue
        _record(out, tracker, snap, prev, method.step(snap), annotations, available_at=snap.t)
        prev = snap

        is_last = snap.t == snapshots[-1].t or snap.t == stop_after
        if on_checkpoint and (is_last or (checkpoint_every and snap.t % checkpoint_every == 0)):
            on_checkpoint(
                ckpt.Checkpoint(
                    t=snap.t,
                    data_hash=data_hash,
                    method_state=method.get_state(),
                    tracker_state=tracker.get_state(),
                    prev_snapshot=snap,
                    rows=list(out.rows),
                    predictions=dict(out.predictions),
                    identities=dict(out.identities),
                    events=list(out.events),
                )
            )
        if snap.t == stop_after:
            break
    return out


def run_lookahead_sequence(
    adapter: LookaheadAdapter,
    tracker: Tracker,
    snapshots: list[GraphSnapshot],
    annotations: EvalAnnotations,
) -> SequenceOutput:
    """Like `run_sequence` for a method that needs t+1; rows record `available_at`."""
    out = SequenceOutput(rows=[], predictions={}, identities={}, events=[])
    prev: GraphSnapshot | None = None
    outputs = [adapter.step(snap) for snap in OnlineStream(snapshots)] + [adapter.finish()]
    for delayed in outputs:
        if delayed is None:
            continue
        snap = delayed.snapshot
        _record(out, tracker, snap, prev, delayed.result, annotations, delayed.available_at)
        prev = snap
    return out


def _record(
    out: SequenceOutput,
    tracker: Tracker,
    snap: GraphSnapshot,
    prev: GraphSnapshot | None,
    result: StepResult,
    annotations: EvalAnnotations,
    available_at: int,
) -> None:
    """Track, evaluate and store one step's output."""
    events, identities = tracker.step(snap.t, result.labels)
    row: dict[str, Any] = {"t": snap.t, "available_at": available_at}
    row |= step_accounting(result, len(snap.nodes), len(lifecycle(prev, snap).arrived))
    row |= _quality(snap, result.labels, annotations.labels.get(snap.t))
    out.rows.append(row)
    out.predictions[snap.t] = result.labels
    out.identities[snap.t] = identities
    out.events.extend(events)


def _quality(snap: GraphSnapshot, pred: Partition, true: Partition | None) -> dict[str, float]:
    graph = pm.to_networkx(snap)
    row = {
        "modularity": pm.modularity(graph, pred),
        "conductance": pm.mean_conductance(graph, pred),
    }
    if true is not None:
        row |= {
            "nmi": pm.nmi(true, pred),
            "ari": pm.ari(true, pred),
            "nf1": pm.nf1(pm.communities(pred), pm.communities(true)),
        }
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    """Mean of `key` over rows, skipping NaN (undefined per-snapshot values)."""
    values = [r[key] for r in rows if not math.isnan(r[key])]
    return sum(values) / len(values) if values else float("nan")


def summarise(config: RunConfig, out: SequenceOutput, annotations: EvalAnnotations) -> dict:
    """Run-level aggregates: per-snapshot means first, then event scores per kind."""
    metrics = [c for c in ("nmi", "ari", "nf1", "modularity", "conductance") if c in out.rows[0]]
    scores = match_events(
        out.events, annotations.events, config.event_tolerance, config.event_min_overlap
    )
    return {
        "snapshot_means": {c: _mean(out.rows, c) for c in metrics},
        "update_fraction_mean": _mean(out.rows, "update_fraction"),
        "events": [asdict(s) for s in scores],
        "complex_events": sum(e.kind == "complex" for e in out.events),
    }


def run(
    config_path: Path,
    out_root: Path,
    resume_path: Path | None = None,
    graph_seed: int | None = None,
    algo_seed: int | None = None,
) -> Path:
    """Run one config; seed overrides replace the config's seeds and are recorded."""
    config = load_config(config_path)
    if graph_seed is not None:
        config = replace(config, graph_seed=graph_seed)
    if algo_seed is not None:
        config = replace(config, algo_seed=algo_seed)
    snapshots, annotations = build_data(config)
    data_hash = dataset_hash(snapshots, annotations)
    manifest = build_manifest(config.resolved(), data_hash)
    run_dir = out_root / config.name / manifest["run_id"]
    run_dir.mkdir(parents=True)

    method = build_method(config, annotations)
    tracker = Tracker(config.tracker_threshold)
    manifest["resumed_from"] = str(resume_path) if resume_path else None
    if isinstance(method, LookaheadAdapter):
        if resume_path:
            raise ValueError("lookahead runs do not support resuming")
        out = run_lookahead_sequence(method, tracker, snapshots, annotations)
    else:
        out = run_sequence(
            method,
            tracker,
            snapshots,
            annotations,
            data_hash=data_hash,
            on_checkpoint=lambda c: ckpt.save(run_dir / "checkpoint.pkl", c),
            checkpoint_every=config.checkpoint_every,
            resume=ckpt.load(resume_path) if resume_path else None,
        )

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(out.rows).to_csv(run_dir / "per_snapshot.csv", index=False)
    pd.DataFrame(
        [
            {"t": t, "node": n, "label": label, "identity": out.identities[t][n]}
            for t, labels in sorted(out.predictions.items())
            for n, label in sorted(labels.items())
        ]
    ).to_parquet(run_dir / "labels.parquet", index=False)
    (run_dir / "events.json").write_text(
        json.dumps(
            [
                {
                    "t": e.t,
                    "kind": e.kind,
                    "before": [sorted(c) for c in e.before],
                    "after": [sorted(c) for c in e.after],
                }
                for e in out.events
            ]
        )
    )
    (run_dir / "summary.json").write_text(json.dumps(summarise(config, out, annotations), indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--graph-seed", type=int, default=None)
    parser.add_argument("--algo-seed", type=int, default=None)
    args = parser.parse_args()
    print(run(args.config, args.out, args.resume, args.graph_seed, args.algo_seed))


if __name__ == "__main__":
    main()
