"""Contract tests every causal method must pass: no future access, no label leakage,
exact checkpoint resume and run-to-run determinism."""

import random

import pytest

from dgc.data.synthetic import ScheduledEvent, SyntheticConfig, generate
from dgc.data.types import EvalAnnotations
from dgc.methods.dygmf import CausalDyGMF, DyGMFConfig
from dgc.methods.louvain import LouvainBaseline
from dgc.runner.run import run_sequence
from dgc.tracking.tracker import Tracker

DYGMF = dict(r=8, n_iter=15, k_max=6)
METHODS = {
    "louvain": lambda: LouvainBaseline(seed=3),
    "dygmf_causal_eq14": lambda: CausalDyGMF(DyGMFConfig(**DYGMF, scorer="eq14"), seed=3),
    "dygmf_causal_combined": lambda: CausalDyGMF(DyGMFConfig(**DYGMF, scorer="combined"), seed=3),
}

CONFIG = SyntheticConfig(
    n_communities=4,
    community_size=15,
    n_snapshots=6,
    p_in=0.4,
    p_out=0.02,
    move_fraction=0.05,
    events=[ScheduledEvent(2, "merge"), ScheduledEvent(4, "birth")],
)
K = 3  # cut point: compare outputs for t <= K


def run(make_method, snapshots, annotations, **kwargs):
    return run_sequence(make_method(), Tracker(0.3), snapshots, annotations, **kwargs)


def scientific(rows):
    """Rows without timing fields, which legitimately differ between runs."""
    return [{k: v for k, v in r.items() if not k.startswith("time_")} for r in rows]


@pytest.fixture(scope="module")
def data():
    return generate(CONFIG, seed=11)


@pytest.mark.parametrize("name", METHODS)
def test_output_does_not_depend_on_future(name, data):
    snapshots, ann = data
    full = run(METHODS[name], snapshots, ann)
    prefix = run(METHODS[name], snapshots[: K + 1], ann)
    other, _ = generate(CONFIG, seed=99)
    altered = run(METHODS[name], snapshots[: K + 1] + other[K + 1 :], EvalAnnotations(labels={}))
    for t in range(K + 1):
        assert full.predictions[t] == prefix.predictions[t] == altered.predictions[t]


@pytest.mark.parametrize("name", METHODS)
def test_evaluation_labels_do_not_reach_method(name, data):
    snapshots, ann = data
    rng = random.Random(0)
    scrambled = EvalAnnotations(
        labels={t: {n: rng.randrange(50) for n in p} for t, p in ann.labels.items()},
        events=[],
    )
    assert run(METHODS[name], snapshots, ann).predictions == (
        run(METHODS[name], snapshots, scrambled).predictions
    )


@pytest.mark.parametrize("name", METHODS)
def test_checkpoint_resume_matches_uninterrupted_run(name, data):
    snapshots, ann = data
    full = run(METHODS[name], snapshots, ann, data_hash="h")
    saved = []
    run(METHODS[name], snapshots, ann, data_hash="h", on_checkpoint=saved.append, stop_after=K)
    resumed = run(METHODS[name], snapshots, ann, data_hash="h", resume=saved[-1])
    assert resumed.predictions == full.predictions
    assert resumed.identities == full.identities
    assert resumed.events == full.events
    assert scientific(resumed.rows) == scientific(full.rows)


def test_resume_rejects_other_dataset(data):
    snapshots, ann = data
    saved = []
    run(METHODS["louvain"], snapshots, ann, data_hash="h", on_checkpoint=saved.append, stop_after=K)
    with pytest.raises(ValueError):
        run(METHODS["louvain"], snapshots, ann, data_hash="other", resume=saved[-1])


@pytest.mark.parametrize("name", METHODS)
def test_repeat_runs_give_identical_scientific_output(name, data):
    snapshots, ann = data
    a, b = run(METHODS[name], snapshots, ann), run(METHODS[name], snapshots, ann)
    assert a.predictions == b.predictions and a.events == b.events
    assert scientific(a.rows) == scientific(b.rows)
