# Dynamic Graph Clustering FYP

Resource-aware online dynamic graph clustering via causal adaptive selective updating (CCDS26-0526).

The project studies how an online clustering method should choose which nodes to update, and how much computation to spend, as a graph changes over time. The baseline is a reimplementation of DyG-MF (Li et al., WWW 2025); the research adds causal (no-lookahead) change scores, adaptive and cost-aware update budgets, drift handling and community-event tracking.

## Status

| Part | Content | State |
|---|---|---|
| 1 | Data types, synthetic generator, metrics, tracker, runner, DyG-MF source audit | Done |
| 2 | DyG-MF reimplementation, lookahead adapter, Birth-Death loader, E1 pilot | Done |
| 3 | Causal change estimators, causal DyG-MF, E2/E3 pilots (RQ1, RQ2) | Done |
| 4 | Adaptive update budgets (E4, RQ3) | In progress on branch `week4-adaptive-budget` |

The full eight-part plan and the experiments (E1–E9) are described in the FYP report; decisions and findings are in `docs/design_log.md`.

## Main results so far

These are **pilot results**: 3 test seeds on planted synthetic graphs (10 communities × 40 nodes), with a budget of μ = 16% of nodes. They were run before a later fix that gives each random step its own stream (Part 4). They will be rerun with 10 seeds before final use, so treat the exact numbers as provisional. Full tables: `reports/e2e3_pilot.csv`.

### RQ1: what changes when selective updating cannot see the next snapshot?

**Removing the lookahead costs no quality, and events are detected about a snapshot earlier.** The published DyG-MF score uses snapshot t+1, so its answer for t only exists once t+1 arrives. Causal scores use the present and past only.

| Method | Gradual drift NMI | Abrupt events NMI | Event detection delay (snapshots) |
|---|---|---|---|
| Original DyG-MF (lookahead) | 0.886 ± 0.008 | 0.977 ± 0.009 | 1.25 |
| Causal community profile | **0.991 ± 0.007** | 0.961 ± 0.018 | 0.08 |
| Causal combined | 0.964 ± 0.010 | **0.979 ± 0.000** | 0.08 |

The best causal score matches or beats the lookahead original in both regimes.

### RQ2: which causal change signal finds the nodes that need updating?

**The community-profile score.** It compares where a node's neighbours sat in the previous snapshot's communities, now versus before. It finds every node that changed community under gradual drift. It nearly matches updating every node, with about a third of the recomputation.

| Score | Gradual NMI | Changed nodes found (gradual / abrupt) |
|---|---|---|
| Original DyG-MF score (lookahead) | 0.886 | 0.19 / 0.21 |
| Random selection | 0.879 | 0.17 / 0.18 |
| One-step change | 0.915 | 0.52 / 0.45 |
| EWMA history (γ = 0.2) | 0.928 | 0.54 / 0.44 |
| **Community profile** (selected on validation seeds) | **0.991** | **1.00** / 0.53 |
| Every node updated (μ = 1) | 0.995 | 1.00 / 1.00 |

- The paper's own score finds changed nodes barely better than random.
- The community-profile score is weaker under abrupt merges and splits (0.961), where many neighbourhoods change at once.

### Limits

- **Synthetic graphs only.** Per-snapshot Louvain scores 0.995–1.000 on them. Claims are therefore about choosing nodes *within* DyG-MF's incremental method, with full updating as the reference, not about beating static methods. Real data comes in Part 7.
- **A 16% budget recomputes about 35% of nodes,** because landmarks are re-chosen at every snapshot.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pre-commit install
```

Experiments that use the Birth-Death benchmark need the official DyG-MF repository cloned into `external/` (not committed):

```bash
git clone https://github.com/Clearloveyuan/DyG-MF.git external/DyG-MF
```

## Run

```bash
.venv/bin/python -m dgc.runner.run configs/e0_smoke.yaml                  # writes results/<name>/<run_id>/
.venv/bin/python -m dgc.runner.run configs/e1_birthdeath_dygmf_oracle.yaml --graph-seed 2 --algo-seed 2
.venv/bin/python -m dgc.runner.run configs/e0_smoke.yaml --resume results/.../checkpoint.pkl
```

Overrides (`--graph-seed`, `--algo-seed`, `--method`, `--param KEY=VALUE`, `--name`, `--clear-params`) are recorded in the manifest. `scripts/run_e2_e3.sh` runs the E2/E3 grid and `scripts/summarise_runs.py PREFIX` rebuilds its tables from saved runs. Each run directory holds:

- `manifest.json`: run ID, code commit, DyG-MF commit, resolved config, dependency versions, thread settings, dataset hash, seeds and hardware
- `per_snapshot.csv`: quality (NMI, ARI, NF1, modularity, conductance), update accounting, timings and `available_at`
- `labels.parquet`, `events.json`, `summary.json`
- `checkpoint.pkl` (causal methods only; lookahead runs cannot be resumed yet)

## Configs

| Config | What it runs |
|---|---|
| `e0_smoke.yaml` | Tiny planted sequence with every event type, per-snapshot Louvain |
| `e1_birthdeath_dygmf_oracle.yaml` | DyG-MF (published lookahead) on a 6-community Birth-Death subsample, true number of clusters (diagnostic) |
| `e1_birthdeath_dygmf_elbow.yaml` | As above, number of clusters chosen by the elbow method |
| `e1_birthdeath_dygmf_oracle_full.yaml` | As the oracle config, but every node updated (μ = 1) |
| `e1_birthdeath_louvain.yaml` | Per-snapshot Louvain on the same subsample |
| `e2e3_gradual.yaml`, `e2e3_abrupt.yaml` | Planted graphs (10 × 40 nodes, 20% mixing) with gradual drift or merge/split/birth/death events; scorer and method set by `scripts/run_e2_e3.sh` |

## Methods

- **Louvain per snapshot** (`methods/louvain.py`): clusters every snapshot from scratch; the "update everything" reference.
- **DyG-MF** (`methods/dygmf.py`): reimplemented from the paper and the released code. The released code cannot run as published, so the code and parameters are used where they work, and each defect has a documented fix: NetMF PMI with a two-step window, the corrected landmark-centre update (Eq. 7), and a fixed embedding width r = 20 with a k-means read-out. See `docs/design_log.md` (audit F1–F10, P1–P12, spec D1–D9).
- **Lookahead adapter** (`methods/lookahead.py`): runs methods that need snapshot t+1, recording when each output actually became available.
- **Causal DyG-MF** (`CausalDyGMF` in `methods/dygmf.py`): never sees t+1; the dynamic nodes are chosen by a causal score from `methods/scores.py` (one-step, EWMA, community profile, combined or random). The community-profile score was selected on validation data for the next experiments.

## Checks

```bash
.venv/bin/pytest -q
.venv/bin/pyright src/
.venv/bin/pre-commit run --all-files
```

The tests include contract checks every method must pass: output at t does not depend on later snapshots (at most one step ahead for the lookahead adapter), evaluation labels never reach a method, checkpoints resume exactly, and repeated runs give identical results.

## Layout

- `src/dgc/data`: snapshot types, online stream, synthetic generator, benchmark loaders
- `src/dgc/methods`: online clustering methods and the lookahead adapter
- `src/dgc/tracking`: community tracker
- `src/dgc/metrics`: partition, event and accounting metrics
- `src/dgc/runner`: config, manifest, checkpoints, CLI
- `configs/`: experiment configs
- `reports/`: small committed result summaries (raw runs stay in `results/`, not committed)
- `docs/design_log.md`: decisions, the DyG-MF audit and pilot results (dated entries, labelled by week)
- `docs/related_work.md`: temporal baseline shortlist and choice
