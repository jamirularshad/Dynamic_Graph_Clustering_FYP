# Dynamic Graph Clustering FYP

Resource-aware online dynamic graph clustering via causal adaptive selective updating (CCDS26-0526).

The project studies how an online clustering method should choose which nodes to update, and how much computation to spend, as a graph changes over time. The baseline is a reimplementation of DyG-MF (Li et al., WWW 2025); the research adds causal (no-lookahead) change scores, adaptive and cost-aware update budgets, drift handling and community-event tracking.

## Status

| Week | Content | State |
|---|---|---|
| 1 | Data types, synthetic generator, metrics, tracker, runner, DyG-MF source audit | Done |
| 2 | DyG-MF reimplementation, lookahead adapter, Birth-Death loader, E1 pilot | Done |
| 3 | Causal change estimators (E2, E3) | Next |

The full eight-week plan and the experiments (E1–E9) are described in the FYP report; decisions and findings are in `docs/design_log.md`.

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

Seed overrides replace the config's seeds and are recorded in the manifest. Each run directory holds:

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

## Methods

- **Louvain per snapshot** (`methods/louvain.py`): clusters every snapshot from scratch; the "update everything" reference.
- **DyG-MF** (`methods/dygmf.py`): reimplemented from the paper and the released code. The released code cannot run as published, so the code and parameters are used where they work, and each defect has a documented fix: NetMF PMI with a two-step window, the corrected landmark-centre update (Eq. 7), and a fixed embedding width r = 20 with a k-means read-out. See `docs/design_log.md` (audit F1–F10, P1–P12, spec D1–D9).
- **Lookahead adapter** (`methods/lookahead.py`): runs methods that need snapshot t+1, recording when each output actually became available.

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
- `docs/design_log.md`: decisions, the DyG-MF audit and pilot results
- `docs/related_work.md`: temporal baseline shortlist and choice
