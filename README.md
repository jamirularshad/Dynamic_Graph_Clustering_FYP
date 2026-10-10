# Dynamic Graph Clustering FYP

Resource-aware online dynamic graph clustering via causal adaptive selective updating (CCDS26-0526).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pre-commit install
```

## Run

```bash
.venv/bin/python -m dgc.runner.run configs/e0_smoke.yaml            # writes results/<name>/<run_id>/
.venv/bin/python -m dgc.runner.run configs/e0_smoke.yaml --resume results/.../checkpoint.pkl
```

Each run directory holds `manifest.json` (commit, DyG-MF commit, resolved config, dependency versions, thread settings, dataset hash, seeds, hardware), `per_snapshot.csv`, `labels.parquet`, `events.json`, `summary.json` and `checkpoint.pkl`.

## Checks

```bash
.venv/bin/pytest -q
.venv/bin/pyright src/
.venv/bin/pre-commit run --all-files
```

## Layout

- `src/dgc/data`: snapshot types, online stream, synthetic generator
- `src/dgc/methods`: online clustering methods
- `src/dgc/tracking`: community tracker
- `src/dgc/metrics`: partition, event and accounting metrics
- `src/dgc/runner`: config, manifest, checkpoints, CLI
- `docs/design_log.md`: decisions and the DyG-MF audit
- `external/DyG-MF`: the official DyG-MF repository, cloned for the audit (not committed)
