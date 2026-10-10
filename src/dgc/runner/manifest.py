"""Run manifest: everything needed to identify and repeat a run."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_info

from dgc.data.types import EvalAnnotations, GraphSnapshot

REPO_ROOT = Path(__file__).resolve().parents[3]
DYGMF_DIR = REPO_ROOT / "external" / "DyG-MF"
PACKAGES = ("numpy", "scipy", "networkx", "scikit-learn", "pandas", "pyarrow", "pyyaml")
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def dataset_hash(snapshots: list[GraphSnapshot], annotations: EvalAnnotations) -> str:
    """SHA-256 over a canonical serialisation of the graphs, labels and planted events."""
    payload = {
        "snapshots": [[s.t, sorted(s.nodes), sorted(s.edges)] for s in snapshots],
        "labels": {t: sorted(p.items()) for t, p in sorted(annotations.labels.items())},
        "events": [
            [e.t, e.kind, [sorted(c) for c in e.before], [sorted(c) for c in e.after]]
            for e in annotations.events
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    """Output of a git command, or None if git or the repository is unavailable."""
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def build_manifest(resolved_config: dict[str, Any], data_hash: str) -> dict[str, Any]:
    status = _git(["status", "--porcelain"], REPO_ROOT)
    return {
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git(["rev-parse", "HEAD"], REPO_ROOT),
        "code_dirty": bool(status) if status is not None else None,
        "dygmf_commit": _git(["rev-parse", "HEAD"], DYGMF_DIR) if DYGMF_DIR.exists() else None,
        "config": resolved_config,
        "dataset_hash": data_hash,
        "seeds": {
            "graph_seed": resolved_config["graph_seed"],
            "algo_seed": resolved_config["algo_seed"],
        },
        "python": sys.version,
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
        "threadpools": [
            {k: info.get(k) for k in ("internal_api", "num_threads", "version")}
            for info in threadpool_info()
        ],
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    }
