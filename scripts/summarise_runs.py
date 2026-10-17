"""Summarise saved runs whose name starts with PREFIX: mean and std over runs.

Usage: python scripts/summarise_runs.py PREFIX [--csv OUT.csv]
Per-run values are aggregated over snapshots first (t >= 1 for update and selection
columns), then averaged across runs. Event F1 is the mean over the non-continuation
kinds that have planted events.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def run_row(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    frame = pd.read_csv(run_dir / "per_snapshot.csv")
    later = frame[frame["t"] >= 1]
    times = [c for c in frame if c.startswith("time_")]
    events = [e for e in summary["events"] if e["kind"] != "continuation" and e["planted"]]
    f1s = [e["f1"] for e in events if e["f1"] is not None]
    delays = [e["mean_operational_delay"] for e in events]
    delays = [d for d in delays if d is not None]
    row = {
        "name": manifest["config"]["name"],
        "seed": manifest["seeds"]["graph_seed"],
        "nmi": frame["nmi"].mean(),
        "nf1": frame["nf1"].mean(),
        "updated": later["update_fraction"].mean(),
        "time_s": frame[times].to_numpy().sum(),
        "event_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
        "op_delay": sum(delays) / len(delays) if delays else float("nan"),
    }
    for col in ("sel_precision", "sel_recall", "sel_f1"):
        if col in later:
            row[col] = later[col].mean()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = [
        run_row(d)
        for d in sorted(args.results.glob(f"{args.prefix}*/*"))
        if (d / "summary.json").exists()
    ]
    frame = pd.DataFrame(rows)
    table = frame.drop(columns="seed").groupby("name").agg(["mean", "std"]).round(3)
    table.insert(0, ("runs", ""), frame.groupby("name").size())
    print(table.to_string())
    if args.csv:
        table.to_csv(args.csv)


if __name__ == "__main__":
    main()
