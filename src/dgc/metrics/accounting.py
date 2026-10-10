"""Update accounting per snapshot.

Node budget (distinct nodes recomputed) and measured runtime are separate quantities
and are reported separately; fewer updated nodes need not mean less runtime.
"""

from dgc.methods.base import StepResult


def step_accounting(result: StepResult, n_active: int, n_arrived: int) -> dict[str, float]:
    """Node-budget and timing figures for one step.

    `n_arrived` counts new or reappearing nodes, whose initialisation is mandatory work.
    """
    timings = {f"time_{phase}": seconds for phase, seconds in result.timings.items()}
    return {
        "n_active": n_active,
        "n_arrived": n_arrived,
        "n_updated": len(result.updated_nodes),
        "update_fraction": len(result.updated_nodes) / n_active if n_active else 0.0,
        **timings,
    }
