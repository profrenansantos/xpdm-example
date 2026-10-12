"""Simulated maintenance event log.

C-MAPSS contains sensor trajectories, not maintenance records. To demonstrate
process mining, this module replays out-of-fold RUL predictions through a simple
condition-based maintenance policy and records what a maintenance information
system would log. The log is SIMULATED: activities, delays and the policy are
assumptions stated in `Policy`, chosen to be plausible, not taken from an operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACTIVITIES = ["Monitoring started", "Degradation alert", "Alert dismissed", "Inspection scheduled",
              "Inspection performed", "Engine removal (planned)", "Engine failure (unplanned removal)"]


@dataclass
class Policy:
    alert_rul: float = 25.0      # raise an alert when the lower RUL bound falls below this
    dismiss_prob: float = 0.15   # share of first alerts dismissed by a planner
    schedule_delay: tuple = (1, 6)    # cycles from alert to inspection being scheduled
    inspection_delay: tuple = (3, 20)  # cycles from scheduling to inspection
    removal_delay: tuple = (1, 5)      # cycles from inspection to planned removal
    start_date: str = "2026-01-01"     # one cycle = one day of operation
    seed: int = 7


def simulate(oof: pd.DataFrame, policy: Policy = Policy(), in_service: bool = False) -> pd.DataFrame:
    """Replay predictions through the maintenance policy.

    oof: columns unit, cycle, lower (lower RUL bound). For run-to-failure engines
    (in_service=False) the engine fails after its last cycle unless removed first.
    For engines still flying (in_service=True) the log simply stops at the last cycle.
    """
    rng = np.random.default_rng(policy.seed)
    t0 = pd.Timestamp(policy.start_date)
    rows = []
    for unit, g in oof.sort_values(["unit", "cycle"]).groupby("unit"):
        end = int(g["cycle"].max())  # last observed cycle

        def log(act, c, unit=unit):
            if in_service and (c > end or "failure" in act):
                return False
            rows.append({"case_id": f"engine_{unit:03d}", "activity": act, "cycle": int(c),
                         "timestamp": t0 + pd.Timedelta(days=int(c))})
            return True

        log("Monitoring started", g["cycle"].min())
        alerts = g.loc[g["lower"] < policy.alert_rul, "cycle"]
        if alerts.empty:
            log("Engine failure (unplanned removal)", end + 1)
            continue
        c = int(alerts.iloc[0])
        log("Degradation alert", c)
        if rng.random() < policy.dismiss_prob:  # dismissed, re-raised a few cycles later
            log("Alert dismissed", c + 1)
            c = c + int(rng.integers(5, 15))
            if c > end:
                log("Engine failure (unplanned removal)", end + 1)
                continue
            log("Degradation alert", c)
        c += int(rng.integers(*policy.schedule_delay))
        if c > end:
            log("Engine failure (unplanned removal)", end + 1)
            continue
        log("Inspection scheduled", c)
        c += int(rng.integers(*policy.inspection_delay))
        if c > end:
            log("Engine failure (unplanned removal)", end + 1)
            continue
        log("Inspection performed", c)
        c += int(rng.integers(*policy.removal_delay))
        if c > end:
            log("Engine failure (unplanned removal)", end + 1)
        else:
            log("Engine removal (planned)", c)
    return pd.DataFrame(rows)


def to_xes_columns(log: pd.DataFrame) -> pd.DataFrame:
    """Rename to the XES-style column names that pm4py and most process-mining tools expect."""
    return log.rename(columns={"case_id": "case:concept:name", "activity": "concept:name",
                               "timestamp": "time:timestamp"})
