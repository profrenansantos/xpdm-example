"""Turn the model and the ranking into statements an engineer can check."""
from __future__ import annotations

import pandas as pd

CRITERIA_TEXT = {  # name, unit, higher value means more urgent
    "rul_lower": ("pessimistic (10th percentile) RUL", "cycles", False),
    "degradation_rate": ("health-index degradation rate", "sd per cycle", True),
    "process_deviations": ("open process deviations", "", True),
}


def sensor_deviations(snapshot: pd.Series, healthy: pd.DataFrame, top_features: list[str], k: int = 3) -> list[str]:
    """Largest standardised deviations from the healthy fleet baseline, among the most important features."""
    z = ((snapshot[top_features] - healthy[top_features].mean()) / healthy[top_features].std()).dropna()
    z = z.reindex(z.abs().sort_values(ascending=False).index)[:k]
    return [f"{f} is {v:+.1f} sd from the healthy baseline" for f, v in z.items()]


def rationale(unit, row: pd.Series, crit: pd.DataFrame, weights: pd.Series, n_units: int) -> str:
    """Deterministic, template-based explanation of one engine's rank. Every number comes from the table."""
    shares = row.filter(like="share_").sort_values(ascending=False)
    lead = [s.replace("share_", "") for s in shares.index[:2]]
    parts = []
    for c in lead:
        name, unit_txt, higher_is_urgent = CRITERIA_TEXT.get(c, (c, "", True))
        worse = (crit[c] < crit.at[unit, c]).mean() if higher_is_urgent else (crit[c] > crit.at[unit, c]).mean()
        val = f"{crit.at[unit, c]:.3f}" if abs(crit.at[unit, c]) < 1 else f"{crit.at[unit, c]:.1f}"
        parts.append(f"{name} of {val} {unit_txt}".replace("  ", " ").strip()
                     + f" (weight {weights[c]:.2f}; more urgent than {100 * worse:.0f}% of the fleet)")
    return (f"Engine {unit} is ranked {int(row['rank'])} of {n_units} (closeness {row['closeness']:.3f}). "
            f"The ranking is driven mainly by its {parts[0]} and its {parts[1]}.")
