"""Feature engineering: drop flat sensors, add rolling statistics and a per-engine trend."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SENSORS, SETTINGS


def informative_sensors(train: pd.DataFrame, tol: float = 1e-4) -> list[str]:
    """Sensors whose standard deviation is above `tol` (several are flat in FD001)."""
    return [s for s in SENSORS if train[s].std() > tol]


def _slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / (x ** 2).sum())


def build_features(df: pd.DataFrame, sensors: list[str], window: int = 15) -> pd.DataFrame:
    """Rolling mean, rolling std and rolling slope per sensor, computed within each engine."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")
    cols = {"unit": df["unit"], "cycle": df["cycle"]}
    for s in sensors:
        roll = g[s].rolling(window, min_periods=1)
        cols[f"{s}_mean"] = roll.mean().reset_index(level=0, drop=True)
        cols[f"{s}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        cols[f"{s}_slope"] = (g[s].rolling(window, min_periods=2).apply(_slope, raw=True)
                              .reset_index(level=0, drop=True).fillna(0.0))
    if "rul" in df:
        cols["rul"] = df["rul"]
    return pd.DataFrame(cols)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ("unit", "cycle", "rul")]


def last_cycle(feats: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its most recent cycle (the fleet snapshot used for ranking)."""
    return feats.loc[feats.groupby("unit")["cycle"].idxmax()].set_index("unit")


def health_index(df: pd.DataFrame, reference: pd.DataFrame, sensors: list[str], rul_col: str = "rul") -> pd.Series:
    """Fused health index: sensors z-scored against healthy cycles of `reference` and oriented so that
    higher means more degraded, then averaged. `reference` must contain run-to-failure RUL."""
    healthy = reference[reference[rul_col] >= reference[rul_col].quantile(0.75)][sensors]  # early-life cycles
    sign = np.sign(reference[sensors].corrwith(-reference[rul_col])).replace(0, 1)
    z = (df[sensors] - healthy.mean()) / healthy.std().replace(0, 1)
    return (z * sign).mean(axis=1).rename("health_index")


def degradation_rate(df: pd.DataFrame, hi: pd.Series, window: int = 30) -> pd.Series:
    """Per engine: slope of the health index over its last `window` cycles (clipped at zero)."""
    d = df[["unit", "cycle"]].assign(hi=hi.values)
    return d.groupby("unit").apply(lambda g: _slope(g["hi"].tail(window).to_numpy()) if len(g) > 2 else 0.0,
                                   include_groups=False).clip(lower=0).rename("degradation_rate")
