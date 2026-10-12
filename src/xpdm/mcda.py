"""Multicriteria ranking of engines for maintenance: CRITIC weights and TOPSIS.

The ranking is explainable by construction: every engine's position can be
decomposed into per-criterion contributions to its distance from the ideal
and anti-ideal solutions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def critic_weights(X: pd.DataFrame, benefit: dict[str, bool]) -> pd.Series:
    """CRITIC (Diakoulaki et al., 1995): weight = contrast (std) x conflict (1 - correlation)."""
    N = X.copy().astype(float)
    for c in N:
        lo, hi = N[c].min(), N[c].max()
        N[c] = 0.0 if hi == lo else ((N[c] - lo) / (hi - lo) if benefit[c] else (hi - N[c]) / (hi - lo))
    info = N.std(ddof=0) * (1 - N.corr().fillna(0)).sum()
    return info / info.sum()


def roc_weights(order: list[str]) -> pd.Series:
    """Rank-order centroid weights (Barron & Barrett, 1996) from criteria listed most to least important."""
    n = len(order)
    return pd.Series([sum(1 / j for j in range(i, n + 1)) / n for i in range(1, n + 1)], index=order)


def topsis(X: pd.DataFrame, weights: pd.Series, benefit: dict[str, bool]) -> pd.DataFrame:
    """Return closeness coefficient, rank and per-criterion contributions."""
    V = X.astype(float) / np.sqrt((X.astype(float) ** 2).sum()).replace(0, 1)
    V = V * weights[X.columns]
    ideal = pd.Series({c: V[c].max() if benefit[c] else V[c].min() for c in V})
    anti = pd.Series({c: V[c].min() if benefit[c] else V[c].max() for c in V})
    d_pos2, d_neg2 = (V - ideal) ** 2, (V - anti) ** 2
    d_pos, d_neg = np.sqrt(d_pos2.sum(axis=1)), np.sqrt(d_neg2.sum(axis=1))
    cc = d_neg / (d_pos + d_neg)
    out = pd.DataFrame({"closeness": cc, "rank": cc.rank(ascending=False, method="first").astype(int)})
    # Share of the squared distance to the anti-ideal that each criterion supplies:
    contrib = d_neg2.div(d_neg2.sum(axis=1).replace(0, 1), axis=0).add_prefix("share_")
    return out.join(contrib).sort_values("rank")


def sensitivity(X: pd.DataFrame, weights: pd.Series, benefit: dict[str, bool],
                top: int = 10, n: int = 200, spread: float = 0.25, seed: int = 0) -> pd.Series:
    """How often each engine stays in the top `top` when weights are perturbed by +/- `spread`."""
    rng = np.random.default_rng(seed)
    hits = pd.Series(0, index=X.index)
    for _ in range(n):
        w = weights * rng.uniform(1 - spread, 1 + spread, len(weights))
        r = topsis(X, w / w.sum(), benefit)
        hits[r.index[r["rank"] <= top]] += 1
    return (hits / n).rename("top_stability")
