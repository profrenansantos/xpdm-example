"""Remaining-useful-life model: gradient boosting with quantile models for an uncertainty band."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold


def phm_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 score: late predictions (overestimating RUL) are penalised more."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


MAX_ITER = 300  # boosting rounds; set lower (e.g. 100) for a quick run


def _gbr(**kw):
    return HistGradientBoostingRegressor(max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0, **kw)


@dataclass
class RULModel:
    """Point model plus lower and upper quantile models (default 10% and 90%)."""
    lower_q: float = 0.1
    upper_q: float = 0.9
    models: dict = field(default_factory=dict)
    features: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RULModel":
        self.features = list(X.columns)
        self.models = {
            "point": _gbr().fit(X, y),
            "lower": _gbr(loss="quantile", quantile=self.lower_q).fit(X, y),
            "upper": _gbr(loss="quantile", quantile=self.upper_q).fit(X, y),
        }
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({k: m.predict(X[self.features]) for k, m in self.models.items()},
                           index=X.index).clip(lower=0)
        out["lower"] = np.minimum(out["lower"], out["point"])
        out["upper"] = np.maximum(out["upper"], out["point"])
        return out

    def importance(self, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.Series:
        r = permutation_importance(self.models["point"], X[self.features], y, n_repeats=n_repeats,
                                   random_state=0, scoring="neg_root_mean_squared_error")
        return pd.Series(r.importances_mean, index=self.features).sort_values(ascending=False)


def out_of_fold(feats: pd.DataFrame, cols: list[str], n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold predictions on run-to-failure engines, grouped by engine to avoid leakage."""
    preds = []
    for tr, va in GroupKFold(n_splits=n_splits).split(feats, groups=feats["unit"]):
        m = RULModel().fit(feats.iloc[tr][cols], feats.iloc[tr]["rul"])
        p = m.predict(feats.iloc[va][cols])
        preds.append(pd.concat([feats.iloc[va][["unit", "cycle", "rul"]], p], axis=1))
    return pd.concat(preds).sort_index()
