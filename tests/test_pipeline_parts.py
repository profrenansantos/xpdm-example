import numpy as np
import pandas as pd

from xpdm import eventlog, processmining
from xpdm.data import COLUMNS, add_rul
from xpdm.features import build_features, degradation_rate, feature_columns, health_index, last_cycle
from xpdm.model import RULModel, phm_score, rmse


def synthetic_fleet(n_units=12, seed=0):
    """Small run-to-failure fleet with one drifting sensor, so tests run without the NASA data."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(80, 140))
        for c in range(1, life + 1):
            drift = (c / life) ** 2
            vals = rng.normal(0, 0.05, 24)
            vals[3 + 1] += 5 * drift   # s2 drifts with wear
            vals[3 + 6] -= 3 * drift   # s7 drifts the other way
            rows.append([u, c, *vals])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_rul_model_learns_drift():
    tr = add_rul(synthetic_fleet(), clip=None)
    sensors = ["s2", "s7"]
    F = build_features(tr, sensors)
    cols = feature_columns(F)
    m = RULModel().fit(F[cols], F["rul"])
    p = m.predict(F[cols])
    assert rmse(F["rul"], p["point"]) < F["rul"].std()
    assert (p["lower"] <= p["point"]).all() and (p["point"] <= p["upper"]).all()
    assert phm_score([10, 10], [5, 15]) > 0


def test_health_index_rises_with_wear():
    tr = add_rul(synthetic_fleet(), clip=None)
    hi = health_index(tr, tr, ["s2", "s7"])
    early, late = hi[tr["rul"] > 60].mean(), hi[tr["rul"] < 10].mean()
    assert late > early
    assert (degradation_rate(tr, hi) >= 0).all()


def test_eventlog_and_conformance():
    tr = add_rul(synthetic_fleet(), clip=None)
    oof = tr[["unit", "cycle"]].assign(lower=tr["rul"])  # perfect predictions
    log = eventlog.simulate(oof)
    assert set(log["activity"]) <= set(eventlog.ACTIVITIES)
    conf = processmining.conformance(log)
    assert conf["fitness"].between(0, 1).all()
    d = processmining.dfg(log)
    assert {"source", "target", "count", "mean_wait"} <= set(d.columns)
    live = eventlog.simulate(oof, in_service=True)
    assert not live["activity"].str.contains("failure").any()


def test_last_cycle_one_row_per_engine():
    F = build_features(synthetic_fleet(), ["s2"])
    assert last_cycle(F).index.is_unique
