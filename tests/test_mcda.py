import numpy as np
import pandas as pd
import pytest

from xpdm import mcda


def test_roc_weights_sum_and_order():
    w = mcda.roc_weights(["a", "b", "c"])
    assert w.sum() == pytest.approx(1.0)
    assert list(w.round(3)) == [0.611, 0.278, 0.111]


def test_topsis_dominant_alternative_ranks_first():
    X = pd.DataFrame({"risk": [1.0, 5.0, 9.0], "rate": [3.0, 2.0, 1.0]}, index=["A", "B", "C"])
    benefit = {"risk": False, "rate": True}  # lower risk value = more urgent here
    r = mcda.topsis(X, pd.Series({"risk": 0.5, "rate": 0.5}), benefit)
    assert r.index[0] == "A" and r.index[-1] == "C"
    shares = r.filter(like="share_").sum(axis=1)
    assert np.allclose(shares[r["closeness"] > 0], 1.0)


def test_critic_weights_are_a_distribution():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
    w = mcda.critic_weights(X, {c: True for c in X})
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
