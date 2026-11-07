import numpy as np
import pandas as pd
import pytest

from readmission_risk.analysis import (
    chi_square_independence,
    missingness,
    rate_table,
    wilson_interval,
)

T = "readmitted_30d"


def test_wilson_textbook_value():
    lo, hi = wilson_interval(50, 100)
    assert float(lo) == pytest.approx(0.4038, abs=1e-3)
    assert float(hi) == pytest.approx(0.5962, abs=1e-3)


def test_wilson_stays_in_unit_interval_at_extremes():
    lo, hi = wilson_interval(np.array([0, 5]), np.array([5, 5]))
    assert lo.min() >= 0 and hi.max() <= 1
    assert lo[0] == pytest.approx(0) and hi[1] == pytest.approx(1)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "grp": ["a", "a", "a", "b", "b", None],
            "age": pd.Categorical(
                ["[10-20)", "[0-10)", "[0-10)", "[10-20)", "[0-10)", "[0-10)"],
                categories=["[0-10)", "[10-20)", "[20-30)"],
                ordered=True,
            ),
            T: [1, 0, 0, 1, 1, 0],
        }
    )


def test_rate_table_counts_and_missing_level(df):
    t = rate_table(df, "grp")
    assert t.loc["a", "n"] == 3 and t.loc["a", "events"] == 1
    assert t.loc["b", "rate"] == 1.0
    assert t.loc["Missing", "n"] == 1
    assert (t["ci_low"] <= t["rate"]).all() and (t["rate"] <= t["ci_high"]).all()


def test_rate_table_keeps_category_order_and_drops_empty(df):
    t = rate_table(df, "age")
    assert list(t.index) == ["[0-10)", "[10-20)"]


def test_rate_table_min_count(df):
    assert "Missing" not in rate_table(df, "grp", min_count=2).index


def test_missingness_only_missing_columns_sorted():
    d = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3], "c": [None, 1, 2]})
    s = missingness(d)
    assert list(s.index) == ["a", "c"]
    assert s["a"] == pytest.approx(2 / 3)


def test_chi_square_detects_association():
    rng = np.random.default_rng(1)
    x = rng.choice(["low", "high"], size=4000)
    y = (rng.random(4000) < np.where(x == "high", 0.3, 0.1)).astype(int)
    res = chi_square_independence(pd.DataFrame({"x": x, T: y}), "x")
    assert res["p_value"] < 1e-6
    assert 0 < res["cramers_v"] < 1
