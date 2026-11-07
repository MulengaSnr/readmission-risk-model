import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from readmission_risk.models.explain import (
    column_to_feature,
    dependence_table,
    importance,
    shap_by_feature,
)
from readmission_risk.models.fairness import (
    audit_groups,
    fairness_table,
    group_metrics,
    top_share_threshold,
)
from readmission_risk.visualisation import plots

# --- explain -------------------------------------------------------------


def test_column_to_feature_prefers_longest_name():
    features = ["diag_1", "diag_1_group", "number_inpatient"]
    columns = ["number_inpatient", "diag_1_group_Circulatory", "diag_1_X"]
    assert column_to_feature(columns, features) == [
        "number_inpatient", "diag_1_group", "diag_1",
    ]  # fmt: skip


def test_column_to_feature_rejects_unknown_columns():
    with pytest.raises(ValueError, match="matches no input feature"):
        column_to_feature(["mystery"], ["number_inpatient"])


def make_pipeline_and_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "unused": rng.normal(size=n),
        }
    )
    y = ((x["num"] + (x["cat"] == "a")) + rng.normal(size=n) > 1).astype(int)
    prep = ColumnTransformer(
        [
            ("num", "passthrough", ["num"]),
            ("cat", OneHotEncoder(sparse_output=False), ["cat"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        [("prep", prep), ("model", XGBClassifier(n_estimators=30, max_depth=3))]
    )
    return model.fit(x, y), x


def test_shap_values_are_grouped_and_additive():
    model, x = make_pipeline_and_data()
    values, bias = shap_by_feature(model, x)
    assert list(values.columns) == ["num", "cat"]  # one-hot summed, unused absent
    margin = (
        model["model"]
        .get_booster()
        .predict(xgboost.DMatrix(model["prep"].transform(x)), output_margin=True)
    )
    np.testing.assert_allclose(values.sum(axis=1) + bias, margin, atol=1e-4)


def test_importance_is_sorted_with_shares():
    values = pd.DataFrame({"number_inpatient": [1.0, -1.0], "other": [0.1, 0.1]})
    table = importance(values)
    assert table["feature"].tolist() == ["number_inpatient", "other"]
    assert table.loc[0, "label"] == "Inpatient stays, previous year"
    assert table.loc[1, "label"] == "other"  # no label -> column name
    assert table["share"].sum() == pytest.approx(1.0)


def test_dependence_table_pools_top_counts():
    raw = pd.Series([0, 0, 1, 5, 7, 9])
    shap = pd.Series([-1.0, -0.5, 0.0, 1.0, 2.0, 3.0])
    table = dependence_table(raw, shap, cap=5)
    assert table["level"].tolist() == ["0", "1", "5+"]
    assert table["n"].tolist() == [2, 1, 3]
    assert table.loc[2, "median"] == 2.0


# --- fairness ------------------------------------------------------------


def test_audit_groups_bands_and_unknown_race():
    df = pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "age": ["[30-40)", "[80-90)"],
            "race": ["Caucasian", None],
        }
    )
    groups = audit_groups(df)
    assert groups["age_band"].tolist() == ["<40", "80+"]
    assert groups["race"].tolist() == ["Caucasian", "Unknown"]
    with pytest.raises(ValueError, match="Unmapped age"):
        audit_groups(df.assign(age=["[30-40)", "100+"]))


def test_top_share_threshold_flags_requested_share():
    prob = np.linspace(0, 1, 100)
    assert (prob >= top_share_threshold(prob, 0.2)).sum() == 20


def test_group_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.2, 0.8, 0.1])
    flagged = np.array([True, False, True, False])
    m = group_metrics(y, prob, flagged)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5
    assert m["selection_rate"] == 0.5
    assert m["roc_auc"] == 0.75
    assert m["calibration_gap"] == pytest.approx(0.5 - 0.5)
    single = group_metrics(
        np.zeros(3, int), np.array([0.1, 0.2, 0.3]), np.ones(3, bool)
    )
    assert np.isnan(single["roc_auc"])
    assert np.isnan(single["recall"])


def test_fairness_table_structure():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], n),
            "age": rng.choice(["[50-60)", "[70-80)", "[80-90)"], n),
            "race": rng.choice(
                ["Caucasian", "AfricanAmerican", "Asian"], n, p=[0.7, 0.27, 0.03]
            ),
        }
    )
    y = rng.binomial(1, 0.15, n)
    prob = np.clip(0.15 + 0.2 * (y - 0.15) + rng.normal(0, 0.05, n), 0.01, 0.99)
    patients = pd.Series(rng.integers(0, 600, n))
    table = fairness_table(y, prob, audit_groups(df), patients, n_boot=30, seed=1)
    ref = table.query(
        "attribute == 'race' and group == 'Caucasian' and metric == 'recall_diff'"
    )
    assert ref["estimate"].item() == 0.0
    asian = table.query("group == 'Asian'")
    assert asian["small_group"].all()
    defined = table.dropna(subset=["ci_low", "ci_high"])
    assert (defined["ci_low"] <= defined["ci_high"] + 1e-12).all()
    assert set(table["metric"]) == {
        "roc_auc", "calibration_gap", "selection_rate",
        "recall", "precision", "recall_diff",
    }  # fmt: skip


# --- plots ---------------------------------------------------------------


def test_explain_and_fairness_plots_draw():
    plots.apply_style()
    table = importance(pd.DataFrame({"a": [1.0, 0.5], "b": [0.2, 0.1]}))
    assert len(plots.plot_importance(table, "t").patches) == 2
    dep = dependence_table(pd.Series([0, 1, 2, 2]), pd.Series([-0.2, 0.1, 0.4, 0.5]))
    ax = plots.plot_dependence(dep, "t", "x")
    assert ax.get_xticklabels()[0].get_text().startswith("0")
    rows = pd.DataFrame(
        {
            "attribute": ["sex", "sex", "age_band", "age_band"],
            "group": ["Female", "Male", "80+", "<40"],
            "estimate": [0.4, 0.38, 0.34, 0.5],
            "ci_low": [0.35, 0.33, 0.3, 0.4],
            "ci_high": [0.45, 0.43, 0.38, 0.6],
            "n": [10, 9, 5, 2],
            "small_group": [False, False, False, True],
        }
    )
    panels = {"sex": "Sex", "age_band": "Age band"}
    order = {"age_band": ["<40", "80+"]}
    fig = plots.plot_group_metric(rows, panels, 0.39, "t", "x", orders=order)
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert labels == ["<40 *", "80+"]
