import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from readmission_risk.features.build import build_preprocessor, feature_columns
from readmission_risk.models.baselines import grouped_folds
from readmission_risk.models.sensitivity import (
    WITH,
    WITHOUT,
    cross_validate_oof,
    decide,
    group_recall,
    make_variants,
)


def test_exclude_removes_feature_everywhere():
    assert "payer_code" in feature_columns()
    assert "payer_code" not in feature_columns(exclude=("payer_code",))
    prep = build_preprocessor(exclude=("payer_code", "number_inpatient"))
    used = [c for _, _, cols in prep.transformers for c in cols]
    assert "payer_code" not in used and "number_inpatient" not in used
    with pytest.raises(ValueError, match="unknown"):
        feature_columns(exclude=("not_a_feature",))


def test_variants_differ_only_in_payer():
    params = {"xgboost": {"model__max_depth": 3}}
    variants = make_variants(params)
    assert set(variants) == {WITH, WITHOUT}
    for v in variants.values():
        assert v.get_params()["model__max_depth"] == 3
    cols = {
        name: [c for _, _, cs in v["prep"].transformers for c in cs]
        for name, v in variants.items()
    }
    assert set(cols[WITH]) - set(cols[WITHOUT]) == {"payer_code"}


@pytest.mark.parametrize(("loss", "expected"), [(-0.001, "drop"), (-0.01, "keep")])
def test_decision_rule(loss, expected):
    diff = pd.DataFrame({"mean_diff": [loss]}, index=[WITHOUT])
    assert decide(diff, margin=0.005) == expected


def test_oof_predictions_cover_every_row_once():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=300)})
    y = (x["a"] + rng.normal(size=300) > 1).astype(int).to_numpy()
    folds = grouped_folds(y, pd.Series(rng.integers(0, 100, 300)), n_splits=3)
    results, oof = cross_validate_oof(x, y, folds, {"lr": LogisticRegression()})
    assert len(results) == 3
    assert not np.isnan(oof["lr"]).any()


def test_group_recall_change():
    y = np.array([1, 1, 0, 0, 1, 0])
    oof = {
        WITH: np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.1]),
        WITHOUT: np.array([0.9, 0.2, 0.1, 0.2, 0.8, 0.1]),
    }
    groups = pd.DataFrame({"sex": ["F", "F", "F", "M", "M", "M"]})
    table = group_recall(y, oof, groups, capacity=2 / 6).set_index("group")
    assert table.loc["F", "recall_with_payer"] == 1.0
    assert table.loc["F", "recall_change"] == -0.5
    assert table.loc["M", "recall_change"] == 1.0
