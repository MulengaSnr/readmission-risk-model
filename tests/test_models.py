import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from readmission_risk.config import PROJECT_ROOT
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features
from readmission_risk.models.baselines import (
    cross_validate_models,
    grouped_folds,
    make_baselines,
    summarise,
)
from readmission_risk.models.metrics import capture_at_capacity, evaluate

# --- capture_at_capacity -------------------------------------------------


def test_perfect_ranking_captures_all_positives():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.linspace(1, 0, 10)
    assert capture_at_capacity(y, score, 0.2) == (1.0, 1.0)
    assert capture_at_capacity(y, score, 0.1) == (1.0, 0.5)


def test_worst_ranking_captures_nothing():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert capture_at_capacity(y, np.linspace(0, 1, 10), 0.2) == (0.0, 0.0)


def test_constant_score_is_the_no_skill_reference():
    # All tied: recall equals capacity and precision equals prevalence.
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    precision, recall = capture_at_capacity(y, np.full(10, 0.3), 0.3)
    assert precision == pytest.approx(0.2)
    assert recall == pytest.approx(0.3)


def test_ties_at_cutoff_are_shared_proportionally():
    # Top 2 of 4: one clear winner (positive), then a 1-of-3 pick from a tie
    # containing one positive -> expected 1 + 1/3 positives reviewed.
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.5, 0.5, 0.5])
    precision, recall = capture_at_capacity(y, score, 0.5)
    assert precision == pytest.approx((1 + 1 / 3) / 2)
    assert recall == pytest.approx((1 + 1 / 3) / 2)


def test_float_error_does_not_add_a_row():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # 0.3 * 10 is 3.0000000000000004 in floating point; must review 3, not 4.
    assert capture_at_capacity(y, np.linspace(1, 0, 10), 0.3) == (1.0, 1.0)


@pytest.mark.parametrize("capacity", [0, -0.1, 1.5])
def test_invalid_capacity_raises(capacity):
    with pytest.raises(ValueError, match="capacity"):
        capture_at_capacity([1, 0], [0.5, 0.2], capacity)


@pytest.mark.parametrize(("y", "s"), [([1, 0], [0.5]), ([], []), ([0, 0], [0.1, 0.2])])
def test_bad_inputs_raise(y, s):
    with pytest.raises(ValueError):
        capture_at_capacity(y, s, 0.5)


def test_evaluate_reports_all_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    m = evaluate(y, rng.random(200), capacities=(0.1, 0.25))
    assert set(m) == {
        "pr_auc", "roc_auc", "brier",
        "precision_at_10pct", "recall_at_10pct",
        "precision_at_25pct", "recall_at_25pct",
    }  # fmt: skip
    assert all(0 <= v <= 1 for v in m.values())


# --- cross-validation ----------------------------------------------------


def make_xy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    x = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    y = (signal + rng.normal(scale=1.0, size=n) > 1.2).astype(int)
    groups = pd.Series(rng.integers(0, n // 3, n))
    return x, y, groups


def test_grouped_folds_are_patient_disjoint_and_cover_all_rows():
    _, y, groups = make_xy()
    folds = grouped_folds(y, groups, n_splits=5, seed=1)
    assert len(folds) == 5
    g = groups.to_numpy()
    for tr, va in folds:
        assert not set(g[tr]) & set(g[va])
    all_valid = np.concatenate([va for _, va in folds])
    assert sorted(all_valid) == list(range(len(y)))


def test_cross_validation_scores_every_model_on_every_fold():
    x, y, groups = make_xy()
    folds = grouped_folds(y, groups, n_splits=4)
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "lr": LogisticRegression(),
    }
    results = cross_validate_models(x, y, folds, models)
    assert len(results) == 8
    assert set(results["model"]) == {"dummy", "lr"}
    summary = summarise(results)
    # The informative model ranks first and beats no skill on PR-AUC.
    assert summary.index[0] == "lr"
    assert (
        summary.loc["lr", ("pr_auc", "mean")] > summary.loc["dummy", ("pr_auc", "mean")]
    )
    assert summary.loc["dummy", ("roc_auc", "mean")] == pytest.approx(0.5)


# --- real data -----------------------------------------------------------

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"


@pytest.mark.integration
@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="data not prepared")
def test_baselines_fit_on_real_training_data():
    train = pd.read_parquet(TRAIN_PATH).sample(5000, random_state=0)
    x, y = add_features(train), train[TARGET].to_numpy()
    folds = grouped_folds(y, train[GROUP_COL], n_splits=2)
    results = cross_validate_models(x, y, folds, make_baselines())
    assert set(results["model"]) == set(make_baselines())
    assert results.drop(columns=["model", "fold"]).notna().all().all()
