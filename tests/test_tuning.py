import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from readmission_risk.config import PROJECT_ROOT
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features
from readmission_risk.models.baselines import grouped_folds
from readmission_risk.models.tuning import (
    load_best_params,
    make_candidates,
    paired_difference,
    save_best_params,
    tune,
    tuned_models,
)


def make_results():
    # Model "b" beats "ref" on 2 of 3 folds; "c" loses on every fold.
    return pd.DataFrame(
        {
            "model": ["ref"] * 3 + ["b"] * 3 + ["c"] * 3,
            "fold": [0, 1, 2] * 3,
            "pr_auc": [0.20, 0.22, 0.21, 0.23, 0.24, 0.20, 0.10, 0.11, 0.12],
        }
    )


def test_paired_difference_is_fold_by_fold():
    diff = paired_difference(make_results(), reference="ref")
    assert list(diff.index) == ["b", "c"]
    assert diff.loc["b", "mean_diff"] == pytest.approx((0.03 + 0.02 - 0.01) / 3)
    assert diff.loc["b", "folds_better"] == 2
    assert diff.loc["c", "folds_better"] == 0
    assert (diff["n_folds"] == 3).all()


def test_paired_difference_unknown_reference():
    with pytest.raises(KeyError, match="missing"):
        paired_difference(make_results(), reference="missing")


def test_tune_returns_best_params_and_sorted_trials():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"signal": rng.normal(size=300)})
    y = (x["signal"] + rng.normal(size=300) > 1).astype(int).to_numpy()
    folds = grouped_folds(y, pd.Series(rng.integers(0, 100, 300)), n_splits=3)
    pipe = Pipeline([("model", LogisticRegression())])
    params, trials = tune(x, y, folds, pipe, {"model__C": [0.01, 1.0]}, n_iter=2)
    assert set(params) == {"model__C"}
    assert len(trials) == 2
    assert trials["mean_test_score"].is_monotonic_decreasing
    assert "C" in trials.columns  # "param_model__" prefix stripped


def test_best_params_round_trip_and_merge(tmp_path):
    path = tmp_path / "best.json"
    with pytest.raises(FileNotFoundError, match="--tune"):
        load_best_params(path)
    save_best_params(path, "random_forest", {"model__min_samples_leaf": np.int64(10)})
    save_best_params(path, "xgboost", {"model__max_depth": 3})
    save_best_params(path, "xgboost", {"model__max_depth": 4})  # replaces
    assert load_best_params(path) == {
        "random_forest": {"model__min_samples_leaf": 10},
        "xgboost": {"model__max_depth": 4},
    }


def test_tuned_models_apply_saved_params():
    models = tuned_models({"xgboost": {"model__max_depth": 3}})
    assert models["xgboost"].get_params()["model__max_depth"] == 3
    with pytest.raises(KeyError, match="Unknown"):
        tuned_models({"svm": {}})


def test_candidate_search_spaces_match_their_pipelines():
    for pipeline, space in make_candidates().values():
        valid = pipeline.get_params()
        assert set(space) <= set(valid)


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"


@pytest.mark.integration
@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="data not prepared")
def test_candidates_fit_on_real_training_data():
    train = pd.read_parquet(TRAIN_PATH).sample(3000, random_state=0)
    x, y = add_features(train), train[TARGET].to_numpy()
    tr, va = grouped_folds(y, train[GROUP_COL], n_splits=2)[0]
    for pipeline, space in make_candidates().values():
        # One cheap setting from each space keeps the test fast.
        cheap = {
            k: v[0] if isinstance(v, list) else v.rvs(random_state=0)
            for k, v in space.items()
        }
        cheap = {k: (50 if k.endswith("n_estimators") else v) for k, v in cheap.items()}
        model = pipeline.set_params(**cheap).fit(x.iloc[tr], y[tr])
        prob = model.predict_proba(x.iloc[va])[:, 1]
        assert ((prob > 0) & (prob < 1)).all()
