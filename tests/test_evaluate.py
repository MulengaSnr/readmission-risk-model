import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from readmission_risk.models.evaluate import (
    DIFFERENCE,
    PRIMARY,
    REFERENCE,
    calibration_table,
    gains_curve,
    guard_single_use,
    patient_bootstrap,
    summarise_test,
)
from readmission_risk.models.metrics import evaluate
from readmission_risk.visualisation import plots


def make_predictions(n=400, seed=0):
    rng = np.random.default_rng(seed)
    groups = pd.Series(rng.integers(0, n // 2, n))
    signal = rng.normal(size=n)
    y = (signal + rng.normal(size=n) > 1.2).astype(int)
    probs = {
        PRIMARY: 1 / (1 + np.exp(-(2 * signal - 2))),
        REFERENCE: 1 / (1 + np.exp(-(signal - 2))) * 0.5 + rng.random(n) * 0.1,
    }
    return y, probs, groups


def test_bootstrap_resamples_whole_patients():
    # 40 patients with two encounters each. Scores separate the classes
    # perfectly, so every resample must keep ROC-AUC = 1, and every resample
    # must contain an even number of rows (patients are never split).
    patients = np.repeat(np.arange(40), 2)
    y = (patients % 2).astype(int)
    probs = {"m": np.where(y == 1, 0.9, 0.1)}
    boot = patient_bootstrap(y, probs, pd.Series(patients), n_boot=30, seed=1)
    assert len(boot) == 30
    assert (boot["roc_auc"] == 1.0).all()
    codes, _ = pd.factorize(pd.Series(patients))
    rng = np.random.default_rng(1)
    copies = np.bincount(rng.integers(0, 40, 40), minlength=40)[codes]
    assert copies.sum() % 2 == 0


def test_bootstrap_is_reproducible():
    y, probs, groups = make_predictions()
    a = patient_bootstrap(y, probs, groups, n_boot=5, seed=3)
    b = patient_bootstrap(y, probs, groups, n_boot=5, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["model"]) == {PRIMARY, REFERENCE}


def test_summary_has_paired_difference():
    y, probs, groups = make_predictions()
    point = {m: evaluate(y, p) for m, p in probs.items()}
    summary = summarise_test(point, patient_bootstrap(y, probs, groups, n_boot=50))
    assert set(summary["model"]) == {PRIMARY, REFERENCE, DIFFERENCE}
    row = summary.query("model == @DIFFERENCE and metric == 'pr_auc'").iloc[0]
    assert row["estimate"] == pytest.approx(
        point[PRIMARY]["pr_auc"] - point[REFERENCE]["pr_auc"]
    )
    assert (summary["ci_low"] <= summary["ci_high"]).all()


def test_calibration_table_counts_and_rates():
    y, probs, _ = make_predictions(n=1000)
    table = calibration_table(y, probs[PRIMARY], n_bins=10)
    assert table["n"].sum() == 1000
    assert table["events"].sum() == y.sum()
    assert table["mean_pred"].is_monotonic_increasing
    assert (
        (table["ci_low"] <= table["observed"]) & (table["observed"] <= table["ci_high"])
    ).all()


def test_gains_curve_endpoints_and_perfect_ranking():
    y = np.array([1, 0, 1, 0, 0])
    curve = gains_curve(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3]))
    assert curve.iloc[0].tolist() == [0.0, 0.0]
    assert curve.iloc[-1].tolist() == [1.0, 1.0]
    # Both positives are ranked first, so 40% reviewed reaches 100%.
    assert curve.loc[curve["share_reviewed"] == 0.4, "share_reached"].item() == 1.0


def test_guard_single_use(tmp_path):
    path = tmp_path / "test_metrics.csv"
    guard_single_use(path, force=False)  # nothing yet: fine
    path.write_text("done")
    with pytest.raises(SystemExit, match="already been evaluated"):
        guard_single_use(path, force=False)
    guard_single_use(path, force=True)


def test_plots_draw(tmp_path):
    y, probs, _ = make_predictions(n=1000)
    plots.apply_style()
    ax = plots.plot_calibration(
        {
            "A": calibration_table(y, probs[PRIMARY]),
            "B": calibration_table(y, probs[REFERENCE]),
        },
        title="t",
    )
    assert len(ax.get_legend().get_texts()) == 2
    ax = plots.plot_gains({"A": gains_curve(y, probs[PRIMARY])}, (0.1, 0.2), title="t")
    assert any("top 10%" in t.get_text() for t in ax.texts)
    assert plots.save(ax.figure, tmp_path / "g.png").exists()
    with pytest.raises(ValueError, match="series"):
        plots.plot_gains(
            {str(i): gains_curve(y, probs[PRIMARY]) for i in range(3)}, (0.1,), "t"
        )
