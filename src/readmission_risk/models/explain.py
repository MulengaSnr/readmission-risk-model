"""Explain the final XGBoost model with SHAP values (step 6b).

Run from the project root (after ``make evaluate``):

    uv run python -m readmission_risk.models.explain

Design:

- SHAP values come from XGBoost's built-in TreeSHAP (``pred_contribs=True``),
  which is exact for tree ensembles and needs no extra dependency.
- Values are in log-odds. For each encounter, the contributions plus the bias
  term add up to the model's raw output, which a test checks.
- One-hot columns are summed back to their original feature (SHAP values are
  additive), so "discharge disposition" gets one importance score rather than
  one per destination.
- Computed on the test set: explanation needs no outcomes, and these are the
  patients whose predictions were evaluated.
- SHAP describes what the *model* relies on. It is not evidence that a feature
  *causes* readmission.
"""

from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.pipeline import Pipeline

from readmission_risk.config import PROJECT_ROOT, load_config
from readmission_risk.features.build import add_features
from readmission_risk.visualisation import plots

logger = logging.getLogger(__name__)

# Readable names for the chart; anything missing falls back to the column name.
FEATURE_LABELS = {
    "number_inpatient": "Inpatient stays, previous year",
    "number_emergency": "Emergency visits, previous year",
    "number_outpatient": "Outpatient visits, previous year",
    "prior_visits_total": "All prior visits, previous year",
    "discharge_disposition": "Discharge destination",
    "number_diagnoses": "Number of diagnoses",
    "time_in_hospital": "Length of stay",
    "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of procedures",
    "num_medications": "Number of medications",
    "admission_source": "Admission source",
    "admission_type": "Admission type",
    "medical_specialty": "Admitting specialty",
    "payer_code": "Payer",
    "age_ordinal": "Age band",
    "diag_1_group": "Primary diagnosis group",
    "diag_2_group": "Secondary diagnosis group",
    "diag_3_group": "Additional diagnosis group",
    "n_diabetes_drugs": "Number of diabetes drugs",
    "n_drug_changes": "Diabetes drug dose changes",
    "A1Cresult": "HbA1c result",
    "max_glu_serum": "Glucose serum result",
    "diabetesMed": "On diabetes medication",
    "change": "Diabetes medication changed",
    "gender": "Sex",
}


def column_to_feature(output_columns: list[str], features: list[str]) -> list[str]:
    """Map each preprocessed column back to the input feature it came from.

    Numeric columns keep their name; one-hot columns are named
    ``{feature}_{category}``. The longest matching feature name wins, so a
    feature whose name starts another feature's name cannot steal its columns.

    Args:
        output_columns: ``get_feature_names_out()`` of the preprocessor.
        features: The input feature names.

    Returns:
        The source feature for each output column, in order.

    Raises:
        ValueError: If a column matches no feature.
    """
    by_length = sorted(features, key=len, reverse=True)
    mapped = []
    for column in output_columns:
        source = next(
            (f for f in by_length if column == f or column.startswith(f + "_")), None
        )
        if source is None:
            raise ValueError(f"Column {column!r} matches no input feature")
        mapped.append(source)
    return mapped


def shap_by_feature(model: Pipeline, x: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP values per input feature, in log-odds.

    Args:
        model: Fitted pipeline with steps ``prep`` and ``model`` (XGBoost).
        x: Feature table (output of ``add_features``).

    Returns:
        ``(values, bias)``: one column per input feature (one-hot columns
        summed) and the expected log-odds over the training data.
    """
    prep, booster = model["prep"], model["model"].get_booster()
    matrix = prep.transform(x)
    contribs = booster.predict(xgboost.DMatrix(matrix), pred_contribs=True)
    columns = list(prep.get_feature_names_out())
    used = [c for name, _, cols in prep.transformers_ if name != "remainder"
            for c in cols]  # fmt: skip
    sources = column_to_feature(columns, used)
    per_column = pd.DataFrame(contribs[:, :-1], columns=columns, index=x.index)
    values = per_column.T.groupby(sources, sort=False).sum().T
    return values, float(contribs[0, -1])


def importance(values: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, largest first.

    Args:
        values: Output of :func:`shap_by_feature`.

    Returns:
        ``feature``, ``label``, ``mean_abs_shap`` and its share of the total.
    """
    mean_abs = values.abs().mean().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "feature": mean_abs.index,
            "label": [FEATURE_LABELS.get(f, f) for f in mean_abs.index],
            "mean_abs_shap": mean_abs.to_numpy(),
            "share": (mean_abs / mean_abs.sum()).to_numpy(),
        }
    )


def dependence_table(raw: pd.Series, shap: pd.Series, cap: int = 5) -> pd.DataFrame:
    """SHAP value by count of a discrete feature, with the top values pooled.

    Args:
        raw: Raw feature values (e.g. number of prior inpatient stays).
        shap: SHAP values of that feature, aligned with ``raw``.
        cap: Counts at or above this are pooled into one ``"{cap}+"`` level.

    Returns:
        One row per level: ``level``, ``n``, median and quartiles of SHAP.
    """
    levels = raw.clip(upper=cap).astype(int)
    table = (
        pd.DataFrame({"level": levels, "shap": shap})
        .groupby("level")["shap"]
        .agg(n="size", median="median", q1=lambda s: s.quantile(0.25),
             q3=lambda s: s.quantile(0.75))
    )  # fmt: skip
    table.index = [f"{i}+" if i == cap else str(i) for i in table.index]
    return table.rename_axis("level").reset_index()


def main() -> None:
    """Command-line entry point: SHAP importance and dependence on the test set."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config()
    model_path = PROJECT_ROOT / "models" / "xgboost.joblib"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found: run `make evaluate` first")
    model = joblib.load(model_path)
    test = pd.read_parquet(PROJECT_ROOT / config["paths"]["processed"] / "test.parquet")
    x = add_features(test)

    values, bias = shap_by_feature(model, x)
    table = importance(values)
    metrics = PROJECT_ROOT / config["paths"]["metrics"]
    figures = PROJECT_ROOT / config["paths"]["figures"]
    table.to_csv(metrics / "shap_importance.csv", index=False, float_format="%.5f")
    dependence = dependence_table(x["number_inpatient"], values["number_inpatient"])
    dependence.to_csv(metrics / "shap_prior_inpatient.csv", index=False,
                      float_format="%.5f")  # fmt: skip

    plots.apply_style()
    ax = plots.plot_importance(
        table.head(12),
        title="What drives the model's risk scores (test set)",
    )
    plots.save(ax.figure, figures / "shap_importance.png")
    plt.close(ax.figure)
    ax = plots.plot_dependence(
        dependence,
        title="Each prior inpatient stay pushes predicted risk up",
        xlabel="Inpatient stays in the previous year",
    )
    plots.save(ax.figure, figures / "shap_prior_inpatient.png")
    logger.info("Bias (expected log-odds) %.3f -> %.1f%% risk", bias,
                100 / (1 + np.exp(-bias)))  # fmt: skip
    print(table.head(12).round(4).to_string(index=False))
    print(dependence.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
