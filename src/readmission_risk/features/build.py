"""Feature engineering: stateless row transforms + a fitted preprocessing step.

Two stages, split by whether they learn from data:

1. :func:`add_features` is **stateless**. Each output depends only on the same
   row (age band -> number, diagnosis code -> category, medication counts). It
   learns nothing, so it cannot leak information between train and test.
2. :func:`build_preprocessor` returns an unfitted scikit-learn
   ``ColumnTransformer``. Everything that learns from data (imputation, scaling,
   one-hot vocabularies, rare-level grouping) lives here, so it is fitted only on
   the training folds inside cross-validation.

Never derive features from a patient's *other* encounters in the dataset: a
readmission is itself another encounter, so such features leak the answer
(see ``notebooks/01_eda.ipynb``, section 4).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from readmission_risk.data.clean import AGE_BANDS, GROUP_COL, TARGET
from readmission_risk.features.icd9 import icd9_chapter

DiagGrouper = Callable[[object], str]

# All 21 diabetes drug columns (examide/citoglipton were dropped in cleaning).
DRUG_COLUMNS = (
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
)  # fmt: skip

# Drugs used in >= 1% of training encounters keep their own feature; the rest
# only contribute to the summary counts below.
COMMON_DRUGS = (
    "insulin", "metformin", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "glimepiride", "repaglinide",
)  # fmt: skip

DIAG_COLUMNS = ("diag_1", "diag_2", "diag_3")

# Pre-encounter utilisation counts: heavily right-skewed, so log1p-transformed.
COUNT_FEATURES = [
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "prior_visits_total",
]
NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "n_diabetes_drugs",
    "n_drug_changes",
    "age_ordinal",
]
CATEGORICAL_FEATURES = [
    "gender",
    "admission_type",
    "admission_source",
    "discharge_disposition",
    "payer_code",
    "medical_specialty",
    "A1Cresult",
    "max_glu_serum",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    *COMMON_DRUGS,
]
SENSITIVE_FEATURES = ["race"]  # optional model input; always kept for auditing

# Columns that must never be model inputs.
NON_FEATURES = frozenset({TARGET, GROUP_COL, "encounter_id", *DIAG_COLUMNS})

# One-hot levels seen in fewer than this share of training rows are pooled
# into a single "infrequent" level (learned on training folds only).
MIN_CATEGORY_SHARE = 0.005


def add_features(
    df: pd.DataFrame, diag_grouper: DiagGrouper = icd9_chapter
) -> pd.DataFrame:
    """Add stateless, row-wise engineered features.

    Args:
        df: Cleaned encounters (output of ``readmission_risk.data.clean``).
        diag_grouper: Function mapping one ICD-9 code to a category. Defaults
            to :func:`readmission_risk.features.icd9.icd9_chapter`; injectable
            so the pipeline can be tested independently.

    Returns:
        A new data frame with the engineered columns added.
    """
    out = df.copy()
    out["age_ordinal"] = pd.Categorical(
        out["age"], categories=AGE_BANDS, ordered=True
    ).codes.astype("int8")
    out["prior_visits_total"] = (
        out["number_inpatient"] + out["number_emergency"] + out["number_outpatient"]
    )
    drugs = out[list(DRUG_COLUMNS)]
    out["n_diabetes_drugs"] = (drugs != "No").sum(axis=1).astype("int8")
    out["n_drug_changes"] = drugs.isin(["Up", "Down"]).sum(axis=1).astype("int8")
    for col in DIAG_COLUMNS:
        out[f"{col}_group"] = out[col].map(diag_grouper)
    return out


def feature_columns(include_race: bool = False) -> list[str]:
    """Names of the model input columns.

    Args:
        include_race: Whether ``race`` is used as a model input. It is always
            retained in the data for fairness auditing either way.

    Returns:
        Ordered list of input column names.
    """
    cats = CATEGORICAL_FEATURES + (SENSITIVE_FEATURES if include_race else [])
    return COUNT_FEATURES + NUMERIC_FEATURES + cats


def _as_object(x: pd.DataFrame) -> pd.DataFrame:
    """Cast to object dtype so missing values impute cleanly as a category."""
    return x.astype("object").where(x.notna(), np.nan)


def build_preprocessor(include_race: bool = False) -> ColumnTransformer:
    """Create the (unfitted) preprocessing step for the model pipeline.

    - Counts: median-impute, ``log1p``, standardise.
    - Other numerics: median-impute, standardise.
    - Categoricals: missing -> ``"Missing"`` level, one-hot encode, pool rare
      levels; unseen levels at prediction time map to the infrequent bucket.

    Scaling does not change tree models but puts logistic-regression
    coefficients on a comparable scale.

    Args:
        include_race: Whether ``race`` is one of the inputs.

    Returns:
        A ``ColumnTransformer`` producing a dense float matrix with readable
        feature names (``get_feature_names_out``).
    """
    counts = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            (
                "as_object",
                FunctionTransformer(_as_object, feature_names_out="one-to-one"),
            ),
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_SHARE,
                    sparse_output=False,
                ),
            ),
        ]
    )
    cats = CATEGORICAL_FEATURES + (SENSITIVE_FEATURES if include_race else [])
    return ColumnTransformer(
        [
            ("counts", counts, COUNT_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, cats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
