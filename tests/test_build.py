import numpy as np
import pandas as pd
import pytest

from readmission_risk.config import PROJECT_ROOT
from readmission_risk.data.clean import AGE_BANDS, GROUP_COL, TARGET
from readmission_risk.features.build import (
    DRUG_COLUMNS,
    NON_FEATURES,
    add_features,
    build_preprocessor,
    feature_columns,
)


def stand_in_grouper(code):
    """Simple grouper so these tests do not depend on the ICD-9 exercise."""
    return "Missing" if pd.isna(code) else f"G{str(code)[0]}"


def make_encounters(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            GROUP_COL: rng.integers(0, n // 2, n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", None], n),
            "gender": rng.choice(["Female", "Male"], n),
            "age": pd.Categorical(
                rng.choice(AGE_BANDS[3:], n), categories=AGE_BANDS, ordered=True
            ),
            "admission_type": rng.choice(["Emergency", "Elective", None], n),
            "admission_source": rng.choice(["Emergency Room", "Physician Referral"], n),
            "discharge_disposition": rng.choice(["Home", "SNF"], n),
            "payer_code": rng.choice(["MC", "HM", None], n),
            "medical_specialty": rng.choice(["Cardiology", None], n),
            "A1Cresult": rng.choice(["Not measured", ">8", "Norm"], n),
            "max_glu_serum": rng.choice(["Not measured", ">200"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
            "time_in_hospital": rng.integers(1, 15, n),
            "num_lab_procedures": rng.integers(1, 100, n),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 40, n),
            "number_diagnoses": rng.integers(1, 10, n),
            "number_inpatient": rng.poisson(0.6, n),
            "number_emergency": rng.poisson(0.2, n),
            "number_outpatient": rng.poisson(0.4, n),
            "diag_1": rng.choice(["428", "250.83", "V57", None], n),
            "diag_2": rng.choice(["414", "786"], n),
            "diag_3": rng.choice(["250", None], n),
            TARGET: (rng.random(n) < 0.11).astype("int8"),
        }
    )
    for drug in DRUG_COLUMNS:
        df[drug] = rng.choice(
            ["No", "Steady", "Up", "Down"], n, p=[0.7, 0.2, 0.05, 0.05]
        )
    return df


@pytest.fixture
def encounters():
    return make_encounters()


def test_add_features_row_values():
    df = make_encounters(n=3)
    df.loc[0, "age"] = "[70-80)"
    df.loc[0, ["number_inpatient", "number_emergency", "number_outpatient"]] = [2, 1, 3]
    df.loc[0, list(DRUG_COLUMNS)] = "No"
    df.loc[0, ["insulin", "metformin", "glipizide"]] = ["Up", "Steady", "Down"]
    df.loc[0, "diag_1"] = "428"
    row = add_features(df, stand_in_grouper).iloc[0]
    assert row["age_ordinal"] == 7
    assert row["prior_visits_total"] == 6
    assert row["n_diabetes_drugs"] == 3
    assert row["n_drug_changes"] == 2
    assert row["diag_1_group"] == "G4"


def test_add_features_handles_missing_diagnosis(encounters):
    out = add_features(encounters, stand_in_grouper)
    missing = encounters["diag_1"].isna()
    assert (out.loc[missing, "diag_1_group"] == "Missing").all()


def test_add_features_does_not_mutate_input(encounters):
    before = encounters.copy()
    add_features(encounters, stand_in_grouper)
    pd.testing.assert_frame_equal(encounters, before)


def test_feature_columns_exclude_ids_target_and_raw_codes():
    for include_race in (False, True):
        assert not NON_FEATURES & set(feature_columns(include_race))


def test_race_is_opt_in():
    assert "race" not in feature_columns()
    assert "race" in feature_columns(include_race=True)


@pytest.fixture
def fitted(encounters):
    df = add_features(encounters, stand_in_grouper)
    train, test = df.iloc[:300], df.iloc[300:]
    pre = build_preprocessor().fit(train)
    return pre, train, test


def test_preprocessor_output_is_finite_and_aligned(fitted):
    pre, _, test = fitted
    x = pre.transform(test)
    assert x.shape[0] == len(test)
    assert np.isfinite(x).all()
    assert x.shape[1] == len(pre.get_feature_names_out())


def test_preprocessor_never_sees_forbidden_columns(fitted):
    pre, _, _ = fitted
    consumed = {
        col
        for _, transformer, cols in pre.transformers_
        if transformer != "drop"  # the 'remainder' entry lists dropped columns
        for col in cols
    }
    assert not NON_FEATURES & consumed
    assert "race" not in consumed


def test_missing_categories_become_their_own_level(fitted):
    pre, _, _ = fitted
    assert "payer_code_Missing" in pre.get_feature_names_out()


def test_unseen_category_at_prediction_does_not_fail(fitted):
    pre, _, test = fitted
    test = test.copy()
    test.loc[test.index[0], "discharge_disposition"] = "Brand new destination"
    assert np.isfinite(pre.transform(test)).all()


def test_rare_levels_are_pooled():
    df = add_features(make_encounters(n=1000), stand_in_grouper)
    df.loc[df.index[0], "payer_code"] = "RARE"  # 1 in 1000 < 0.5% threshold
    names = build_preprocessor().fit(df).get_feature_names_out()
    assert "payer_code_RARE" not in names
    assert "payer_code_infrequent_sklearn" in names


def test_preprocessor_fit_uses_training_rows_only(encounters):
    """Scaler statistics must come from the data passed to fit, not elsewhere."""
    df = add_features(encounters, stand_in_grouper)
    train = df.iloc[:300]
    pre = build_preprocessor().fit(train)
    scaler = pre.named_transformers_["numeric"].named_steps["scale"]
    idx = list(pre.transformers_[1][2]).index("time_in_hospital")
    assert scaler.mean_[idx] == pytest.approx(train["time_in_hospital"].mean())


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"


def _exercise_done():
    from readmission_risk.features.icd9 import icd9_chapter

    try:
        icd9_chapter("428")
    except NotImplementedError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="data not prepared")
@pytest.mark.skipif(not _exercise_done(), reason="ICD-9 exercise not implemented yet")
def test_real_training_data_end_to_end():
    train = add_features(pd.read_parquet(TRAIN_PATH))
    x = build_preprocessor(include_race=True).fit_transform(train)
    assert x.shape[0] == len(train)
    assert np.isfinite(x).all()
