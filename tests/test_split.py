import numpy as np
import pandas as pd
import pytest

from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.data.split import patient_split


@pytest.fixture
def encounters():
    rng = np.random.default_rng(0)
    patients = rng.integers(0, 2000, size=6000)  # many patients with repeats
    return pd.DataFrame(
        {
            "encounter_id": np.arange(6000),
            GROUP_COL: patients,
            TARGET: (rng.random(6000) < 0.11).astype("int8"),
        }
    )


def test_no_patient_in_both_sets(encounters):
    train, test = patient_split(encounters)
    assert not set(train[GROUP_COL]) & set(test[GROUP_COL])


def test_all_rows_used_once(encounters):
    train, test = patient_split(encounters)
    combined = pd.concat([train, test])["encounter_id"]
    assert len(combined) == len(encounters)
    assert combined.is_unique


def test_sizes_and_stratification(encounters):
    train, test = patient_split(encounters, test_size=0.2)
    assert len(test) / len(encounters) == pytest.approx(0.2, abs=0.03)
    assert test[TARGET].mean() == pytest.approx(encounters[TARGET].mean(), abs=0.015)


def test_reproducible(encounters):
    a, _ = patient_split(encounters, seed=7)
    b, _ = patient_split(encounters, seed=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("bad", [0.3, 0.6, 1.0])
def test_rejects_non_fold_sizes(encounters, bad):
    with pytest.raises(ValueError):
        patient_split(encounters, test_size=bad)
