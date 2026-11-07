import pandas as pd
import pytest

from readmission_risk.config import PROJECT_ROOT
from readmission_risk.data.clean import (
    DROP_COLUMNS,
    EXCLUDED_DISCHARGE_IDS,
    LAB_NOT_MEASURED,
    TARGET,
    clean,
    parse_id_mappings,
    read_raw,
    validate,
)

MAPPING_CSV = """admission_type_id,description
1,Emergency
6,NULL
,
discharge_disposition_id,description
1,Discharged to home
11,Expired
13,Hospice / home
25,Not Mapped
,
admission_source_id,description
1, Physician Referral
7, Emergency Room
"""


@pytest.fixture
def mappings(tmp_path):
    path = tmp_path / "IDS_mapping.csv"
    path.write_text(MAPPING_CSV)
    return parse_id_mappings(path)


def _raw_row(**overrides):
    row = {
        "encounter_id": 1,
        "patient_nbr": 100,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "weight": None,
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "A1Cresult": "None",
        "max_glu_serum": ">200",
        "examide": "No",
        "citoglipton": "No",
        "readmitted": "NO",
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw():
    return pd.DataFrame(
        [
            _raw_row(encounter_id=1, readmitted="<30"),
            _raw_row(encounter_id=2, readmitted=">30", admission_type_id=6),
            _raw_row(encounter_id=3, discharge_disposition_id=11),  # expired
            _raw_row(encounter_id=4, discharge_disposition_id=13),  # hospice
            _raw_row(encounter_id=5, gender="Unknown/Invalid"),
            _raw_row(encounter_id=6, discharge_disposition_id=25, A1Cresult=">8"),
        ]
    )


def test_read_raw_keeps_none_and_converts_question_mark(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("race,A1Cresult\n?,None\nAsian,>8\n")
    df = read_raw(path)
    assert df["race"].isna().tolist() == [True, False]
    assert df["A1Cresult"].tolist() == ["None", ">8"]


def test_parse_id_mappings_splits_tables_and_strips(mappings):
    assert set(mappings) == {
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    }
    assert mappings["discharge_disposition_id"][11] == "Expired"
    assert mappings["admission_source_id"][7] == "Emergency Room"


def test_parse_id_mappings_empty_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(",\n")
    with pytest.raises(ValueError):
        parse_id_mappings(path)


def test_clean_applies_exclusions(raw, mappings):
    out, summary = clean(raw, mappings)
    assert sorted(out["encounter_id"]) == [1, 2, 6]
    removed = {s["step"]: s["removed"] for s in summary.steps}
    assert removed["Exclude death / hospice discharges"] == 2
    assert removed["Exclude invalid sex"] == 1
    assert summary.steps[-1]["remaining"] == 3


def test_clean_defines_binary_target_and_drops_raw_outcome(raw, mappings):
    out, _ = clean(raw, mappings)
    assert dict(zip(out["encounter_id"], out[TARGET], strict=True)) == {
        1: 1,
        2: 0,
        6: 0,
    }
    assert "readmitted" not in out.columns
    assert not set(DROP_COLUMNS) & set(out.columns)


def test_clean_recodes_lab_none(raw, mappings):
    out, _ = clean(raw, mappings)
    assert out.set_index("encounter_id").loc[1, "A1Cresult"] == LAB_NOT_MEASURED
    assert out.set_index("encounter_id").loc[6, "A1Cresult"] == ">8"


def test_clean_maps_ids_and_unknowns_to_missing(raw, mappings):
    out = clean(raw, mappings)[0].set_index("encounter_id")
    assert out.loc[1, "admission_type"] == "Emergency"
    assert pd.isna(out.loc[2, "admission_type"])  # "NULL"
    assert pd.isna(out.loc[6, "discharge_disposition"])  # "Not Mapped"
    assert out.loc[1, "admission_source"] == "Emergency Room"
    assert not any(c.endswith("_id") and c != "encounter_id" for c in out.columns)


def test_clean_age_is_ordered(raw, mappings):
    out, _ = clean(raw, mappings)
    assert out["age"].cat.ordered
    assert out["age"].cat.categories[0] == "[0-10)"


def test_clean_does_not_mutate_input(raw, mappings):
    before = raw.copy()
    clean(raw, mappings)
    pd.testing.assert_frame_equal(raw, before)


def test_validate_passes_on_clean_output(raw, mappings):
    validate(clean(raw, mappings)[0])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.assign(readmitted="NO"), "target leak"),
        (lambda d: d.assign(**{TARGET: 2}), "not binary"),
        (lambda d: d.assign(A1Cresult=None), "Not measured"),
        (lambda d: pd.concat([d, d.iloc[[0]]]), "not unique"),
    ],
)
def test_validate_catches_problems(raw, mappings, mutate, message):
    bad = mutate(clean(raw, mappings)[0])
    with pytest.raises(ValueError, match=message):
        validate(bad)


RAW_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.mark.integration
@pytest.mark.skipif(
    not (RAW_DIR / "diabetic_data.csv").exists(), reason="dataset not downloaded"
)
def test_real_data_cleans_and_validates():
    df = read_raw(RAW_DIR / "diabetic_data.csv")
    out, _ = clean(df, parse_id_mappings(RAW_DIR / "IDS_mapping.csv"))
    validate(out)
    assert not out["discharge_disposition"].isin(["Expired", "Hospice / home"]).any()
    assert (
        len(out)
        == len(df)
        - df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).sum()
        - (df["gender"] == "Unknown/Invalid").sum()
        + (
            df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
            & (df["gender"] == "Unknown/Invalid")
        ).sum()
    )
