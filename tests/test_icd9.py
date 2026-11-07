"""Specification tests for the ICD-9 grouping exercise.

Skipped automatically until ``icd9_chapter`` is implemented.
"""

import math

import pandas as pd
import pytest

from readmission_risk.features.icd9 import CATEGORIES, icd9_chapter

try:
    icd9_chapter("428")
except NotImplementedError:
    pytest.skip("ICD-9 exercise not implemented yet", allow_module_level=True)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # Circulatory: 390-459 and 785
        ("390", "Circulatory"),
        ("428", "Circulatory"),
        ("459.9", "Circulatory"),
        ("785", "Circulatory"),
        # Respiratory: 460-519 and 786
        ("460", "Respiratory"),
        ("486", "Respiratory"),
        ("519", "Respiratory"),
        ("786", "Respiratory"),
        # Digestive: 520-579 and 787
        ("520", "Digestive"),
        ("578", "Digestive"),
        ("787", "Digestive"),
        # Diabetes: 250.xx (but not its neighbours)
        ("250", "Diabetes"),
        ("250.01", "Diabetes"),
        ("250.83", "Diabetes"),
        ("249", "Other"),
        ("251", "Other"),
        # Injury: 800-999
        ("800", "Injury"),
        ("996", "Injury"),
        ("999", "Injury"),
        # Musculoskeletal: 710-739
        ("710", "Musculoskeletal"),
        ("715", "Musculoskeletal"),
        ("739", "Musculoskeletal"),
        # Genitourinary: 580-629 and 788
        ("580", "Genitourinary"),
        ("599", "Genitourinary"),
        ("629", "Genitourinary"),
        ("788", "Genitourinary"),
        # Neoplasms: 140-239
        ("140", "Neoplasms"),
        ("174", "Neoplasms"),
        ("239.99", "Neoplasms"),
        # Boundaries just outside a range fall to Other
        ("139", "Other"),
        ("240", "Other"),
        ("389", "Other"),
        ("630", "Other"),
        ("709", "Other"),
        ("740", "Other"),
        ("784", "Other"),
        ("789", "Other"),
        ("799", "Other"),
        # Supplementary codes
        ("V57", "Other"),
        ("v45", "Other"),
        ("E909", "Other"),
        ("e888", "Other"),
        # Whitespace is tolerated
        (" 414 ", "Circulatory"),
        ("\t250.02\n", "Diabetes"),
    ],
)
def test_maps_code_to_category(code, expected):
    assert icd9_chapter(code) == expected


@pytest.mark.parametrize("missing", [None, math.nan, float("nan"), pd.NA, "", "   "])
def test_missing_values(missing):
    assert icd9_chapter(missing) == "Missing"


@pytest.mark.parametrize(
    "bad", ["abc", "4x8", "--", "250..1", "nan", "inf", "1e2", "-5", "1000", "428.123"]
)
def test_garbage_raises(bad):
    with pytest.raises(ValueError):
        icd9_chapter(bad)


def test_always_returns_a_known_category():
    for code in ["1", "100", "250", "300", "450", "500", "600", "700", "900", "V1"]:
        assert icd9_chapter(code) in CATEGORIES


def test_numeric_input_is_accepted():
    # Codes can arrive as floats if a column is read without a string dtype.
    assert icd9_chapter(428.0) == "Circulatory"
    assert icd9_chapter(250.83) == "Diabetes"
