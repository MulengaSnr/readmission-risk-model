"""Group ICD-9-CM diagnosis codes into clinical categories.

Implements the grouping of Strack et al. (2014), Table 2. The brief for this
module is ``docs/exercises/01_icd9_grouping.md`` and the specification is
``tests/test_icd9.py``.
"""

from __future__ import annotations

import re

import pandas as pd

# The categories used by Strack et al. (2014), Table 2, plus "Missing".
CATEGORIES = (
    "Circulatory",
    "Respiratory",
    "Digestive",
    "Diabetes",
    "Injury",
    "Musculoskeletal",
    "Genitourinary",
    "Neoplasms",
    "Other",
    "Missing",
)

DIABETES_CODE = 250

# Inclusive ranges on the whole-number part of the code: (low, high, category).
# Checked after DIABETES_CODE, because 250 lies inside the unlisted 240-279 block.
RANGES: tuple[tuple[int, int, str], ...] = (
    (140, 239, "Neoplasms"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (710, 739, "Musculoskeletal"),
    (800, 999, "Injury"),
)

# Symptom codes that Strack et al. assign to an organ-system category.
SINGLE_CODES: dict[int, str] = {
    785: "Circulatory",
    786: "Respiratory",
    787: "Digestive",
    788: "Genitourinary",
}

# A numeric ICD-9-CM code: 1-3 digits, optionally up to 2 decimals.
# Stricter than float(), which would also accept "nan", "inf", "1e2" or "-5".
_NUMERIC_CODE = re.compile(r"\d{1,3}(?:\.\d{1,2})?")


def icd9_chapter(code: str | float | None) -> str:
    """Map one ICD-9-CM diagnosis code to a clinical category.

    Rules (Strack et al., 2014, Table 2):

    ==================  =====================================
    Category            ICD-9 codes
    ==================  =====================================
    Circulatory         390–459, 785
    Respiratory         460–519, 786
    Digestive           520–579, 787
    Diabetes            250.xx
    Injury              800–999
    Musculoskeletal     710–739
    Genitourinary       580–629, 788
    Neoplasms           140–239
    Other               everything else, including V and E codes
    Missing             ``None``, ``NaN``, ``pd.NA`` or an empty/blank string
    ==================  =====================================

    Details:
        - Codes arrive as strings such as ``"428"``, ``"250.83"``, ``"V57"``,
          ``"E909"``; they may have surrounding whitespace.
        - Ranges are inclusive and apply to the whole number part, so
          ``"459.9"`` is Circulatory and ``"239.99"`` is Neoplasms.
        - Codes starting with ``V`` or ``E`` (either case) are "Other".
        - Anything else that is not a valid number raises ``ValueError``
          (silently mapping garbage to "Other" would hide data problems).

    Args:
        code: A single diagnosis code.

    Returns:
        One of :data:`CATEGORIES`.

    Raises:
        ValueError: If ``code`` is not missing, not a V/E code and not numeric.
    """
    # pd.isna covers None, float NaN, numpy NaN and pd.NA in one check.
    if not isinstance(code, str) and pd.isna(code):
        return "Missing"

    text = str(code).strip().upper()
    if not text:
        return "Missing"
    if text[0] in "VE":
        return "Other"
    if not _NUMERIC_CODE.fullmatch(text):
        raise ValueError(f"Not a valid ICD-9-CM code: {code!r}")

    number = int(float(text))
    if number == DIABETES_CODE:
        return "Diabetes"
    if number in SINGLE_CODES:
        return SINGLE_CODES[number]
    for low, high, category in RANGES:
        if low <= number <= high:
            return category
    return "Other"
