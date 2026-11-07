"""Group ICD-9-CM diagnosis codes into clinical categories.

EXERCISE (Enock): implement :func:`icd9_chapter`. The full brief is in
``docs/exercises/01_icd9_grouping.md``, and the tests in ``tests/test_icd9.py``
are the specification. They are skipped until this function stops raising
``NotImplementedError``.

Run just these tests while you work:

    uv run pytest tests/test_icd9.py -v
"""

from __future__ import annotations

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
    Missing             ``None``, ``NaN`` or an empty/blank string
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
    raise NotImplementedError("Exercise: implement icd9_chapter (see docstring)")
