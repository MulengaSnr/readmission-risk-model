# Exercise 1: Group ICD-9 diagnosis codes

**Estimated time:** 1–2 hours · **File:** `src/readmission_risk/features/icd9.py` · **Tests:** `tests/test_icd9.py`

## Why this matters
Each encounter has up to three diagnosis codes (`diag_1`–`diag_3`), with about 700 distinct values in `diag_1` alone. Most codes are rare, so as raw categories they are noise: a model can't learn much from a code seen 12 times. Grouping them into clinical categories (circulatory, respiratory, diabetes, …) gives the model a stable signal. It also follows the published method (Strack et al., 2014, Table 2), so results can be compared with the literature.

This is the same kind of work you did on SmartCare data quality: turn messy coded values into something reliable, with rules you can defend.

## Your task
Implement `icd9_chapter(code) -> str`. The docstring has the full rules table. In short:

| Category | Codes |
|---|---|
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Diabetes | 250.xx |
| Injury | 800–999 |
| Musculoskeletal | 710–739 |
| Genitourinary | 580–629, 788 |
| Neoplasms | 140–239 |
| Other | everything else, including V and E codes |
| Missing | `None`, `NaN`, empty or blank string |

Non-numeric values that aren't V/E codes must raise `ValueError`.

## How to work
```bash
git switch main && git pull
git switch -c exercise/icd9-grouping
uv run pytest tests/test_icd9.py -v      # skipped until you start
```
1. Replace the `raise NotImplementedError` line with your implementation.
2. Run the tests. Make them pass one group at a time.
3. Run `make check` (lint + all tests) before committing.
4. Commit, push the branch and open a pull request. I'll review it like a code review at work.

## Hints (read only if stuck)
<details><summary>Hint 1: order of checks</summary>

Handle missing values first, then V/E codes, then parse the number. Check **Diabetes before the ranges**: 250 falls inside 240–279, which would otherwise be "Other".
</details>

<details><summary>Hint 2: detecting NaN</summary>

`NaN != NaN`, so `code != code` is true only for NaN. `math.isnan` works on floats but fails on strings. `pandas.isna(code)` handles both.
</details>

<details><summary>Hint 3: parsing</summary>

`float("250.83")` gives 250.83, and `int(...)` then gives the whole-number part. `float("250..1")` raises `ValueError`, which is exactly the behaviour the tests want.
</details>

<details><summary>Hint 4: avoiding a long if/elif chain</summary>

A tuple of `(low, high, category)` rows plus a dict for the single codes (785, 786, 787, 788) keeps the rules readable as data, much like the table above.
</details>

## What I will review
- **Correctness:** all tests pass, and boundaries are handled.
- **Readability:** could a colleague check your rules against the table in 30 seconds?
- **Robustness:** whitespace, case and NaN are handled without try/except around everything.
- **Style:** type hints, no magic numbers scattered through the code, passes `ruff`.
