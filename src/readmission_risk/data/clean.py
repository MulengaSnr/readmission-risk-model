"""Clean raw encounters into an analysis-ready table.

Run from the project root (after downloading):

    uv run python -m readmission_risk.data.clean

Cleaning is deliberately limited to fixing types, applying cohort exclusions,
defining the target and dropping columns with no information. Choices that
depend on the model (imputation, encoding, feature selection) happen later,
inside the scikit-learn pipeline, so they are fitted on training data only.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from readmission_risk.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

TARGET = "readmitted_30d"
GROUP_COL = "patient_nbr"

# Discharge dispositions where a readmission is impossible or not meaningful:
# 11 Expired; 19-21 Expired (Medicaid hospice); 13-14 Hospice.
EXCLUDED_DISCHARGE_IDS = frozenset({11, 13, 14, 19, 20, 21})

# Columns with no usable information: weight is ~97% missing; the two drugs
# have a single value for every encounter.
DROP_COLUMNS = ("weight", "examide", "citoglipton")

# In the lab columns "None" means "test not performed", which is information,
# not a missing value.
LAB_COLUMNS = ("A1Cresult", "max_glu_serum")
LAB_NOT_MEASURED = "Not measured"

# Lookup labels that mean "unknown" and should become proper missing values.
UNKNOWN_LABELS = frozenset({"NULL", "Not Mapped", "Not Available", "Unknown/Invalid"})

ID_COLUMNS = ("admission_type_id", "discharge_disposition_id", "admission_source_id")

AGE_BANDS = [f"[{lo}-{lo + 10})" for lo in range(0, 100, 10)]


@dataclass
class CleaningSummary:
    """Row counts at each cleaning step (a cohort flow for the README)."""

    steps: list[dict[str, int | str]] = field(default_factory=list)

    def record(self, step: str, before: int, after: int) -> None:
        """Log and store how many rows a step removed."""
        removed = before - after
        self.steps.append({"step": step, "removed": removed, "remaining": after})
        logger.info("%-45s removed %6d  remaining %6d", step, removed, after)


def read_raw(path: Path) -> pd.DataFrame:
    """Read the raw encounter CSV, treating only ``?`` as missing.

    pandas treats the string ``"None"`` as missing by default. In this dataset
    ``None`` in the lab columns means the test was not performed, so the default
    NA list is switched off.

    Args:
        path: Path to ``diabetic_data.csv``.

    Returns:
        The raw data with ``?`` converted to missing values.
    """
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)


def parse_id_mappings(path: Path) -> dict[str, dict[int, str]]:
    """Parse ``IDS_mapping.csv``, which stacks three lookup tables in one file.

    Each table starts with a header row ``<column_name>,description`` and tables
    are separated by rows that are just a comma.

    Args:
        path: Path to ``IDS_mapping.csv``.

    Returns:
        ``{id_column: {code: label}}`` for each lookup table.

    Raises:
        ValueError: If no tables are found.
    """
    mappings: dict[str, dict[int, str]] = {}
    current: str | None = None
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    for code, label in raw.itertuples(index=False, name=None):
        code, label = code.strip(), label.strip()
        if not code:
            current = None  # separator row
        elif label == "description":
            current = code
            mappings[current] = {}
        elif current is not None:
            mappings[current][int(code)] = label
    if not mappings:
        raise ValueError(f"No lookup tables found in {path}")
    return mappings


def clean(
    df: pd.DataFrame, mappings: dict[str, dict[int, str]]
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Apply cohort exclusions, define the target and fix column types.

    Args:
        df: Raw encounters from :func:`read_raw`.
        mappings: Lookup tables from :func:`parse_id_mappings`.

    Returns:
        The cleaned encounters and a summary of rows removed at each step.
    """
    summary = CleaningSummary()
    out = df.copy()
    summary.record("Raw encounters", len(out), len(out))

    n = len(out)
    out = out[~out["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
    summary.record("Exclude death / hospice discharges", n, len(out))

    n = len(out)
    out = out[out["gender"] != "Unknown/Invalid"]
    summary.record("Exclude invalid sex", n, len(out))

    out[TARGET] = (out["readmitted"] == "<30").astype("int8")
    out = out.drop(columns=["readmitted", *DROP_COLUMNS])

    for col in LAB_COLUMNS:
        out[col] = out[col].replace("None", LAB_NOT_MEASURED)

    for col in ID_COLUMNS:
        labels = out[col].map(mappings[col])
        out[col.removesuffix("_id")] = labels.where(~labels.isin(UNKNOWN_LABELS))
    out = out.drop(columns=list(ID_COLUMNS))

    out["age"] = pd.Categorical(out["age"], categories=AGE_BANDS, ordered=True)

    return out.reset_index(drop=True), summary


def validate(df: pd.DataFrame) -> None:
    """Check invariants of the cleaned table and raise if any fail.

    Args:
        df: Output of :func:`clean`.

    Raises:
        ValueError: Listing every failed check.
    """
    problems = []
    if not df["encounter_id"].is_unique:
        problems.append("encounter_id is not unique")
    if not set(df[TARGET].unique()) <= {0, 1}:
        problems.append(f"{TARGET} is not binary")
    if "readmitted" in df.columns:
        problems.append("raw outcome column 'readmitted' still present (target leak)")
    if df[GROUP_COL].isna().any():
        problems.append(f"{GROUP_COL} has missing values")
    if df["age"].isna().any():
        problems.append("age has values outside the expected bands")
    if (df["gender"] == "Unknown/Invalid").any():
        problems.append("invalid gender values remain")
    for col in LAB_COLUMNS:
        if df[col].isna().any():
            problems.append(f"{col} has missing values ('None' must be 'Not measured')")
    if problems:
        raise ValueError("Cleaned data failed validation: " + "; ".join(problems))


def main() -> None:
    """Command-line entry point: raw CSVs -> parquet + cohort summary."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw"]
    out_dir = PROJECT_ROOT / config["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_raw(raw_dir / "diabetic_data.csv")
    mappings = parse_id_mappings(raw_dir / "IDS_mapping.csv")
    cleaned, summary = clean(df, mappings)
    validate(cleaned)

    out_path = out_dir / "encounters_clean.parquet"
    cleaned.to_parquet(out_path, index=False)
    summary_path = PROJECT_ROOT / "reports" / "cohort_flow.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    logger.info(
        "Wrote %s (%d rows, %d columns); %d patients; readmission rate %.2f%%",
        out_path.relative_to(PROJECT_ROOT),
        len(cleaned),
        cleaned.shape[1],
        cleaned[GROUP_COL].nunique(),
        100 * cleaned[TARGET].mean(),
    )


if __name__ == "__main__":
    main()
