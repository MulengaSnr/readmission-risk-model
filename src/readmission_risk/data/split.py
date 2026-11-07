"""Split cleaned encounters into patient-disjoint train and test sets.

Run from the project root (after cleaning):

    uv run python -m readmission_risk.data.split

The split is made once, before any exploration, and the test set is not
looked at again until final evaluation. Splitting by patient (not encounter)
stops the same person appearing on both sides, which would inflate scores.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from readmission_risk.config import PROJECT_ROOT, load_config
from readmission_risk.data.clean import GROUP_COL, TARGET

logger = logging.getLogger(__name__)


def patient_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split encounters so that no patient appears in both sets.

    Uses one fold of ``StratifiedGroupKFold``: groups (patients) are kept
    together, and the readmission rate is kept as similar as possible on each
    side.

    Args:
        df: Cleaned encounters containing ``TARGET`` and ``GROUP_COL``.
        test_size: Approximate share of encounters in the test set. Must give a
            whole number of folds, e.g. 0.2 -> 5 folds, 0.25 -> 4 folds.
        seed: Random seed for reproducibility.

    Returns:
        ``(train, test)`` data frames with reset indexes.

    Raises:
        ValueError: If ``test_size`` does not correspond to a whole number of
            folds of at least 2.
    """
    n_splits = round(1 / test_size)
    if n_splits < 2 or abs(n_splits * test_size - 1) > 1e-9:
        raise ValueError(f"test_size must be 1/k for integer k >= 2, got {test_size}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df[GROUP_COL]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    overlap = set(train[GROUP_COL]) & set(test[GROUP_COL])
    if overlap:  # defensive: should be impossible with a group splitter
        raise RuntimeError(f"{len(overlap)} patients appear in both train and test")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    """Command-line entry point: clean parquet -> train/test parquet."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    processed = PROJECT_ROOT / config["paths"]["processed"]
    df = pd.read_parquet(processed / "encounters_clean.parquet")
    train, test = patient_split(df, config["split"]["test_size"], config["seed"])
    for name, part in (("train", train), ("test", test)):
        part.to_parquet(processed / f"{name}.parquet", index=False)
        logger.info(
            "%-5s %6d encounters  %6d patients  readmission rate %.2f%%",
            name,
            len(part),
            part[GROUP_COL].nunique(),
            100 * part[TARGET].mean(),
        )


if __name__ == "__main__":
    main()
