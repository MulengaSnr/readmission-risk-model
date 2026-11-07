# Notebooks

Notebooks tell the story; reusable logic lives in `src/readmission_risk/` and is unit-tested.

| Notebook | Purpose |
|---|---|
| [01_eda.ipynb](01_eda.ipynb) | Exploratory analysis on the **training set only**: missing data, drivers of readmission, a leakage trap, fairness preview, implications for features |

Each notebook is paired with a `.py` file (jupytext percent format). Edit the `.py` for readable diffs in pull requests, then regenerate with `make eda`. Every notebook runs top to bottom from a fresh kernel.
