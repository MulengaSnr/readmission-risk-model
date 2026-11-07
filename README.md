# Hospital Readmission Risk Model

Predicting which diabetic inpatients will be readmitted within 30 days, so a care team with limited follow-up capacity knows whom to prioritise.

![CI](https://github.com/MulengaSnr/readmission-risk-model/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

**Status:** In development. Step 1 of 7 (data acquisition) complete. Results will be added once the model is evaluated, and none are claimed yet.

## At a glance
| | |
|---|---|
| **Problem** | Which discharged patients are at highest risk of unplanned readmission within 30 days? |
| **Data** | UCI Diabetes 130-US Hospitals 1999–2008: 101,766 encounters, CC BY 4.0 |
| **Approach** | Patient-grouped cross-validation; logistic regression, random forest and XGBoost vs a baseline; calibrated probabilities; SHAP; fairness audit |
| **Result** | _To be reported from the held-out test set_ |
| **Stack** | Python 3.12, pandas, scikit-learn, XGBoost, SHAP, fairlearn, uv, pytest, GitHub Actions |

## 1. Business problem
Unplanned readmissions within 30 days are costly, often avoidable and widely used as a quality indicator. The NHS, for example, publishes 30-day emergency readmission rates. Hospitals can offer only a limited number of post-discharge interventions, such as pharmacist medicine reviews, follow-up calls and community nursing. This project estimates each patient's readmission risk at discharge and explains it, so those interventions go where they are most likely to help.

## 2. Objectives
- Build a model that ranks discharges by readmission risk better than a no-skill baseline, measured by PR-AUC on a patient-disjoint test set.
- Produce **calibrated** probabilities that staff can interpret.
- Translate performance into capacity terms: if the team can review the top 10% or 20% of discharges, how many readmissions would they reach?
- Explain predictions (SHAP) and audit performance across age, sex and race.

## 3. Data
Inpatient encounters with a diabetes diagnosis from 130 US hospitals, 1999–2008, published by the UCI Machine Learning Repository under CC BY 4.0. Each row is an encounter. Patients can appear more than once, so all splits are grouped by patient to prevent leakage. Full provenance, licence, checksum and data-protection notes are in [data/README.md](data/README.md).

## 4. Architecture
```mermaid
flowchart LR
    A[UCI archive] -->|download.py<br/>SHA-256 verified| B[data/raw]
    B -->|clean.py<br/>exclusions, types| C[Clean encounters]
    C -->|features/build.py| D[Feature table]
    D -->|models/train.py<br/>grouped CV + tuning| E[Trained pipeline]
    E -->|evaluate.py| F[Metrics, calibration,<br/>SHAP, fairness]
```

## 5. Methodology (planned)
- **Leakage control:** train/test split and all cross-validation grouped by `patient_nbr`. All preprocessing sits inside a scikit-learn `Pipeline` fitted on training folds only.
- **Exclusions:** encounters ending in death or hospice discharge, where readmission is impossible.
- **Models:** dummy baseline → logistic regression → random forest → XGBoost, with hyperparameter tuning in grouped cross-validation. The test set is used once.
- **Metrics:** PR-AUC (primary, because positives are rare), ROC-AUC, Brier score, calibration curve, and precision/recall at capacity thresholds.

_Sections on exploratory analysis, results, explainability, fairness, business interpretation and limitations will be added as each step is completed._

## Run it yourself
```bash
git clone https://github.com/MulengaSnr/readmission-risk-model.git
cd readmission-risk-model
uv sync                                            # Python 3.12 + dependencies
uv run python -m readmission_risk.data.download    # fetch + verify the dataset
make check                                         # lint + tests
```

## Project structure
```
├── configs/default.yaml     dataset URL, checksum, seed, paths
├── data/                    raw/processed (gitignored) + provenance README
├── docs/                    model card, checklist, standards
├── notebooks/               narrative analysis
├── reports/                 findings + figures
├── src/readmission_risk/    data · features · models · visualisation
└── tests/
```

## Progress
- [x] 1. Data acquisition with checksum verification
- [ ] 2. Cleaning and exclusions
- [ ] 3. Exploratory analysis
- [ ] 4. Feature engineering
- [ ] 5. Modelling and grouped cross-validation
- [ ] 6. Evaluation, calibration, SHAP and fairness
- [ ] 7. Documentation and review

## Author
**Enock Mulenga**, Data Scientist · [LinkedIn](https://www.linkedin.com/in/enock-mulenga-8472ba137) · [Portfolio](https://mulengasnr.github.io)
