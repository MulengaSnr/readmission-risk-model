# Model card: 30-day readmission risk (XGBoost)

_Structure based on Mitchell et al. (2019), "Model Cards for Model Reporting". Every number below comes from files in `reports/metrics/`._

## Model details
- **Type:** XGBoost binary classifier (gradient-boosted trees, `hist` method) inside a scikit-learn `Pipeline` with the project's preprocessor (imputation, `log1p` on counts, scaling, one-hot encoding with rare levels pooled).
- **Settings:** 659 trees, learning rate 0.019, max depth 5, min child weight 2.05, subsample 0.60, column sample 0.92, L2 penalty 1.67. Chosen by random search (25 settings) for PR-AUC in 5-fold patient-grouped cross-validation (`reports/metrics/best_params.json`). No class weighting.
- **Reference model:** logistic regression on the same features. It is kept as a simpler fallback and is evaluated alongside throughout.
- **Version:** 1.0. Trained with the code at commit `f645c3c` on the full training set. `models/metadata.json` records the library versions (scikit-learn 1.9.1, XGBoost 3.4.1).
- **Owner:** Enock Mulenga
- **Date:** 2026-11-07

## Intended use
- **Primary use (illustrative):** rank adult inpatients with diabetes at discharge by their risk of unplanned readmission within 30 days, so that a care team with limited capacity can prioritise follow-up such as a pharmacist medicine review, a follow-up call or community nursing.
- **Users:** clinical and care-coordination staff, who make the final decision.
- **The score supports prioritisation only.** It must not be used to exclude anyone from care. At 20% capacity, about 61% of readmissions are still not flagged.
- **Out of scope:**
  - Any use on real patients without local validation, governance approval and a data protection impact assessment. This is a portfolio project trained on public US data from 1999–2008.
  - Decisions about insurance, payment, eligibility or discharge timing.
  - Patients outside the training population: children, admissions without a diabetes diagnosis, stays longer than 14 days.
  - Estimating *who would benefit* from an intervention. The model predicts risk, not treatment effect.

## Training and evaluation data
- **Source:** UCI Diabetes 130-US Hospitals for Years 1999–2008 (CC BY 4.0). These are de-identified inpatient encounters with any diabetes diagnosis and a 1–14 day stay. Provenance and licence are in [data/README.md](../data/README.md).
- **Cohort:** 99,340 encounters from 69,987 patients. Discharges to death or hospice and 3 records with invalid sex were excluded. 11.39% were readmitted within 30 days.
- **Split:** a patient-disjoint, outcome-stratified 80/20 hold-out, made before any exploration. The training set has 79,473 encounters and 56,036 patients. The test set has 19,867 encounters, 13,951 patients and 2,263 readmissions. All tuning used grouped cross-validation on the training set only.
- **Known biases:** US hospitals only; ICD-9 coding; a selection of encounters made by the dataset's original authors; readmissions to hospitals outside the network are probably missed; race recording quality is unknown. No dates are available, so no temporal validation was possible.

## Metrics
PR-AUC is the primary metric, because readmissions are the minority class and the task is to rank. Capacity metrics translate performance into the team's terms. The operating point is the top 20% of discharges by predicted risk. Test set, scored once, with 95% intervals from 1,000 patient-level bootstrap resamples:

| Metric | XGBoost | Logistic regression | No-skill baseline |
|---|---:|---:|---:|
| PR-AUC | 0.235 (0.216–0.257) | 0.211 (0.193–0.231) | 0.114 |
| ROC-AUC | 0.672 (0.659–0.685) | 0.660 (0.647–0.673) | 0.500 |
| Brier score | 0.0958 | 0.0970 | 0.101 |
| Readmissions reached, top 10% | 24.5% | 23.2% | 10% |
| Readmissions reached, top 20% | 38.5% | 37.8% | 20% |
| Precision, top 20% | 21.9% | 21.5% | 11.4% |

- **Paired difference (XGBoost minus logistic regression):** PR-AUC +0.023 (+0.014 to +0.033). Recall at 20% capacity +0.7 pts (−0.6 to +1.9), which is **not distinguishable**.
- **Calibration:** good without a recalibration step. Every decile's observed rate is within 1.4 pts of its predicted rate.
- **Cross-validation vs test:** PR-AUC 0.234 in CV and 0.235 on the test set, so there is no sign that tuning overfitted.

## Subgroup performance (fairness)
Policy: flag the top 20% with one cut-off for everyone. Recall is the share of each group's readmissions that are flagged. Race is not a model input. Full table: README section 11 and `reports/metrics/fairness_test.csv`.

| Group | Readmissions | Recall | Gap vs reference (95% CI) | Calibration gap |
|---|---:|---:|---:|---:|
| Female (ref.) / Male | 1,207 / 1,056 | 37.4% / 39.8% | +2.3 pts (−2.7 to +7.2) | ≤ 0.2 pts |
| Age 60–79 (ref.) | 1,110 | 39.1% | — | +0.1 pts |
| Age 80+ | 491 | 34.2% | −4.9 pts (−10.6 to +0.9) | −0.5 pts |
| Age <40 | 129 | 50.4% | +11.3 pts (−1.8 to +23.4) | 0.0 pts |
| Caucasian (ref.) / African American | 1,731 / 407 | 38.9% / 40.0% | +1.1 pts (−5.1 to +7.9) | ≤ 0.6 pts |
| Race unknown | 37 | 24.3% | −14.6 pts (−28.6 to +0.8) | +0.8 pts |

- Calibration holds in every group: a score means the same risk regardless of sex, age or race.
- **Main concern: patients aged 80+.** They have the lowest recall among the adequately sized groups and the weakest ranking (ROC-AUC 0.628). Logistic regression's gap is smaller (−2.4 pts).
- Patients with missing race are flagged 9.3% of the time, against 20% overall.
- The Asian, Hispanic and Other groups each have fewer than 50 readmissions. The test set cannot assess fairness for them.

## Explainability
- **Exact TreeSHAP:** values come from XGBoost's built-in method, with one-hot columns summed back to their original feature. The values plus the bias equal the model's raw output to within 10⁻⁵ for every patient.
- **Global drivers:** prior inpatient stays (19% of the total mean absolute SHAP), discharge destination (17%), all prior visits (7%), payer (5%) and primary diagnosis group (5%).
- **Prior stays:** compared with an average patient, five or more prior stays multiplies the odds about 3×, and none multiplies them by about 0.8.
- **Local explanations** (per-patient contributions) are available from `readmission_risk.models.explain.shap_by_feature` and could be shown to staff next to each score.
- **SHAP describes what the model uses, not causes.**

## Ethical considerations and UK GDPR
- **No real or personal data was used.** The data is public, de-identified and CC BY 4.0.
- **Health data is special-category data** under UK GDPR Article 9. Any real deployment would need a lawful basis and an Article 9 condition (for example, 9(2)(h), provision of health care), a data protection impact assessment, and local information-governance approval. This is a summary, not legal advice.
- **Human in the loop:** the design keeps a person in the loop. The score prioritises outreach and does not make decisions about individuals by itself, which is relevant to Article 22 on solely automated decision-making. Patients would still need to be told how their data is used.
- **Payer may be a proxy:** insurance type is the fourth most important feature and may stand in for income or social disadvantage. A sensitivity analysis without it is planned (see Monitoring and next steps).
- **Risk of reinforcing access patterns:** the model learns from past utilisation, so patients who historically used fewer services, possibly because of access barriers, may be under-flagged.

## Limitations and risks
- Trained on US hospital data from 1999–2008 with ICD-9 codes. Care pathways, coding (ICD-10 and SNOMED CT in the NHS) and populations differ, so performance in a UK setting is **unknown**.
- Modest discrimination (ROC-AUC 0.67). This is typical for readmission models built on administrative data, which lack social, functional and post-discharge information.
- A random patient split, not a temporal one: no dates are available, so drift over time is untested.
- Readmissions elsewhere are probably not recorded, so the labels likely undercount events.
- Groups with fewer than 50 readmissions cannot be audited reliably.
- High risk does not mean most preventable. Targeting interventions well needs evidence of benefit, such as an uplift model or a trial.

## Monitoring (if deployed)
Review monthly, with alerts:

| Signal | Why | Action if triggered |
|---|---|---|
| Observed vs predicted readmission rate, by decile | Calibration drift | Recalibrate on recent local data |
| Recall and precision at the agreed capacity | Usefulness to the care team | Investigate; retrain if sustained |
| Recall for age 80+ and for missing race, vs overall | Known fairness risks | Top-up rule or separate review; retrain |
| Distribution shift (PSI) in predicted risk, prior inpatient stays, discharge destination and payer mix | Input drift | Check data pipelines; retrain if real |
| Share of missing values per feature | Upstream data-quality failures | Fix the feed before trusting scores |

Retrain at least yearly or when a trigger persists for two consecutive months. Re-run the full fairness audit before every release.
