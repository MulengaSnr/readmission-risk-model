# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 · Exploratory analysis
#
# **Question for this notebook:** what in the data is associated with 30-day
# readmission, where are the traps, and what does that mean for feature design?
#
# **Ground rule:** this notebook uses the **training set only**. The patient-disjoint
# test set (20%) was split off before any exploration (`readmission_risk.data.split`)
# and is not opened until final evaluation, so nothing seen here can leak into the
# test result.
#
# Reusable logic lives in `src/readmission_risk/` and is unit-tested; this notebook
# is the narrative.

# %%
import pandas as pd
from IPython.display import display

from readmission_risk.analysis import (
    chi_square_independence,
    missingness,
    rate_table,
)
from readmission_risk.config import PROJECT_ROOT
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.visualisation.plots import (
    apply_style,
    plot_missingness,
    plot_rates,
    save,
)

apply_style()
pd.set_option("display.precision", 3)
FIGURES = PROJECT_ROOT / "reports" / "figures"

train = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "train.parquet")
overall = train[TARGET].mean()
print(f"{len(train):,} encounters · {train[GROUP_COL].nunique():,} patients")
print(f"Readmitted within 30 days: {overall:.2%}")

# %% [markdown]
# ## 1. Outcome balance
#
# About **11.4%** of encounters are readmitted within 30 days. With roughly 1 positive
# in 9, accuracy is meaningless (predicting "no" for everyone scores ~89%). The project
# therefore uses **PR-AUC** as its primary metric, plus calibration.

# %% [markdown]
# ## 2. Missing data

# %%
miss = missingness(train)
display(miss.to_frame("share_missing"))
ax = plot_missingness(miss, "Two fields are missing for 40–50% of encounters")
save(ax.figure, FIGURES / "eda_missingness.png")
# %%
for col in ["medical_specialty", "payer_code", "race"]:
    t = rate_table(train, col)
    print(
        f"{col:18s} readmission when missing: {t.loc['Missing', 'rate']:.1%}  "
        f"(overall {overall:.1%})"
    )

# %% [markdown]
# **Decision:** `medical_specialty` and `payer_code` are missing so often that
# imputing a "most likely" value would invent data. Missingness is kept as its own
# category ("Missing"). Missing `race` has a *lower* readmission rate, which hints
# that recording practices differ between hospitals. That is a reminder that the
# data reflects processes as well as patients.

# %% [markdown]
# ## 3. Prior utilisation: the strongest signal

# %%
train["prior_inpatient"] = (
    train["number_inpatient"].clip(upper=5).astype(str).replace("5", "5+")
)
t = rate_table(train, "prior_inpatient")
display(t)
display(chi_square_independence(train, "prior_inpatient"))
ax = plot_rates(
    t,
    "Readmission risk rises steeply with prior inpatient stays",
    overall,
    xlabel="Inpatient admissions in the year before this encounter",
)
save(ax.figure, FIGURES / "eda_prior_inpatient.png")
# %% [markdown]
# Patients with no prior admissions are readmitted about **8.6%** of the time; with
# five or more, about **37%**. This is the largest association in the dataset
# (Cramér's V ≈ 0.17). `number_inpatient` counts visits *before* the encounter, so it
# is legitimately available at discharge.

# %% [markdown]
# ## 4. A leakage trap: "how many times is this patient in the data?"

# %%
train["encounters_in_dataset"] = (
    train.groupby(GROUP_COL)["encounter_id"]
    .transform("size")
    .clip(upper=5)
    .astype(str)
    .replace("5", "5+")
)
t = rate_table(train, "encounters_in_dataset")
display(t)
ax = plot_rates(
    t,
    "Leakage trap: counting a patient's encounters 'predicts' readmission",
    overall,
    xlabel="Encounters for this patient anywhere in the dataset (uses future data)",
)
save(ax.figure, FIGURES / "eda_leakage_trap.png")
# %% [markdown]
# Patients who appear once have a ~4% readmission rate; those who appear five or
# more times, ~28%. This looks like a superb feature and is **pure leakage**: a
# readmission *is* another encounter in the dataset, so the count partly contains the
# answer, and it uses visits that happen after discharge.
#
# **Decisions:**
# - Never build features from a patient's other rows in the dataset. Use only the
#   pre-encounter counts the data provides (`number_inpatient`, `number_emergency`,
#   `number_outpatient`).
# - This is also why every split is grouped by patient. With a random split, the
#   model would learn patient identity from repeat visits.

# %% [markdown]
# ## 5. Where patients go after discharge

# %%
# Short display names for the official (long) disposition labels.
SHORT = {
    "Discharged to home": "Home",
    "Discharged/transferred to home with home health service": "Home with home health",
    "Discharged/transferred to SNF": "Skilled nursing facility",
    "Discharged/transferred to ICF": "Intermediate care facility",
    "Discharged/transferred to another short term hospital": "Another short-term hospital",
    "Discharged/transferred to another rehab fac including rehab units of a hospital .": "Rehab facility",
    "Discharged/transferred to another type of inpatient care institution": "Other inpatient care",
    "Discharged/transferred to a long term care hospital.": "Long-term care hospital",
    "Left AMA": "Left against medical advice",
}
t = rate_table(train, "discharge_disposition", min_count=300)
t.index = [SHORT.get(s, s) for s in t.index]
t = t.sort_values("rate", ascending=False)
display(t)
display(chi_square_independence(train, "discharge_disposition"))
ax = plot_rates(
    t,
    "Discharge to rehab or other inpatient care carries the highest risk",
    overall,
    horizontal=True,
)
save(ax.figure, FIGURES / "eda_discharge.png")
# %% [markdown]
# Patients discharged home have the lowest risk. Those transferred to rehab, other
# inpatient care or a skilled nursing facility have markedly higher risk. These are
# sicker patients, so discharge destination works as a proxy for severity. Because
# it's known at discharge, it is a valid input.

# %% [markdown]
# ## 6. Age

# %%
t = rate_table(train, "age")
t.index = [
    f"{band[1:-1].split('-')[0]}–{int(band[1:-1].split('-')[1]) - 1}"
    for band in t.index
]
display(t)
ax = plot_rates(
    t,
    "Risk is lowest under 20; the 20–29 band is high but small",
    overall,
    xlabel="Age band (years)",
)
save(ax.figure, FIGURES / "eda_age.png")
# %% [markdown]
# The under-20 bands are small (wide intervals). The 20–29 band has the highest rate
# (about 15.5%), but it rests on only about 1,300 encounters, so treat it with care.
# Among the large bands, risk is lowest at 50–59 and highest at 70–89. Age is kept as an **ordinal** feature (band
# order preserved) instead of 10 unrelated categories.

# %% [markdown]
# ## 7. HbA1c testing: the original study's question

# %%
t = rate_table(train, "A1Cresult").loc[["Norm", ">7", ">8", "Not measured"]]
display(t)
display(chi_square_independence(train, "A1Cresult"))
ax = plot_rates(
    t,
    "Patients with an HbA1c test are readmitted slightly less often",
    overall,
    xlabel="HbA1c result during the encounter",
)
save(ax.figure, FIGURES / "eda_a1c.png")
# %% [markdown]
# Encounters with any HbA1c result are readmitted at about 10%, against about 11.7%
# when it was not measured. The difference is statistically significant but
# **small** (Cramér's V ≈ 0.02). With 80,000 rows, almost any difference reaches
# significance, so effect sizes are reported alongside p-values. This is
# association, not causation: patients who get tested may also differ in other
# ways.

# %% [markdown]
# ## 8. Sex and race: fairness preview

# %%
for col in ["gender", "race"]:
    display(rate_table(train, col))
    display(chi_square_independence(train, col))

# %% [markdown]
# Readmission rates barely differ by sex (p ≈ 0.47) and differ only slightly by
# race. That doesn't make a model fair. Error rates and calibration can still vary
# across groups, so the fairness audit in step 6 checks model performance for each
# group, not just base rates. Whether `race` should be a model input at all is
# tested explicitly in modelling.

# %% [markdown]
# ## 9. Diagnoses need grouping

# %%
for col in ["diag_1", "diag_2", "diag_3"]:
    print(f"{col}: {train[col].nunique()} distinct ICD-9 codes")
display(train["diag_1"].value_counts().head(10).to_frame("encounters"))

# %% [markdown]
# Primary diagnosis alone has about 700 distinct ICD-9 codes. Most are rare, which
# makes them noisy as raw categories. **Step 4 (your exercise)** groups codes into
# clinically meaningful chapters (circulatory, respiratory, diabetes, …).

# %% [markdown]
# ## 10. Summary: what this means for modelling
#
# | Finding | Decision |
# |---|---|
# | 11.4% positive rate | PR-AUC primary, calibration checked, threshold set by capacity |
# | `medical_specialty` / `payer_code` 40–50% missing | Keep "Missing" as a category; no invented values |
# | Prior inpatient stays: strongest signal (8.6% → 37%) | Keep pre-encounter utilisation counts, capped for robustness |
# | Patient encounter count is a leak | Never derive features from a patient's other rows; split by patient |
# | Discharge destination informative | Keep as categorical, rare levels grouped |
# | Age: ordered bands | Ordinal encoding |
# | HbA1c effect small | Keep, but expect little lift from it |
# | ~700 diagnosis codes | Group into ICD-9 chapters (step 4) |
# | All single-variable associations are weak (max V ≈ 0.17) | Expect modest discrimination, and report it honestly |
