# Data

Raw and processed data are **not committed** (see `.gitignore`). Recreate them with:

```bash
uv run python -m readmission_risk.data.download
```

## Provenance and licence

| Field | Value |
|---|---|
| Dataset | Diabetes 130-US Hospitals for Years 1999–2008 |
| Publisher | UCI Machine Learning Repository, dataset 296 |
| Source | <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008> |
| Licence | [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Retrieved | 2026-11-07 |
| Archive SHA-256 | `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a` (pinned in `configs/default.yaml`) |
| Contents | `diabetic_data.csv` (101,766 encounters × 50 columns), `IDS_mapping.csv` (lookup tables for admission type, discharge disposition, admission source) |

### Citation
> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014, 781670.

Dataset citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999–2008 [Dataset]. UCI Machine Learning Repository.

## What the data is
Inpatient encounters from 130 US hospitals and integrated delivery networks, 1999–2008, selected by the original authors where:
- the encounter was a hospital admission with any diabetes diagnosis,
- length of stay was 1–14 days,
- laboratory tests were performed and medications administered.

Each row is one **encounter**. `patient_nbr` identifies the patient, and one patient can have several encounters. That is why train/test splits in this project are **grouped by patient**.

Known data issues (to be handled in cleaning):
- Missing values are coded as `?`. `weight`, `payer_code` and `medical_specialty` are largely missing.
- `IDS_mapping.csv` contains three lookup tables stacked in one file, separated by blank rows.
- Some discharge dispositions (expired, hospice) mean readmission is impossible, so those encounters are excluded.

## Data protection
- The data is public, de-identified and published for research under CC BY 4.0. It contains **no names, dates or addresses**. Patient and encounter IDs are surrogate keys.
- It still includes sensitive attributes (age band, sex, race) and health data, which would be **special-category data** under UK GDPR Article 9 if it related to identifiable people. The project treats these attributes carefully: they are used for fairness auditing, and whether race should be a model input is evaluated explicitly.
- No re-identification is attempted, and no real patient, employer or client data is used anywhere in this project.

## Folders
- `raw/`: downloaded archive and extracted CSVs, never edited by hand
- `processed/`: outputs of `src/` pipelines, always reproducible from `raw/`
