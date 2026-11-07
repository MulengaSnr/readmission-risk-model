# Project checklist

Copy into each new repo and tick off as work progresses.

_Status for this repo, 2026-11-07: all items complete. Work was tracked as one PR per step (#1–#12). Follow-on work is in Issues #13–#15. The repo is public, with topics, and is linked from the profile README and landing page._

## Plan
- [x] Business problem and decision-maker written down
- [x] Technical objectives and success metric (with baseline) defined
- [x] Dataset chosen; licence checked and recorded in `data/README.md`
- [x] Architecture sketched in README
- [x] Issues created for each major task

## Build
- [x] Reproducible data download / generation script
- [x] Data validation checks
- [x] EDA notebook with key findings
- [x] Train / validation / test split decided **before** modelling; leakage reviewed
- [x] Baseline model
- [x] Candidate models + cross-validation
- [x] Final evaluation on test set (once)
- [x] Explainability + fairness analysis
- [x] Unit tests for `src/` logic; CI green

## Document
- [x] README complete (two-minute test passes)
- [x] Model card (if a model)
- [x] `reports/findings.md` for a non-technical reader
- [x] Screenshot / headline figure
- [x] Limitations and next steps

## Publish (only after approval)
- [x] No secrets (`git log -p | grep -i key` reviewed), no real personal data
- [x] Repo description + topics set
- [x] Linked from profile README and landing page
