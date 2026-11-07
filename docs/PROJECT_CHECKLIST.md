# Project checklist

Copy into each new repo and tick off as work progresses.

_Status for this repo, 2026-11-07: open items are GitHub Issues (work was tracked through one PR per step instead), repo topics, and linking from the profile, which waits until the repo is public._

## Plan
- [x] Business problem and decision-maker written down
- [x] Technical objectives and success metric (with baseline) defined
- [x] Dataset chosen; licence checked and recorded in `data/README.md`
- [x] Architecture sketched in README
- [ ] Issues created for each major task

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
- [ ] Repo description + topics set
- [ ] Linked from profile README and landing page
