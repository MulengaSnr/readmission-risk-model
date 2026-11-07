# Findings

_Written for a non-technical stakeholder: what we found, how confident we are, and what to do next._

**The question:** a care team can follow up only a limited number of patients after discharge. Can we tell which patients with diabetes are most likely to be back in hospital within 30 days, so the team calls the right people first?

**The short answer:** yes, roughly twice as well as picking at random, but far from perfectly. The score is useful for deciding who to call first. It is not reliable enough to decide who does *not* need help.

## Key findings
1. **Following up the 20% of patients with the highest scores would reach about 39% of all readmissions.** Calling 20% of patients at random reaches 20%. Put another way, about 1 in 5 patients on the high-risk list are readmitted, against about 1 in 9 overall. In the test data, the list contained 872 of 2,263 readmissions.
2. **The scores mean what they say.** When the model says "25% risk", about 25% of those patients are readmitted. This holds for men and women, every age group and every ethnic group we could measure, so staff can read the numbers at face value.
3. **The two strongest warning signs are recent hospital stays and where the patient goes after discharge.** Each inpatient stay in the previous year raises the risk. Five or more stays roughly triples the odds compared with an average patient. Patients discharged to other care facilities are at higher risk than those going home.
4. **The model works about equally well for men and women, and for Black and white patients.** It catches a similar share of each group's readmissions.
5. **It is weaker for the oldest patients.** For patients aged 80 and over, the list catches about 34% of their readmissions against 39% for those aged 60–79. The gap is not certain (it could be chance), but it is the main fairness risk to watch.
6. **A simpler model is almost as good.** A basic statistical model (logistic regression) catches 38% of readmissions at the same 20% capacity, against 39%, and that difference could be chance. The more complex model (XGBoost) is better overall and is our recommendation, but the simpler one is a reasonable fallback.

**How confident are we?** The results come from data the model never saw during development, which was set aside at the start and used once. The ranges in the main report show the uncertainty. The headline figure is between about 36% and 40% of readmissions reached.

## Recommendations
1. **Use the score to put patients in order for follow-up, not to rule anyone out.** About 6 in 10 readmissions are not on the top-20% list, so clinical judgement and referrals must continue alongside it.
2. **Watch the oldest patients.** Track how many of their readmissions the list catches. If the gap persists, add a separate check for patients aged 80+.
3. **Check whether the insurance-type field is needed.** It is the fourth most important input and may reflect income rather than health. Test the model without it and drop it if accuracy holds.
4. **Validate locally before any real use.** These results come from US hospitals in 1999–2008. Any UK use would need testing on local data, information-governance approval and a data protection impact assessment.

## Caveats and limitations
- **Old, foreign data:** the data comes from US hospitals between 1999 and 2008. How well it would work in the NHS today is unknown.
- **Missing information:** there is nothing about home circumstances, support at home or what happened after discharge, which limits how accurate any model built on this data can be.
- **Probable undercounting:** readmissions to other hospitals are probably not recorded, so true readmission rates are likely higher.
- **Small groups:** some ethnic groups are too small in this data to judge whether the model treats them fairly.
- **Risk is not benefit:** a high score shows who is most likely to come back, not who would benefit most from a call. Those can be different people.
