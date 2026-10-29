# Phase 3 — Churn prediction: implementation plan (not yet implemented)

## Objective
Predict whether an eligible customer will make **no valid purchase in the next 90 days** after a monthly as-of date. This is an operational no-repeat-purchase outcome, not permanent customer loss.

## Eligibility and labels
- As-of dates must have a complete 90-day forward observation window before the last observed transaction. With data ending 2011-12-09, **2011-09-01 is the latest monthly snapshot with a complete 90-day label**; October–December 2011 snapshots must be excluded from labeled evaluation.
- Compare two candidate populations before choosing: all customers with prior purchases vs customers with a purchase in the trailing 180 days. Prefer the active 180-day cohort for an actionable retention model; report excluded inactive counts.
- `churn_90d=1` if zero valid invoices occur in `[as_of_date, as_of_date + 90 days)`; else `0`. Document timestamp boundaries exactly.
- Use only as-of historical features; `max_feature_event_at < as_of_date` for every observation.
- Exclude identifier, label, future events and any future-derived segment from model inputs. RFM scores are allowed only if generated as of the snapshot.

## Evaluation design
- Temporal train/validation/test partition by snapshot month, not random row splits. Proposed starting split: train 2010-06 through 2011-03; validation 2011-04 through 2011-06; test 2011-07 through 2011-09, subject to confirming 90-day label completeness and temporal coverage.
- Because a customer appears in multiple snapshots, evaluate customer overlap and overlapping 90-day outcome windows; use a 90-day embargo or rolling-origin sensitivity test to assess leakage from overlapping outcomes near split boundaries.
- Fit scalers, imputers, encoders and thresholds on training data only. Freeze them for validation and test.
- Baselines: majority class, simple recency rule, regularized logistic regression; then a tree ensemble if justified.
- Metrics: PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, calibration and lift at a business-relevant top-N. Report prevalence per period and compare with baseline.
- Assess by prior-repeat flag, cohort age, and snapshot period; analyze drift and class imbalance.

## Gate to training
1. Confirm all labels by recomputation from `orders_clean`.
2. Investigate 28 customers missing from snapshot table.
3. Confirm eligibility and leakage checks, including no post-as-of feature event.
4. Freeze temporal split and outcome boundary definition.
5. Only then implement supervised training.
