# Phase 3 — Model Evaluation

## Target and population

`churn_90d = 1` if the customer has no valid invoice in the next 90 days. Eligible observations are active in the previous 180 days. The source last order is 2011-12-09; only snapshots with a full forward window were labeled. The date coverage is inferred from transaction records, not independently certified completeness.

## Train / validation / holdout

| Dataset | Snapshot period | Rows | Positive class | Rate |
|---|---|---:|---:|---:|
| Train | 2010-06 to 2011-01 | 24,092 | 12,133 | 50.36% |
| Validation | 2011-05 to 2011-06 | 5,708 | 2,916 | 51.09% |
| Holdout | 2011-09 | 2,772 | 1,089 | 39.29% |

Train label windows end before validation starts; validation label windows end before the final test starts. Repeated customers across months remain possible. The holdout contains only one snapshot.

## Model selection protocol

Baseline: class-prior DummyClassifier. Candidate models: Logistic Regression (median imputation, standardization) and Random Forest (median imputation). Preprocessing is fit on train only. Select model by validation average precision (PR-AUC), then select threshold by validation F1. **Do not reselect after test**.

| Validation | Baseline | Logistic Regression | Random Forest |
|---|---:|---:|---:|
| PR-AUC | 0.5109 | **0.7541** | 0.7315 |
| ROC-AUC | 0.5000 | **0.7666** | 0.7649 |
| Brier score (lower better) | 0.2499 | 0.2002 | **0.1960** |
| Best F1 | 0.6763 | 0.7366 | **0.7443** |

Logistic Regression was selected because PR-AUC was the pre-specified selection metric, not because it won every metric. The selected threshold is 0.30.

## Final holdout (September 2011)

| Metric | Validation | Final holdout |
|---|---:|---:|
| PR-AUC | 0.7541 | **0.6226** |
| ROC-AUC | 0.7666 | **0.7407** |
| Precision | 0.6269 | **0.5111** |
| Recall | 0.8930 | **0.8430** |
| F1 | 0.7366 | **0.6364** |
| Brier score | 0.2002 | **0.2003** |

Final confusion matrix: TN=805, FP=878, FN=171, TP=918. The model flagged 1,796 customers and correctly identified 918 of the 1,089 no-repeat outcomes. High recall comes with many false positives.

![Validation and holdout](../reports/figures/validation_vs_test.png)
![Holdout confusion matrix](../reports/figures/test_confusion_matrix.png)

## Caveats

- A single-month holdout does not establish seasonal robustness; rolling-origin checks remain future work.
- PR-AUC depends on positive-class prevalence, which fell from 51.09% in validation to 39.29% in holdout.
- Customers recur across monthly snapshots, so observations are dependent.
- This model predicts an observed future purchase outcome, not the effect of a retention intervention.
- No model retuning, threshold adjustment or retraining was performed after viewing holdout metrics.
- The dataset's last transaction timestamp is used as a proxy for coverage end; independent completeness checks would strengthen this assumption.
