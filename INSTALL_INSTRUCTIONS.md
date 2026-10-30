# Apply Phase 3 GitHub update

1. Unzip this archive.
2. Copy `README.md`, `docs/`, `reports/`, and `scripts/` into the existing `customer-lifecycle-analytics` repository, merging folders and replacing the root README. Do not delete existing Phase 2 files.
3. The repo should already contain the Phase 3 scripts `build_churn_target.py`, `prepare_temporal_splits.py`, `train_churn_models.py`, and `evaluate_churn_test.py`. If missing, copy the previously validated versions into `scripts/` before claiming full reproducibility.
4. Merge the package requirements into the existing `requirements.txt`; do not replace existing dependencies. Consider pinning compatible scikit-learn versions for model-artifact portability.
5. Inspect GitHub Desktop Changes, check that raw Excel, `.duckdb`, `.parquet`, `.joblib`, `.venv` and `output/` are not tracked, then commit: `Document Phase 3 churn modeling and holdout results`.
6. Publish/push changes. Dashboard remains future work.

The package contains documentation and static figures from reported metrics; it is not a new training run. The final holdout numbers are sourced from the user's completed terminal output. The coefficient export was read from the user-provided fitted model, with a scikit-learn version warning; re-export in the training environment for strict reproducibility.
