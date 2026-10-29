# Customer Lifecycle Analytics — RFM, K-Means and Churn (Work in Progress)

Portfolio project using **UCI Online Retail II** to study repeat purchasing, customer lifecycle and retention. Current scope: validated historical features, RFM segmentation and exploratory K-Means. **Supervised churn modeling and Power BI dashboard are planned, not completed.**

## Reproduce Phase 2 (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/build.py --source "C:\path\to\online_retail_II.xlsx" --output output
python tests/validate_customer_features_fixed.py --file output/customer_features.parquet
python tests/validate_project4_orders_features.py --orders output/orders_clean.parquet --features output/customer_features.parquet
python scripts/build_rfm.py --features output/customer_features.parquet --output output
python scripts/evaluate_kmeans_reviewed.py --rfm output/rfm_segments.parquet --snapshot 2011-09-01 --output output
python scripts/evaluate_kmeans_temporal.py --rfm output/rfm_segments.parquet --output output/kmeans_temporal
```

Download the original data separately; see [data instructions](data/README.md). All generated data stay in `output/` and are gitignored. Some validation scripts are retained with their original filenames for compatibility.

## Validated findings
- 36,969 valid invoices and 5,878 customers in the cleaned orders.
- 84,070 customer-month features; eight reconciliation checks passed against cleaned orders.
- RFM: 57,648 active and 26,422 inactive customer-month observations.
- Exploratory K-Means: k=2 led silhouette among k=2..8 at three snapshots (0.4438, 0.4661, 0.4377).

See [methodology](docs/methodology.md), [Phase 3 plan](docs/phase3_plan.md), and [summary results](reports/).

## Important caveats
Exploratory results are not evidence of a trained churn predictor. Temporal validation of a fixed clustering model was not performed. The source dataset is not included.
