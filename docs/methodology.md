# Methodology and validated results

## Source and grain
UCI Online Retail II, two worksheets spanning December 2009 to December 2011. The pipeline creates one row per valid invoice in `orders_clean` and one row per customer and monthly as-of date in `customer_features`.

## Quality and feature checks
Observed source: 1,067,371 transaction lines; 805,549 valid lines; 36,969 valid orders; 5,878 distinct identified customers. Feature output: 84,070 customer-snapshot rows, 19 snapshots, 5,850 distinct customers. Reconciliation against clean orders returned zero mismatches across eight implemented metrics and zero feature events at or after as-of date. This does not independently validate the original workbook.

## RFM segmentation
Active = at least one valid purchase in trailing 180 days; inactive otherwise. Active customers receive snapshot-local R/F/M quintiles (NTILE with customer ID tie-breaking). 57,648 active and 26,422 inactive customer-snapshot observations. Segment names are business labels, not permanent churn outcomes. Snapshot-relative quintiles may change with the population.

## Exploratory clustering
RobustScaler + K-Means on active-customer RFM features. k=2..8 assessed using silhouette, inertia, cluster size and seed ARI. k=2 had highest silhouette on 2010-09-01 (0.4438), 2011-03-01 (0.4661), and 2011-09-01 (0.4377). Independently fitted clusters are not directly label-aligned across time. RFM remains the primary business segmentation.

## Limitations
The 28 customers absent from snapshots require an explicit first-purchase date check. A fixed-model out-of-time clustering test and customer migration analysis were not performed. Source changes between 2010 and 2011 remain a limitation. No supervised churn model has been trained.
