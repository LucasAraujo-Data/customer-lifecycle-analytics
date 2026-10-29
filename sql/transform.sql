CREATE OR REPLACE TABLE quality_events AS
SELECT source_sheet,source_row_number,
 CASE WHEN customer_id IS NULL OR customer_id = '' THEN 'missing_customer_id'
      WHEN invoice IS NULL OR invoice = '' THEN 'missing_invoice'
      WHEN invoice ILIKE 'C%' THEN 'cancellation_invoice'
      WHEN invoice_timestamp IS NULL THEN 'missing_timestamp'
      WHEN quantity IS NULL OR quantity <= 0 THEN 'nonpositive_quantity'
      WHEN price IS NULL OR price <= 0 THEN 'nonpositive_price'
      ELSE 'valid' END AS rejection_reason,
 CASE WHEN rejection_reason = 'valid' THEN 'accepted' ELSE 'rejected' END AS quality_status,
 'v1' AS validation_version
FROM raw_lines;

CREATE OR REPLACE TABLE valid_lines AS
SELECT r.* FROM raw_lines r JOIN quality_events q
USING (source_sheet,source_row_number) WHERE q.quality_status='accepted';

CREATE OR REPLACE TABLE orders_clean AS
SELECT customer_id,invoice AS invoice_id,MIN(invoice_timestamp) AS order_timestamp,
 SUM(quantity*price) AS gross_purchase_value,COUNT(*) AS line_count,
 STRING_AGG(DISTINCT source_sheet,', ' ORDER BY source_sheet) AS source_sheet
FROM valid_lines GROUP BY customer_id,invoice;

CREATE OR REPLACE TABLE quality_metrics AS
SELECT 'raw_rows' AS metric,COUNT(*)::BIGINT AS value FROM raw_lines
UNION ALL SELECT 'valid_lines',COUNT(*) FROM valid_lines
UNION ALL SELECT 'orders',COUNT(*) FROM orders_clean
UNION ALL SELECT 'customers',COUNT(DISTINCT customer_id) FROM orders_clean;

CREATE OR REPLACE TABLE snapshot_dates AS
SELECT CAST(month_start AS TIMESTAMP) AS as_of_date
FROM generate_series(DATE '2010-06-01',DATE '2011-12-01',INTERVAL '1 month') AS t(month_start);

CREATE OR REPLACE TABLE customer_snapshot AS
SELECT o.customer_id,s.as_of_date,MIN(o.order_timestamp) AS first_purchase_at,
 MAX(o.order_timestamp) AS last_purchase_at,COUNT(*) AS historical_order_count,
 TRUE AS eligible_flag
FROM orders_clean o JOIN snapshot_dates s ON o.order_timestamp < s.as_of_date
GROUP BY o.customer_id,s.as_of_date;

CREATE OR REPLACE TABLE customer_features AS
WITH historical AS (
 SELECT s.customer_id,s.as_of_date,s.first_purchase_at,s.last_purchase_at,s.historical_order_count,
  o.order_timestamp,o.gross_purchase_value,
  ROW_NUMBER() OVER (PARTITION BY s.customer_id,s.as_of_date ORDER BY o.order_timestamp,o.invoice_id) AS purchase_rank,
  LAG(o.order_timestamp) OVER (PARTITION BY s.customer_id,s.as_of_date ORDER BY o.order_timestamp,o.invoice_id) AS prev_order_at
 FROM customer_snapshot s JOIN orders_clean o
 ON o.customer_id=s.customer_id AND o.order_timestamp<s.as_of_date
)
SELECT customer_id,as_of_date,'v1' AS feature_version,
 DATE_DIFF('day',MAX(last_purchase_at),as_of_date) AS recency_days,
 DATE_DIFF('day',MIN(first_purchase_at),as_of_date) AS tenure_days,
 MAX(historical_order_count) AS historical_order_count,
 (MAX(historical_order_count)>=2) AS prior_repeat_flag,
 COUNT(*) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '180 days') AS frequency_180d,
 COALESCE(SUM(gross_purchase_value) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '180 days'),0) AS monetary_180d,
 COUNT(*) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '30 days') AS order_count_30d,
 COUNT(*) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '60 days') AS order_count_60d,
 COUNT(*) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '90 days') AS order_count_90d,
 COALESCE(SUM(gross_purchase_value) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '30 days'),0) AS monetary_30d,
 COALESCE(SUM(gross_purchase_value) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '60 days'),0) AS monetary_60d,
 COALESCE(SUM(gross_purchase_value) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '90 days'),0) AS monetary_90d,
 AVG(gross_purchase_value) FILTER (WHERE order_timestamp>=as_of_date-INTERVAL '180 days') AS avg_order_value_180d,
 AVG(DATE_DIFF('second',prev_order_at,order_timestamp)/86400.0) FILTER (WHERE prev_order_at IS NOT NULL) AS avg_interpurchase_days,
 MAX(order_timestamp) AS max_feature_event_at
FROM historical GROUP BY customer_id,as_of_date;
