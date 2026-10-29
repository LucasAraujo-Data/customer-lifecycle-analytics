-- Fail the build if a data-contract invariant is violated.
SELECT CASE WHEN COUNT(*)=1067371 THEN TRUE ELSE error('Raw row count mismatch') END FROM raw_lines;
SELECT CASE WHEN COUNT(*)=0 THEN TRUE ELSE error('Duplicate raw line key') END FROM
 (SELECT source_sheet,source_row_number FROM raw_lines GROUP BY 1,2 HAVING COUNT(*)>1);
SELECT CASE WHEN COUNT(*)=0 THEN TRUE ELSE error('Invalid order record') END FROM orders_clean
 WHERE customer_id IS NULL OR order_timestamp IS NULL OR gross_purchase_value<=0 OR invoice_id ILIKE 'C%';
SELECT CASE WHEN COUNT(*)=0 THEN TRUE ELSE error('Duplicate customer invoice') END FROM
 (SELECT customer_id,invoice_id FROM orders_clean GROUP BY 1,2 HAVING COUNT(*)>1);
SELECT CASE WHEN COUNT(*)=0 THEN TRUE ELSE error('Feature leakage') END FROM customer_features
 WHERE max_feature_event_at>=as_of_date;
SELECT CASE WHEN COUNT(*)=0 THEN TRUE ELSE error('Incorrect prior-repeat flag') END FROM customer_features
 WHERE prior_repeat_flag <> (historical_order_count>=2);
SELECT CASE WHEN COUNT(*)=0 THEN TRUE ELSE error('Duplicate snapshot features') END FROM
 (SELECT customer_id,as_of_date FROM customer_features GROUP BY 1,2 HAVING COUNT(*)>1);
