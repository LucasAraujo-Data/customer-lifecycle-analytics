"""Build the September 2011 Customer Insights dataset for Power BI.

Run from the repository root: python scripts/build_customer_insights.py
Requires: duckdb
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'output'
POWERBI = OUTPUT / 'powerbi'
SNAPSHOT = '2011-09-01'


def main():
    predictions = POWERBI / 'churn_test_predictions.csv'
    rfm = OUTPUT / 'rfm_segments.parquet'
    features = OUTPUT / 'customer_features.parquet'
    for path in (predictions, rfm, features):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')

    POWERBI.mkdir(parents=True, exist_ok=True)
    destination = POWERBI / 'customer_insights.csv'
    con = duckdb.connect()
    try:
        # Explicitly normalize ID and date types before joining.
        query = """
        WITH predictions AS (
            SELECT CAST(customer_id AS VARCHAR) AS customer_id,
                   CAST(as_of_date AS DATE) AS as_of_date,
                   TRY_CAST(churn_probability AS DOUBLE) AS churn_probability,
                   TRY_CAST(actual_churn AS INTEGER) AS actual_churn,
                   TRY_CAST(predicted_churn AS INTEGER) AS predicted_churn,
                   CAST(risk_band AS VARCHAR) AS risk_band,
                   TRY_CAST(risk_band_sort AS INTEGER) AS risk_band_sort
            FROM read_csv_auto(?, ALL_VARCHAR=TRUE, HEADER=TRUE)
        ),
        rfm AS (
            SELECT CAST(customer_id AS VARCHAR) AS customer_id,
                   CAST(as_of_date AS DATE) AS as_of_date,
                   rfm_segment, lifecycle_status
            FROM read_parquet(?)
        ),
        features AS (
            SELECT CAST(customer_id AS VARCHAR) AS customer_id,
                   CAST(as_of_date AS DATE) AS as_of_date,
                   monetary_180d, frequency_180d, recency_days,
                   avg_order_value_180d, historical_order_count
            FROM read_parquet(?)
        )
        SELECT p.customer_id, p.as_of_date, r.rfm_segment, r.lifecycle_status,
               f.monetary_180d, f.frequency_180d, f.recency_days,
               f.avg_order_value_180d, f.historical_order_count,
               p.churn_probability, p.risk_band, p.risk_band_sort,
               p.predicted_churn, p.actual_churn
        FROM predictions p
        LEFT JOIN rfm r USING (customer_id, as_of_date)
        LEFT JOIN features f USING (customer_id, as_of_date)
        WHERE p.as_of_date = DATE '2011-09-01'
        """
        params = [str(predictions), str(rfm), str(features)]
        con.execute('CREATE TEMP TABLE insights AS ' + query, params)
        stats = con.execute("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT customer_id) AS customers,
                   COUNT(DISTINCT as_of_date) AS snapshots,
                   COUNT(*) - COUNT(DISTINCT (customer_id, as_of_date)) AS duplicate_pairs,
                   COUNT(*) FILTER (WHERE rfm_segment IS NULL) AS missing_rfm,
                   COUNT(*) FILTER (WHERE monetary_180d IS NULL) AS missing_features,
                   COUNT(*) FILTER (WHERE churn_probability IS NULL OR actual_churn IS NULL OR predicted_churn IS NULL OR risk_band IS NULL) AS missing_predictions
            FROM insights
        """).fetchone()
        labels = ['rows','customers','snapshots','duplicate_pairs','missing_rfm','missing_features','missing_predictions']
        print('Validation:', dict(zip(labels, stats)))
        if stats != (2772, 2772, 1, 0, 0, 0, 0):
            raise ValueError('Validation failed; CSV was not exported. Check snapshot, joins, and inputs.')
        con.execute(f"COPY insights TO '{destination.as_posix()}' (FORMAT CSV, HEADER TRUE)")
        print(f'Exported {stats[0]:,} rows to {destination}')
        print('Revenue at risk should be described as historical 180-day spending associated with high-risk customers, not forecast lost revenue.')
    finally:
        con.close()


if __name__ == '__main__':
    main()
