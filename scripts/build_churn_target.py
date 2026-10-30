#!/usr/bin/env python3
"""Phase 3: build and audit 90-day no-repeat-purchase labels. No ML training."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', default='output/orders_clean.parquet')
    parser.add_argument('--features', default='output/customer_features.parquet')
    parser.add_argument('--output', default='output/phase3_target')
    parser.add_argument('--horizon-days', type=int, default=90)
    args = parser.parse_args()
    if args.horizon_days != 90:
        parser.error('Phase 3 specification is frozen at a 90-day horizon.')
    orders = Path(args.orders).resolve()
    features = Path(args.features).resolve()
    out = Path(args.output).resolve()
    for path in (orders, features):
        if not path.is_file():
            parser.error(f'Input not found: {path}')
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f'CREATE TEMP VIEW orders AS SELECT * FROM read_parquet({sql_literal(orders)})')
        con.execute(f'CREATE TEMP VIEW features AS SELECT * FROM read_parquet({sql_literal(features)})')
        required_orders = {'customer_id', 'invoice_id', 'order_timestamp'}
        required_features = {'customer_id', 'as_of_date', 'frequency_180d', 'max_feature_event_at'}
        for view, required in [('orders', required_orders), ('features', required_features)]:
            actual = {r[1] for r in con.execute(f"PRAGMA table_info('{view}')").fetchall()}
            if required - actual:
                raise ValueError(f'{view} missing columns: {sorted(required - actual)}')
        if con.execute('SELECT COUNT(*) FROM orders WHERE customer_id IS NULL OR order_timestamp IS NULL').fetchone()[0]:
            raise ValueError('Orders have null customer IDs or timestamps.')
        if con.execute('SELECT COUNT(*) FROM features WHERE customer_id IS NULL OR as_of_date IS NULL').fetchone()[0]:
            raise ValueError('Features have null customer IDs or snapshot dates.')
        duplicates = con.execute('SELECT COUNT(*) FROM (SELECT customer_id, as_of_date FROM features GROUP BY 1,2 HAVING COUNT(*)>1)').fetchone()[0]
        if duplicates:
            raise ValueError(f'Duplicate customer/snapshot pairs: {duplicates}')
        latest_order = con.execute('SELECT MAX(order_timestamp) FROM orders').fetchone()[0]
        # Require a fully observable 90-day window, measured relative to the last transaction.
        # Note: last transaction is a proxy for dataset end; audit source coverage separately.
        con.execute(f'''
            CREATE TEMP TABLE labeled AS
            SELECT f.customer_id, f.as_of_date, f.frequency_180d,
                   (f.frequency_180d > 0) AS eligible_active_180d,
                   COUNT(DISTINCT o.invoice_id) AS future_order_count_90d,
                   CAST(CASE WHEN COUNT(DISTINCT o.invoice_id)=0 THEN 1 ELSE 0 END AS INTEGER) AS churn_90d,
                   f.max_feature_event_at
            FROM features f
            LEFT JOIN orders o
              ON f.customer_id=o.customer_id
             AND o.order_timestamp >= f.as_of_date
             AND o.order_timestamp < f.as_of_date + INTERVAL '90 days'
            WHERE f.as_of_date + INTERVAL '90 days' <= TIMESTAMP '{latest_order.isoformat(sep=' ')}'
            GROUP BY f.customer_id, f.as_of_date, f.frequency_180d, f.max_feature_event_at
        ''')
        # No future-derived features are allowed.
        leakage = con.execute('''SELECT COUNT(*) FROM labeled
            WHERE max_feature_event_at IS NULL OR max_feature_event_at >= as_of_date''').fetchone()[0]
        if leakage:
            raise ValueError(f'Feature timestamp leakage/missing timestamp: {leakage}')
        missing_history = con.execute('''SELECT COUNT(*) FROM labeled l
            WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=l.customer_id AND o.order_timestamp<l.as_of_date)''').fetchone()[0]
        if missing_history:
            raise ValueError(f'Labeled rows without historical purchase: {missing_history}')
        bad_labels = con.execute('''SELECT COUNT(*) FROM labeled
            WHERE (churn_90d=1 AND future_order_count_90d<>0)
               OR (churn_90d=0 AND future_order_count_90d=0)''').fetchone()[0]
        if bad_labels:
            raise ValueError(f'Label inconsistencies: {bad_labels}')
        # Strong independent recomputation from the source orders.
        recompute = con.execute('''SELECT COUNT(*) FROM labeled l WHERE
            (l.churn_90d=1) <> NOT EXISTS (
                SELECT 1 FROM orders o WHERE o.customer_id=l.customer_id
                AND o.order_timestamp >= l.as_of_date
                AND o.order_timestamp < l.as_of_date + INTERVAL '90 days')''').fetchone()[0]
        if recompute:
            raise ValueError(f'Independent label mismatch: {recompute}')
        con.execute(f'''COPY (SELECT customer_id, as_of_date, eligible_active_180d,
            future_order_count_90d, churn_90d FROM labeled ORDER BY as_of_date, customer_id)
            TO {sql_literal(out / 'churn_labels_90d.parquet')} (FORMAT PARQUET)''')
        con.execute(f'''COPY (SELECT as_of_date, eligible_active_180d,
            COUNT(*) AS observations, SUM(churn_90d) AS no_repeat_count,
            ROUND(AVG(churn_90d)*100, 4) AS no_repeat_rate_pct
            FROM labeled GROUP BY 1,2 ORDER BY 1,2)
            TO {sql_literal(out / 'target_monthly_summary.csv')} (HEADER, DELIMITER ',')''')
        con.execute(f'''COPY (SELECT as_of_date, COUNT(*) AS total_customers,
            COUNT(*) FILTER (WHERE eligible_active_180d) AS active_customers,
            COUNT(*) FILTER (WHERE NOT eligible_active_180d) AS inactive_customers,
            ROUND(AVG(churn_90d)*100,4) AS all_customer_churn_pct,
            ROUND(AVG(churn_90d) FILTER (WHERE eligible_active_180d)*100,4) AS active_customer_churn_pct
            FROM labeled GROUP BY 1 ORDER BY 1)
            TO {sql_literal(out / 'target_eligibility_summary.csv')} (HEADER, DELIMITER ',')''')
        counts = con.execute('''SELECT COUNT(*), COUNT(DISTINCT customer_id), COUNT(DISTINCT as_of_date),
            MIN(as_of_date), MAX(as_of_date), SUM(CASE WHEN eligible_active_180d THEN 1 ELSE 0 END),
            SUM(churn_90d) FROM labeled''').fetchone()
        # Temporal candidate split: overlap across boundary needs embargo before training.
        split = con.execute('''SELECT
            CASE WHEN as_of_date < DATE '2011-04-01' THEN 'train_candidate'
                 WHEN as_of_date < DATE '2011-07-01' THEN 'validation_candidate'
                 ELSE 'test_candidate' END AS split,
            COUNT(*) AS observations, COUNT(DISTINCT as_of_date) AS snapshots,
            COUNT(*) FILTER (WHERE eligible_active_180d) AS active_observations,
            ROUND(AVG(churn_90d) FILTER (WHERE eligible_active_180d)*100,4) AS active_no_repeat_rate_pct,
            MIN(as_of_date) AS first_snapshot, MAX(as_of_date) AS last_snapshot
            FROM labeled GROUP BY 1 ORDER BY 1''').fetchall()
        split_rows = [dict(zip(['split','observations','snapshots','active_observations','active_no_repeat_rate_pct','first_snapshot','last_snapshot'], row)) for row in split]
        for row in split_rows:
            for field in ('first_snapshot','last_snapshot'):
                if row[field] is not None:
                    row[field] = row[field].isoformat()
        manifest = {
            'status': 'completed', 'run_utc': datetime.now(timezone.utc).isoformat(),
            'target': 'churn_90d: no valid invoice in [as_of_date, as_of_date + 90 days)',
            'source_last_order_timestamp': latest_order.isoformat(),
            'horizon_days': 90, 'labeled_rows': counts[0], 'distinct_customers': counts[1],
            'distinct_snapshots': counts[2],
            'first_labeled_snapshot': counts[3].isoformat() if counts[3] else None,
            'last_labeled_snapshot': counts[4].isoformat() if counts[4] else None,
            'active_180d_rows': counts[5], 'no_repeat_rows_all': counts[6],
            'duplicate_customer_snapshot_pairs': duplicates,
            'feature_future_event_violations': leakage,
            'rows_without_prior_purchase': missing_history,
            'label_inconsistencies': bad_labels,
            'independent_label_recompute_mismatches': recompute,
            'candidate_splits_not_approved': split_rows,
            'warnings': [
                'The last transaction timestamp is used as an observation-end proxy; verify dataset coverage independently.',
                'The candidate temporal splits have overlapping outcome windows at their boundaries. Do not train until a 90-day label embargo or rolling-origin evaluation is selected.',
                'A customer may appear in multiple monthly snapshots; rows are not independent.',
                'No model has been trained. No model feature set has been approved.'
            ],
            'outputs': ['churn_labels_90d.parquet','target_monthly_summary.csv','target_eligibility_summary.csv','target_manifest.json']
        }
        (out / 'target_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    finally:
        con.close()


if __name__ == '__main__':
    main()
