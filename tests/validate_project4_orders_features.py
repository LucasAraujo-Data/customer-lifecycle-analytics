"""Validate Project 4 historical customer features against cleaned orders.
Run: python validate_project4_orders_features.py --orders output/orders_clean.parquet --features output/customer_features.parquet
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import duckdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--orders', default='output/orders_clean.parquet')
    p.add_argument('--features', default='output/customer_features.parquet')
    p.add_argument('--report', default='output/orders_features_reconciliation.json')
    a = p.parse_args()
    for f in (a.orders, a.features):
        if not Path(f).is_file():
            p.error(f'File not found: {f}')
    con = duckdb.connect()
    con.execute('PRAGMA threads=4')
    con.execute("CREATE TEMP VIEW orders AS SELECT * FROM read_parquet('" + str(Path(a.orders).resolve()).replace("'", "''") + "')")
    con.execute("CREATE TEMP VIEW features AS SELECT * FROM read_parquet('" + str(Path(a.features).resolve()).replace("'", "''") + "')")
    orders_cols = [r[0] for r in con.execute('DESCRIBE orders').fetchall()]
    feature_cols = [r[0] for r in con.execute('DESCRIBE features').fetchall()]
    required_orders = {'customer_id','invoice_id','order_timestamp','gross_purchase_value'}
    required_features = {'customer_id','as_of_date','recency_days','tenure_days','historical_order_count','prior_repeat_flag','frequency_180d','monetary_180d','avg_order_value_180d','max_feature_event_at'}
    report = {'generated_utc':datetime.now(timezone.utc).isoformat(), 'orders_columns':orders_cols, 'features_columns':feature_cols}
    missing = {'orders':sorted(required_orders-set(orders_cols)), 'features':sorted(required_features-set(feature_cols))}
    report['missing_required_columns'] = missing
    if missing['orders'] or missing['features']:
        print(json.dumps(report, indent=2)); return
    report['source_counts'] = con.execute('SELECT (SELECT COUNT(*) FROM orders), (SELECT COUNT(*) FROM features), (SELECT COUNT(DISTINCT customer_id) FROM orders), (SELECT COUNT(DISTINCT customer_id) FROM features)').fetchone()
    report['duplicate_customer_snapshot_pairs'] = con.execute('SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM features GROUP BY customer_id, as_of_date HAVING COUNT(*)>1)').fetchone()[0]
    report['features_with_future_event'] = con.execute('SELECT COUNT(*) FROM features WHERE max_feature_event_at >= as_of_date').fetchone()[0]
    report['features_without_prior_order'] = con.execute('''SELECT COUNT(*) FROM features f WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=f.customer_id AND o.order_timestamp<f.as_of_date)''').fetchone()[0]
    report['orders_customers_not_in_features'] = con.execute('''SELECT COUNT(*) FROM (SELECT DISTINCT customer_id FROM orders EXCEPT SELECT DISTINCT customer_id FROM features)''').fetchone()[0]
    report['orders_customers_not_in_features_examples'] = [r[0] for r in con.execute('''SELECT DISTINCT customer_id FROM orders EXCEPT SELECT DISTINCT customer_id FROM features LIMIT 20''').fetchall()]
    con.execute('''CREATE TEMP TABLE reconciled AS
        SELECT f.customer_id, f.as_of_date, f.recency_days, f.tenure_days,
               f.historical_order_count, f.prior_repeat_flag, f.frequency_180d,
               f.monetary_180d, f.avg_order_value_180d, f.max_feature_event_at,
               COUNT(o.invoice_id) AS expected_historical_order_count,
               MIN(o.order_timestamp) AS first_order_at,
               MAX(o.order_timestamp) AS last_order_at,
               COUNT(o.invoice_id) FILTER (WHERE o.order_timestamp >= f.as_of_date - INTERVAL 180 DAYS) AS expected_frequency_180d,
               COALESCE(SUM(o.gross_purchase_value) FILTER (WHERE o.order_timestamp >= f.as_of_date - INTERVAL 180 DAYS),0) AS expected_monetary_180d
        FROM features f LEFT JOIN orders o
          ON f.customer_id=o.customer_id AND o.order_timestamp<f.as_of_date
        GROUP BY ALL''')
    checks = {
        'historical_order_count_mismatch': 'historical_order_count IS DISTINCT FROM expected_historical_order_count',
        'prior_repeat_flag_mismatch': 'prior_repeat_flag IS DISTINCT FROM (expected_historical_order_count >= 2)',
        'frequency_180d_mismatch': 'frequency_180d IS DISTINCT FROM expected_frequency_180d',
        'monetary_180d_mismatch_over_0_01': 'monetary_180d IS NULL OR ABS(monetary_180d-expected_monetary_180d)>0.01',
        'recency_days_mismatch': "recency_days IS DISTINCT FROM DATE_DIFF('day',CAST(last_order_at AS DATE),CAST(as_of_date AS DATE))",
        'tenure_days_mismatch': "tenure_days IS DISTINCT FROM DATE_DIFF('day',CAST(first_order_at AS DATE),CAST(as_of_date AS DATE))",
        'max_feature_event_at_mismatch': 'max_feature_event_at IS DISTINCT FROM last_order_at',
        'avg_order_value_180d_mismatch': '''(expected_frequency_180d=0 AND avg_order_value_180d IS NOT NULL) OR
             (expected_frequency_180d>0 AND (avg_order_value_180d IS NULL OR ABS(avg_order_value_180d-expected_monetary_180d/expected_frequency_180d)>0.01))''',
    }
    report['reconciliation_checks'] = {}
    for name, expr in checks.items():
        count = con.execute(f'SELECT COUNT(*) FROM reconciled WHERE {expr}').fetchone()[0]
        examples = con.execute(f'SELECT customer_id, CAST(as_of_date AS VARCHAR) FROM reconciled WHERE {expr} LIMIT 5').fetchall() if count else []
        report['reconciliation_checks'][name] = {'mismatches':count,'examples':examples}
    report['snapshots'] = [dict(zip(('as_of_date','rows','missing_180d_orders'),r)) for r in con.execute('''SELECT CAST(as_of_date AS VARCHAR),COUNT(*),COUNT(*) FILTER (WHERE expected_frequency_180d=0) FROM reconciled GROUP BY 1 ORDER BY 1''').fetchall()]
    report['notes'] = ['This checks features against orders_clean only; independent reconciliation to raw Excel and business policy still matters.', 'Any mismatch in recency/tenure may reflect a date-vs-timestamp definition difference; inspect examples before changing production code.', 'December 2011 snapshots cannot have a complete 90-day future label with this dataset.']
    out=Path(a.report);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(json.dumps({'report':str(out),'source_counts':report['source_counts'],'future_event_violations':report['features_with_future_event'],'reconciliation_mismatches':{k:v['mismatches'] for k,v in report['reconciliation_checks'].items()}},indent=2))

if __name__=='__main__':main()
