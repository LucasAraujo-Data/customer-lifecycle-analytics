#!/usr/bin/env python3
"""Build audited, embargoed, chronological churn datasets. No model training."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import duckdb

FEATURES = [
    'recency_days', 'tenure_days', 'historical_order_count', 'prior_repeat_flag',
    'frequency_180d', 'monetary_180d', 'order_count_30d',
    'order_count_60d', 'order_count_90d', 'monetary_30d',
    'monetary_60d', 'monetary_90d', 'avg_order_value_180d',
    'avg_interpurchase_days',
]
SPLITS = {
    'train': ('2010-06-01', '2011-01-01'),
    'validation': ('2011-05-01', '2011-06-01'),
    'test': ('2011-09-01', '2011-09-01'),
}
HORIZON = 90


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def scalar(con, query):
    return con.execute(query).fetchone()[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', default='output/customer_features.parquet')
    p.add_argument('--labels', default='output/phase3_target/churn_labels_90d.parquet')
    p.add_argument('--output', default='output/phase3_splits')
    args = p.parse_args()
    feature_path, label_path = Path(args.features).resolve(), Path(args.labels).resolve()
    output = Path(args.output).resolve()
    for path in (feature_path, label_path):
        if not path.is_file():
            p.error(f'Input file not found: {path}')
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f'CREATE TEMP VIEW source_features AS SELECT * FROM read_parquet({quote(feature_path)})')
        con.execute(f'CREATE TEMP VIEW source_labels AS SELECT * FROM read_parquet({quote(label_path)})')
        fc = {r[1] for r in con.execute("PRAGMA table_info('source_features')").fetchall()}
        lc = {r[1] for r in con.execute("PRAGMA table_info('source_labels')").fetchall()}
        missing_f = set(FEATURES + ['customer_id','as_of_date','max_feature_event_at']) - fc
        missing_l = {'customer_id','as_of_date','eligible_active_180d','churn_90d'} - lc
        if missing_f or missing_l:
            raise ValueError(f'Missing columns: features={sorted(missing_f)} labels={sorted(missing_l)}')
        if scalar(con, "SELECT COUNT(*) FROM source_features WHERE customer_id IS NULL OR as_of_date IS NULL"):
            raise ValueError('Null feature keys')
        if scalar(con, "SELECT COUNT(*) FROM source_labels WHERE customer_id IS NULL OR as_of_date IS NULL"):
            raise ValueError('Null label keys')
        for view in ('source_features', 'source_labels'):
            n = scalar(con, f'''SELECT COUNT(*) FROM (SELECT customer_id, as_of_date FROM {view}
                GROUP BY 1,2 HAVING COUNT(*) > 1)''')
            if n:
                raise ValueError(f'Duplicate keys in {view}: {n}')
        con.execute('''CREATE TEMP TABLE joined AS
            SELECT f.*, l.churn_90d, l.eligible_active_180d
            FROM source_features f INNER JOIN source_labels l
              ON f.customer_id=l.customer_id AND f.as_of_date=l.as_of_date''')
        if scalar(con, 'SELECT COUNT(*) FROM joined WHERE churn_90d IS NULL OR churn_90d NOT IN (0,1)'):
            raise ValueError('Missing/invalid labels')
        if scalar(con, '''SELECT COUNT(*) FROM joined WHERE max_feature_event_at IS NULL
            OR max_feature_event_at >= as_of_date'''):
            raise ValueError('Feature timestamp leakage')
        if scalar(con, '''SELECT COUNT(*) FROM joined WHERE eligible_active_180d
            AND (frequency_180d IS NULL OR frequency_180d <= 0)'''):
            raise ValueError('Active eligibility conflicts with frequency_180d')
        cols = ', '.join('j.' + x for x in FEATURES)
        # Identifiers are for auditing and evaluation, not predictors.
        con.execute(f'''CREATE TEMP TABLE eligible AS SELECT j.customer_id, j.as_of_date,
            {cols}, j.churn_90d FROM joined j WHERE j.eligible_active_180d''')
        split_results = []
        for name, (start, end) in SPLITS.items():
            con.execute(f'''CREATE TEMP TABLE split_{name} AS SELECT * FROM eligible
                WHERE as_of_date >= DATE {quote(start)} AND as_of_date <= DATE {quote(end)}''')
            n, customers, snapshots, churn, min_date, max_date = con.execute(f'''
                SELECT COUNT(*), COUNT(DISTINCT customer_id), COUNT(DISTINCT as_of_date),
                SUM(churn_90d), MIN(as_of_date), MAX(as_of_date) FROM split_{name}''').fetchone()
            if not n:
                raise ValueError(f'Empty {name} split; check input data')
            if min_date.strftime('%Y-%m-%d') != start or max_date.strftime('%Y-%m-%d') != end:
                raise ValueError(f'Missing requested boundary snapshot in {name}')
            expected_months = {'train': 8, 'validation': 2, 'test': 1}[name]
            if snapshots != expected_months:
                raise ValueError(f'Expected {expected_months} monthly snapshots for {name}, got {snapshots}')
            con.execute(f'''COPY (SELECT * FROM split_{name} ORDER BY as_of_date, customer_id)
                TO {quote(output / (name + '.parquet'))} (FORMAT PARQUET)''')
            split_results.append({'split':name,'first_snapshot':start,'last_snapshot':end,
                'rows':n,'distinct_customers':customers,'snapshots':snapshots,
                'no_repeat_count':int(churn),'no_repeat_rate_pct':round(churn/n*100,4)})
        # Require full label observation before the next split begins; stricter than mere disjoint snapshots.
        train_end = datetime.fromisoformat(SPLITS['train'][1]) + timedelta(days=HORIZON)
        val_start = datetime.fromisoformat(SPLITS['validation'][0])
        val_end = datetime.fromisoformat(SPLITS['validation'][1]) + timedelta(days=HORIZON)
        test_start = datetime.fromisoformat(SPLITS['test'][0])
        if not (train_end <= val_start and val_end <= test_start):
            raise ValueError('Outcome windows overlap between splits')
        # No rows may be shared across splits.
        pair_overlap = scalar(con, '''
            SELECT COUNT(*) FROM (
                SELECT customer_id, as_of_date
                FROM (
                    SELECT customer_id, as_of_date FROM split_train
                    UNION ALL SELECT customer_id, as_of_date FROM split_validation
                    UNION ALL SELECT customer_id, as_of_date FROM split_test
                ) combined
                GROUP BY customer_id, as_of_date
                HAVING COUNT(*) > 1
            ) duplicates
        ''')
        if pair_overlap:
            raise ValueError('Customer-snapshot overlap')
        con.execute(f'''COPY (
            SELECT 'train' AS split, as_of_date, COUNT(*) AS rows, SUM(churn_90d) AS no_repeat,
                ROUND(100*AVG(churn_90d),4) AS no_repeat_rate_pct FROM split_train GROUP BY 2
            UNION ALL
            SELECT 'validation', as_of_date, COUNT(*), SUM(churn_90d), ROUND(100*AVG(churn_90d),4)
                FROM split_validation GROUP BY 2
            UNION ALL
            SELECT 'test', as_of_date, COUNT(*), SUM(churn_90d), ROUND(100*AVG(churn_90d),4)
                FROM split_test GROUP BY 2
            ORDER BY as_of_date)
            TO {quote(output / 'split_monthly_summary.csv')} (HEADER, DELIMITER ',')''')
        manifest = {
            'status':'completed', 'created_utc':datetime.now(timezone.utc).isoformat(),
            'target':'churn_90d', 'horizon_days':HORIZON,
            'population':'eligible_active_180d = true',
            'feature_columns':FEATURES,
            'excluded_from_predictors':['customer_id','as_of_date','churn_90d',
                'eligible_active_180d','future_order_count_90d','max_feature_event_at'],
            'splits':split_results,
            'embargo_checks':{'train_last_label_window_end_exclusive':train_end.isoformat(),
                'validation_first_snapshot':val_start.isoformat(),
                'validation_last_label_window_end_exclusive':val_end.isoformat(),
                'test_first_snapshot':test_start.isoformat(),
                'nonoverlapping_label_windows':True},
            'duplicate_customer_snapshot_pairs':0,
            'future_feature_events':0,
            'notes':[
                'Customer IDs can occur in different splits; only snapshot rows and outcome windows are disjoint.',
                'Fit imputation, scaling, encoding, feature selection and hyperparameters using train only.',
                'Validation has two months and final test one month; report uncertainty and consider rolling-origin sensitivity.',
                'The source data end date and churn target must be independently audited.',
                'Do not use label-derived columns or future order counts as model features.',
                'No model training performed.'
            ],
            'outputs':['train.parquet','validation.parquet','test.parquet','split_monthly_summary.csv','split_manifest.json']
        }
        (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps(manifest,indent=2))
    finally:
        con.close()

if __name__=='__main__':
    main()
