"""Read-only validation of Project 4 customer_features.parquet using DuckDB.
Usage: python validate_customer_features.py --file output/customer_features.parquet
"""
import argparse
import json
from pathlib import Path
import duckdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='output/customer_features.parquet')
    p.add_argument('--report', default='output/customer_features_validation.json')
    args = p.parse_args()
    src = Path(args.file).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    con = duckdb.connect(':memory:')
    con.execute('CREATE VIEW features AS SELECT * FROM read_parquet(' + "'" + str(src).replace("'", "''") + "'" + ')')
    desc = con.execute('DESCRIBE features').fetchall()
    cols = [r[0] for r in desc]
    print('COLUMNS:', ', '.join(f'{r[0]} ({r[1]})' for r in desc))
    print('ROWS:', con.execute('SELECT COUNT(*) FROM features').fetchone()[0])
    print('SAMPLE:')
    for row in con.execute('SELECT * FROM features LIMIT 3').fetchall():
        print(dict(zip(cols, [str(v) if v is not None else None for v in row])))

    def ident(s):
        return '"' + s.replace('"', '""') + '"'
    def choose(names):
        return next((c for n in names for c in cols if c.lower() == n), None)
    cust = choose(['customer_id', 'customerid', 'customer id'])
    snap = choose(['snapshot_date', 'snapshot_month', 'snapshot_at', 'as_of_date', 'snapshot'])
    rec = choose(['recency_days', 'recency'])
    freq = choose(['frequency', 'frequency_orders', 'order_count', 'orders_count', 'total_orders', 'historical_order_count'])
    mon = choose(['monetary', 'monetary_value', 'monetary_total', 'total_spend', 'total_revenue'])
    result = {'file': str(src), 'row_count': con.execute('SELECT COUNT(*) FROM features').fetchone()[0], 'columns': [{'name': r[0], 'type': r[1]} for r in desc], 'checks': {}}
    checks = result['checks']
    for col in cols:
        count = con.execute(f'SELECT COUNT(*) FROM features WHERE {ident(col)} IS NULL').fetchone()[0]
        if count:
            checks[f'nulls_{col}'] = count
    if cust and snap:
        checks['duplicate_customer_snapshot_pairs'] = con.execute(f'SELECT COUNT(*) - COUNT(DISTINCT ( {ident(cust)}, {ident(snap)} )) FROM features').fetchone()[0]
        checks['distinct_customers'] = con.execute(f'SELECT COUNT(DISTINCT {ident(cust)}) FROM features').fetchone()[0]
        checks['distinct_snapshots'] = con.execute(f'SELECT COUNT(DISTINCT {ident(snap)}) FROM features').fetchone()[0]
        checks['snapshot_range'] = [str(v) for v in con.execute(f'SELECT MIN({ident(snap)}), MAX({ident(snap)}) FROM features').fetchone()]
        checks['rows_by_snapshot'] = [{'snapshot': str(a), 'rows': b} for a,b in con.execute(f'SELECT {ident(snap)}, COUNT(*) FROM features GROUP BY 1 ORDER BY 1').fetchall()]
    else:
        checks['key_check'] = f'Not run: customer column={cust}, snapshot column={snap}'
    for name, col in [('recency',rec),('frequency',freq),('monetary',mon)]:
        if col:
            checks[f'{name}_summary'] = dict(zip(['min','max','negative_count','null_count'],con.execute(f'SELECT MIN({ident(col)}), MAX({ident(col)}), COUNT(*) FILTER (WHERE {ident(col)} < 0), COUNT(*) FILTER (WHERE {ident(col)} IS NULL) FROM features').fetchone()))
        else:
            checks[f'{name}_summary'] = 'Column not recognized; inspect schema'
    if freq:
        checks['nonpositive_frequency_count'] = con.execute(f'SELECT COUNT(*) FROM features WHERE {ident(freq)} <= 0').fetchone()[0]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('\nCHECKS:', json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    print('\nReport saved:', report)
    print('NOTE: A single feature parquet cannot establish temporal leakage absence. Compare against source orders.')

if __name__ == '__main__':
    main()
