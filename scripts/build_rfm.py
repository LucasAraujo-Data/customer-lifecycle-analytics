"""Build reproducible, snapshot-local RFM scores and business segments.
Usage: python scripts/build_rfm.py --features output/customer_features.parquet --output output
"""
import argparse
import json
from pathlib import Path
import duckdb

SQL = r'''
CREATE OR REPLACE TEMP VIEW features AS SELECT * FROM read_parquet(?);
'''
# Scores are calculated within each snapshot, never across future snapshots.
RFM_SQL = r'''
COPY (
WITH base AS (
 SELECT customer_id, as_of_date, feature_version, recency_days,
        frequency_180d, monetary_180d, historical_order_count, prior_repeat_flag,
        CASE WHEN frequency_180d > 0 THEN 'active' ELSE 'inactive_180d' END AS lifecycle_status
 FROM features
), scored AS (
 SELECT *,
  CASE WHEN lifecycle_status = 'active' THEN
    NTILE(5) OVER (PARTITION BY as_of_date, lifecycle_status ORDER BY recency_days DESC, customer_id)
  END AS r_score,
  CASE WHEN lifecycle_status = 'active' THEN
    NTILE(5) OVER (PARTITION BY as_of_date, lifecycle_status ORDER BY frequency_180d ASC, customer_id)
  END AS f_score,
  CASE WHEN lifecycle_status = 'active' THEN
    NTILE(5) OVER (PARTITION BY as_of_date, lifecycle_status ORDER BY monetary_180d ASC, customer_id)
  END AS m_score
 FROM base
), segments AS (
 SELECT *, CASE
  WHEN lifecycle_status = 'inactive_180d' THEN 'Inactive 180d'
  WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
  WHEN r_score >= 3 AND f_score >= 4 THEN 'Loyal Customers'
  WHEN r_score >= 4 AND f_score <= 2 THEN 'Recent / Developing'
  WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
  WHEN r_score <= 2 AND f_score <= 2 THEN 'Hibernating'
  ELSE 'Needs Attention' END AS rfm_segment
 FROM scored
)
SELECT * FROM segments
) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
'''

def main():
 p = argparse.ArgumentParser()
 p.add_argument('--features', required=True)
 p.add_argument('--output', default='output')
 a=p.parse_args()
 src=Path(a.features).resolve(); out=Path(a.output).resolve()
 if not src.is_file(): raise FileNotFoundError(src)
 out.mkdir(parents=True,exist_ok=True)
 con=duckdb.connect()
 con.execute("CREATE TEMP VIEW features AS SELECT * FROM read_parquet('" + str(src).replace("'", "''") + "')")
 required={'customer_id','as_of_date','recency_days','frequency_180d','monetary_180d','historical_order_count','prior_repeat_flag','feature_version'}
 cols={r[0] for r in con.execute('DESCRIBE features').fetchall()}
 if required-cols: raise ValueError(f'Missing feature columns: {sorted(required-cols)}')
 if con.execute('SELECT count(*) FROM features WHERE customer_id IS NULL OR as_of_date IS NULL OR frequency_180d IS NULL OR monetary_180d IS NULL OR recency_days IS NULL').fetchone()[0]:
  raise ValueError('Nulls in required features')
 if con.execute('SELECT count(*) FROM (SELECT customer_id,as_of_date,count(*) n FROM features GROUP BY 1,2 HAVING count(*)>1)').fetchone()[0]:
  raise ValueError('Duplicate customer/snapshot rows')
 path=out/'rfm_segments.parquet'
 con.execute(RFM_SQL,[str(path)])
 con.execute("CREATE TEMP VIEW segments AS SELECT * FROM read_parquet('" + str(path).replace("'", "''") + "')")
 count=con.execute('SELECT count(*) FROM segments').fetchone()[0]
 original=con.execute('SELECT count(*) FROM features').fetchone()[0]
 if count != original: raise ValueError('RFM row count mismatch')
 null_active=con.execute("SELECT count(*) FROM segments WHERE lifecycle_status='active' AND (r_score IS NULL OR f_score IS NULL OR m_score IS NULL)").fetchone()[0]
 if null_active: raise ValueError('Active customer has null RFM score')
 con.execute("COPY (SELECT as_of_date, lifecycle_status, rfm_segment, COUNT(*) AS customers, ROUND(AVG(recency_days),2) AS avg_recency_days, ROUND(AVG(frequency_180d),2) AS avg_frequency_180d, ROUND(AVG(monetary_180d),2) AS avg_monetary_180d FROM segments GROUP BY 1,2,3 ORDER BY 1,2,3) TO ? (FORMAT CSV, HEADER TRUE)",[str(out/'rfm_segment_summary.csv')])
 by_status=con.execute('SELECT lifecycle_status,count(*) FROM segments GROUP BY 1 ORDER BY 1').fetchall()
 summary={'status':'completed','source':src.name,'rows':count,'distinct_snapshots':con.execute('SELECT count(DISTINCT as_of_date) FROM segments').fetchone()[0], 'by_status':dict(by_status),'score_method':'NTILE(5) within snapshot, active customers only; tie broken by customer_id','inactive_rule':'frequency_180d = 0','outputs':['rfm_segments.parquet','rfm_segment_summary.csv']}
 (out/'rfm_build_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))
 con.close()
if __name__=='__main__':main()
