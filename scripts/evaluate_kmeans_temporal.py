"""Run independent K-Means diagnostics on multiple historical RFM snapshots.

Requires evaluate_kmeans_reviewed.py in the same directory, duckdb, numpy,
scikit-learn. This is a repeated cross-sectional robustness check, NOT
out-of-time stability of a fixed fitted clustering model.

Example:
 python scripts/evaluate_kmeans_temporal.py --rfm output/rfm_segments.parquet --output output/kmeans_temporal
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_DATES = ['2010-09-01', '2011-03-01', '2011-09-01']

def load_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rfm', required=True)
    p.add_argument('--output', default='output/kmeans_temporal')
    p.add_argument('--dates', nargs='+', default=DEFAULT_DATES)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    source = Path(args.rfm).resolve()
    if not source.is_file():
        p.error(f'RFM file not found: {source}')
    base = Path(args.output).resolve()
    base.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().with_name('evaluate_kmeans_reviewed.py')
    if not runner.is_file():
        p.error(f'Place evaluate_kmeans_reviewed.py next to this script: {runner}')
    combined_diag, combined_profiles, snapshot_summaries = [], [], []
    for date in args.dates:
        folder = base / date
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(runner), '--rfm', str(source), '--snapshot', date,
                   '--output', str(folder), '--seed', str(args.seed)]
        print(f'Processing snapshot {date}...', flush=True)
        subprocess.run(command, check=True, stdout=(folder / 'run.log').open('w', encoding='utf-8'))
        manifest = json.loads((folder / 'kmeans_manifest.json').read_text(encoding='utf-8'))
        snapshot_summaries.append({'snapshot': date, 'active_customers': manifest['active_customers']})
        for item in load_csv(folder / 'kmeans_diagnostics.csv'):
            combined_diag.append({'snapshot': date, 'active_customers': manifest['active_customers'], **item})
        combined_profiles.extend(load_csv(folder / 'kmeans_cluster_profiles.csv'))
    def write_csv(path, records):
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_csv(base / 'kmeans_temporal_diagnostics.csv', combined_diag)
    write_csv(base / 'kmeans_temporal_profiles.csv', combined_profiles)
    summary = {
        'status': 'completed', 'snapshots': snapshot_summaries,
        'scope': 'Independent clustering fitted separately at each snapshot',
        'interpretation': 'Compare silhouette, cluster shares, and RFM medians across dates. Cluster IDs are not aligned across snapshots.',
        'limitations': [
            'Seed ARI measures initialization robustness within one snapshot, not temporal stability.',
            'A fixed-model out-of-time test and customer migration analysis are not performed.',
            'No churn labels are used.',
        ],
        'outputs': ['kmeans_temporal_diagnostics.csv', 'kmeans_temporal_profiles.csv'],
    }
    (base / 'kmeans_temporal_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
