"""Exploratory K-Means diagnostics for Project 4 RFM snapshots.

Run from the project root:
    python -m pip install scikit-learn numpy
    python scripts/evaluate_kmeans.py --rfm output/rfm_segments.parquet --snapshot 2011-09-01 --output output

No supervised churn labels are used. Cluster IDs are arbitrary.
"""
import argparse
import csv
import json
from pathlib import Path

import duckdb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rfm', required=True)
    parser.add_argument('--snapshot', required=True, help='YYYY-MM-DD')
    parser.add_argument('--output', default='output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source = Path(args.rfm).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Parameters are supported in SELECT statements. The CREATE VIEW limitation
    # that affected build_rfm.py does not apply here.
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT customer_id, recency_days, frequency_180d, monetary_180d "
            "FROM read_parquet(?) "
            "WHERE lifecycle_status = 'active' AND CAST(as_of_date AS DATE) = CAST(? AS DATE) "
            "ORDER BY customer_id",
            [str(source), args.snapshot],
        ).fetchall()
    finally:
        connection.close()

    if len(rows) < 50:
        raise ValueError(f'At least 50 active customers required; found {len(rows)}')
    ids = [str(row[0]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate customer IDs in selected snapshot')
    raw = np.asarray([row[1:] for row in rows], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError('Null, NaN, or infinite RFM feature values detected')
    if (raw < 0).any():
        raise ValueError('Negative RFM feature values detected; review source data')

    scaled = RobustScaler().fit_transform(np.log1p(raw))
    diagnostics = []
    assignments = []
    profiles = []
    for k in range(2, min(9, len(rows))):
        model = KMeans(n_clusters=k, random_state=args.seed, n_init=20).fit(scaled)
        alternate = KMeans(n_clusters=k, random_state=args.seed + 1, n_init=20).fit(scaled)
        counts = np.bincount(model.labels_, minlength=k)
        if (counts == 0).any():
            raise ValueError(f'Empty cluster at k={k}')
        silhouette = silhouette_score(
            scaled, model.labels_, sample_size=min(len(rows), 2000),
            random_state=args.seed,
        )
        diagnostics.append({
            'k': k,
            'silhouette': round(float(silhouette), 4),
            'seed_stability_ari': round(float(adjusted_rand_score(model.labels_, alternate.labels_)), 4),
            'inertia': round(float(model.inertia_), 2),
            'min_cluster_size': int(counts.min()),
            'max_cluster_size': int(counts.max()),
            'min_cluster_pct': round(float(counts.min() / len(rows) * 100), 2),
        })
        for customer_id, label in zip(ids, model.labels_):
            assignments.append({
                'as_of_date': args.snapshot, 'customer_id': customer_id,
                'k': k, 'cluster_id': int(label),
            })
        for label in range(k):
            members = raw[model.labels_ == label]
            profiles.append({
                'as_of_date': args.snapshot,
                'k': k,
                'cluster_id': label,
                'customers': len(members),
                'share_pct': round(len(members) / len(rows) * 100, 2),
                'median_recency_days': round(float(np.median(members[:, 0])), 2),
                'median_frequency_180d': round(float(np.median(members[:, 1])), 2),
                'median_monetary_180d': round(float(np.median(members[:, 2])), 2),
            })

    write_csv(output / 'kmeans_diagnostics.csv', diagnostics, list(diagnostics[0]))
    write_csv(output / 'kmeans_exploratory_assignments.csv', assignments, list(assignments[0]))
    write_csv(output / 'kmeans_cluster_profiles.csv', profiles, list(profiles[0]))
    summary = {
        'status': 'completed', 'snapshot': args.snapshot,
        'active_customers': len(rows), 'seed': args.seed,
        'evaluated_k': [item['k'] for item in diagnostics],
        'diagnostics': diagnostics,
        'note': 'Exploratory clustering only; one snapshot cannot establish temporal stability. Do not use cluster IDs as churn labels.',
    }
    (output / 'kmeans_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
