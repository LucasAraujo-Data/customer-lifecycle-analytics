"""Export frozen held-out churn predictions for Power BI, with evaluation checks."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', default='output/phase3_splits/test.parquet')
    parser.add_argument('--model-dir', default='output/phase3_models')
    parser.add_argument('--output', default='output/powerbi/churn_test_predictions.csv')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    selection = json.loads((model_dir / 'model_selection.json').read_text(encoding='utf-8'))
    model = joblib.load(model_dir / 'selected_model.joblib')
    df = pd.read_parquet(args.test)
    features = selection['features']
    required = features + ['churn_90d', 'customer_id', 'as_of_date']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df.duplicated(['customer_id', 'as_of_date']).any():
        raise ValueError('Duplicate customer snapshots in test set')
    if not (pd.to_datetime(df['as_of_date']).min() > pd.Timestamp(selection['validation_snapshots'][1]) + pd.Timedelta(days=90)):
        raise ValueError('Test does not start after validation label embargo')

    x = df[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    y = df['churn_90d'].astype(int).to_numpy()
    probability = model.predict_proba(x)[:, 1]
    threshold = float(selection['selected_threshold'])
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())

    final_metrics_path = model_dir / 'final_test_metrics.json'
    if not final_metrics_path.is_file():
        raise FileNotFoundError(f'Missing original test evaluation: {final_metrics_path}')
    reference = json.loads(final_metrics_path.read_text(encoding='utf-8'))
    actual = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    expected = {key: int(reference[key]) for key in actual}
    if actual != expected or len(df) != int(reference['rows']):
        raise ValueError(f'Predictions do not reproduce final test evaluation. Actual: {actual}; expected: {expected}')
    if not np.isclose(threshold, float(reference['frozen_threshold'])):
        raise ValueError('Frozen threshold does not match original test evaluation')

    # Risk bands are descriptive reporting categories, not alternative thresholds.
    risk_band = pd.cut(
        probability,
        bins=[-np.inf, 0.2, 0.3, 0.5, 0.7, np.inf],
        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
        right=False,
    )
    band_order = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
    export = pd.DataFrame({
        'customer_id': df['customer_id'].astype(str).to_numpy(),
        'as_of_date': pd.to_datetime(df['as_of_date']).dt.strftime('%Y-%m-%d').to_numpy(),
        'churn_probability': probability,
        'actual_churn': y,
        'predicted_churn': predicted,
        'risk_band': risk_band.astype(str),
        'risk_band_sort': [band_order[str(value)] for value in risk_band],
    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Exported {len(export):,} rows to {output_path}')
    print(f'Frozen threshold: {threshold:.2f}')
    print(f'Validated confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    print('Risk band counts:')
    print(export.groupby(['risk_band_sort', 'risk_band'], sort=True).size().to_string())


if __name__ == '__main__':
    main()
