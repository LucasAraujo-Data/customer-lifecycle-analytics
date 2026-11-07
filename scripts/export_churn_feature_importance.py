"""Export standardized Logistic Regression coefficients for Power BI.

Run from the customer-lifecycle-analytics repository root:
    python scripts/export_churn_feature_importance.py
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='output/phase3_models')
    parser.add_argument('--output', default='output/powerbi/churn_feature_importance.csv')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    selection = json.loads((model_dir / 'model_selection.json').read_text(encoding='utf-8'))
    pipeline = joblib.load(model_dir / 'selected_model.joblib')
    if selection['selected_model'] != 'logistic_regression':
        raise ValueError('Selected model is not Logistic Regression.')
    expected_steps = ['imputer', 'scaler', 'model']
    if list(pipeline.named_steps) != expected_steps:
        raise ValueError(f'Unexpected pipeline steps: {list(pipeline.named_steps)}')
    if type(pipeline.named_steps['scaler']).__name__ != 'StandardScaler':
        raise ValueError('Expected StandardScaler for standardized coefficients.')

    features = selection['features']
    estimator = pipeline.named_steps['model']
    coefficients = np.asarray(estimator.coef_)
    if coefficients.shape != (1, len(features)):
        raise ValueError(f'Unexpected coefficient shape: {coefficients.shape}')
    if len(pipeline.named_steps['imputer'].statistics_) != len(features):
        raise ValueError('Imputer feature count differs from selection manifest.')
    if len(pipeline.named_steps['scaler'].scale_) != len(features):
        raise ValueError('Scaler feature count differs from selection manifest.')
    if not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite coefficients.')

    records = []
    for feature, coefficient in zip(features, coefficients[0]):
        coefficient = float(coefficient)
        records.append({
            'feature': feature,
            'coefficient': coefficient,
            'absolute_coefficient': abs(coefficient),
            'direction': 'Positive' if coefficient > 0 else 'Negative' if coefficient < 0 else 'Neutral',
            'odds_ratio_per_1sd': float(np.exp(coefficient)),
        })
    records.sort(key=lambda row: row['absolute_coefficient'], reverse=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f'Exported {len(records)} features to {output}')
    print('Coefficients are for features standardized with training-fitted StandardScaler.')
    print('Positive coefficients increase predicted log-odds of churn; negative decrease them.')
    print('Associations are not causal; correlated features can affect coefficient stability.')
    for row in records:
        print(f"{row['feature']:<30} {row['coefficient']:+.6f}  {row['direction']}")


if __name__ == '__main__':
    main()
