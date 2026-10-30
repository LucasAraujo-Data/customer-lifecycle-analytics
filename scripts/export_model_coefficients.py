"""Export Logistic Regression standardized coefficients from the selected fitted pipeline."""
import argparse, csv, json, math
from pathlib import Path
import joblib

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    selection=json.loads(Path(a.selection).read_text(encoding='utf-8'))
    if selection['selected_model']!='logistic_regression':raise ValueError('Selected model is not logistic regression')
    model=joblib.load(a.model)
    coefficients=model.named_steps['model'].coef_[0]
    features=selection['features']
    if len(coefficients)!=len(features):raise ValueError('Coefficient/feature count mismatch')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['feature','standardized_coefficient','odds_ratio_per_1_sd','interpretation'])
        for name,coefficient in sorted(zip(features,coefficients),key=lambda x:abs(x[1]),reverse=True):
            w.writerow([name,round(float(coefficient),6),round(math.exp(float(coefficient)),6),'Positive association with no repurchase' if coefficient>0 else 'Negative association with no repurchase'])
    print(f'Exported {len(features)} coefficients to {out}')
if __name__=='__main__':main()
