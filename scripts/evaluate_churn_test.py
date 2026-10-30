"""Phase 3: one-time held-out test evaluation of the frozen model and threshold."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support,confusion_matrix,brier_score_loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--test',required=True);ap.add_argument('--model-dir',required=True);args=ap.parse_args()
    from train_churn_models import read_data
    out=Path(args.model_dir);selection=json.loads((out/'model_selection.json').read_text(encoding='utf-8'))
    model=joblib.load(out/'selected_model.joblib')
    df=read_data(args.test); features=selection['features']
    missing=[c for c in features+['churn_90d','customer_id','as_of_date'] if c not in df.columns]
    if missing:raise ValueError(f'Missing columns: {missing}')
    if df.duplicated(['customer_id','as_of_date']).any():raise ValueError('Duplicate customer snapshots')
    if not (pd.to_datetime(df['as_of_date']).min()>pd.Timestamp(selection['validation_snapshots'][1])+pd.Timedelta(days=90)):
        raise ValueError('Test does not start after validation label embargo')
    y=df['churn_90d'].astype(int).to_numpy();x=df[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    prob=model.predict_proba(x)[:,1];threshold=float(selection['selected_threshold']);pred=(prob>=threshold).astype(int)
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    result={'status':'final_test_evaluated','selected_model':selection['selected_model'],'frozen_threshold':threshold,'rows':len(df),'positive_rate':float(y.mean()),'pr_auc':float(average_precision_score(y,prob)),'roc_auc':float(roc_auc_score(y,prob)) if len(set(y))==2 else None,'precision':float(p),'recall':float(r),'f1':float(f),'brier':float(brier_score_loss(y,prob)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'note':'Final holdout: do not tune on these metrics.'}
    (out/'final_test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
