"""Phase 3: fit and select churn models without opening the final test set."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['recency_days','tenure_days','historical_order_count','prior_repeat_flag','frequency_180d','monetary_180d','order_count_30d','order_count_60d','order_count_90d','monetary_30d','monetary_60d','monetary_90d','avg_order_value_180d','avg_interpurchase_days']
TARGET = 'churn_90d'

def read_data(path):
    import duckdb
    con=duckdb.connect(':memory:')
    try: return con.read_parquet(str(path)).df()
    finally: con.close()

def validate(df, name):
    missing=[c for c in FEATURES+[TARGET,'customer_id','as_of_date'] if c not in df.columns]
    if missing: raise ValueError(f'{name}: missing columns {missing}')
    if df.duplicated(['customer_id','as_of_date']).any(): raise ValueError(f'{name}: duplicate customer/snapshot')
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0,1}): raise ValueError(f'{name}: invalid target')
    if df[FEATURES].apply(lambda s: pd.to_numeric(s,errors='coerce')).replace([np.inf,-np.inf],np.nan).isna().all(axis=0).any():
        raise ValueError(f'{name}: at least one feature entirely null/non-numeric')
    return pd.to_datetime(df['as_of_date']).min(),pd.to_datetime(df['as_of_date']).max()

def metrics(y,prob,threshold):
    pred=(prob>=threshold).astype(int)
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'pr_auc':float(average_precision_score(y,prob)),'roc_auc':float(roc_auc_score(y,prob)), 'precision':float(p),'recall':float(r),'f1':float(f),'brier':float(brier_score_loss(y,prob)),'threshold':float(threshold),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True);ap.add_argument('--validation',required=True);ap.add_argument('--output',default='output/phase3_models')
    args=ap.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=read_data(args.train); val=read_data(args.validation)
    t0,t1=validate(train,'train');v0,v1=validate(val,'validation')
    if t1>=v0: raise ValueError('Train and validation snapshot periods overlap or are reversed')
    if t1+pd.Timedelta(days=90)>v0: raise ValueError('90-day label windows overlap train/validation')
    for df in (train,val):
        if 'eligible_active_180d' in df and not df['eligible_active_180d'].astype(bool).all(): raise ValueError('Noneligible customers present')
    xtr=train[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    xv=val[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    ytr=train[TARGET].astype(int).to_numpy();yv=val[TARGET].astype(int).to_numpy()
    if len(set(ytr))<2 or len(set(yv))<2: raise ValueError('Both target classes required')
    models={
      'majority_baseline':DummyClassifier(strategy='prior'),
      'logistic_regression':Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=42))]),
      'random_forest':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=250,min_samples_leaf=20,max_features='sqrt',n_jobs=-1,random_state=42))])
    }
    thresholds=np.round(np.arange(.10,.901,.05),2)
    results=[]; fitted={}; predictions={}
    for name,model in models.items():
        model.fit(xtr,ytr);prob=model.predict_proba(xv)[:,1]
        fitted[name]=model;predictions[name]=prob
        for threshold in ([.5] if name=='majority_baseline' else thresholds):
            results.append({'model':name,**metrics(yv,prob,threshold)})
    # Select by validation PR-AUC, then F1 threshold (no test used).
    model_candidates=[r for r in results if r['model']!='majority_baseline']
    best_model_name=max(set(r['model'] for r in model_candidates),key=lambda n: max(r['pr_auc'] for r in model_candidates if r['model']==n))
    best=max((r for r in model_candidates if r['model']==best_model_name),key=lambda r:(r['f1'],-abs(r['threshold']-.5)))
    pd.DataFrame(results).to_csv(out/'validation_metrics.csv',index=False)
    joblib.dump(fitted[best_model_name],out/'selected_model.joblib')
    selection={'status':'selected_on_validation_only','created_utc':datetime.now(timezone.utc).isoformat(),'features':FEATURES,'train_rows':len(train),'validation_rows':len(val),'train_snapshots':[str(t0.date()),str(t1.date())],'validation_snapshots':[str(v0.date()),str(v1.date())],'selection_metric':'validation PR-AUC; threshold selected by validation F1','selected_model':best_model_name,'selected_threshold':best['threshold'],'selected_validation_metrics':best,'baseline_validation_metrics':next(r for r in results if r['model']=='majority_baseline'),'note':'Do not alter selection after inspecting final test results. Test is not read by this script.'}
    (out/'model_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(selection,indent=2))
if __name__=='__main__': main()
