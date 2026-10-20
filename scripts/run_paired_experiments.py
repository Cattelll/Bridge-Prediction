from pathlib import Path
import json, pickle, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import load_splits
from src.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'results'
MODEL_DIR = ROOT / 'outputs' / 'models'
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = {'DDS': 'processed', 'noDDS': 'processed_no_dds'}
METRICS = ['accuracy','f1_macro','f1_weighted','roc_auc','average_precision','precision_positive','recall_positive','f1_positive']
XGB_GRID = {
 'default (config.yaml)': dict(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8),
 'deeper_more_trees': dict(n_estimators=500,max_depth=8,learning_rate=0.05,subsample=0.9,colsample_bytree=0.8,min_child_weight=1,reg_lambda=1.0),
 'mid_capacity': dict(n_estimators=400,max_depth=7,learning_rate=0.08,subsample=0.85,colsample_bytree=0.7,min_child_weight=3,reg_lambda=1.5),
 'leaner_faster': dict(n_estimators=250,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8),
}
LGBM_GRID = {
 'default (config.yaml)': dict(n_estimators=300,num_leaves=63,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8),
 'more_leaves': dict(n_estimators=400,num_leaves=95,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8),
 'lower_lr_more_trees': dict(n_estimators=500,num_leaves=63,learning_rate=0.03,subsample=0.85,colsample_bytree=0.7),
 'deep_constrained': dict(n_estimators=350,num_leaves=127,max_depth=9,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8),
}
RF_GRID = {
 'default (config.yaml)': dict(n_estimators=300,max_depth=None,min_samples_split=5),
 'more_trees': dict(n_estimators=600,max_depth=None,min_samples_split=5),
 'regularized_shallow': dict(n_estimators=400,max_depth=25,min_samples_split=10,min_samples_leaf=4),
 'more_features': dict(n_estimators=400,max_depth=None,min_samples_split=5,max_features=0.5),
}

def score(dataset, algorithm, experiment, params, model, split, X, y, le, threshold=0.5):
    proba=model.predict_proba(X)
    pred=(proba[:,1] >= threshold).astype(int) if threshold != 0.5 else model.predict(X)
    res=evaluate(y,pred,proba,le,model_name=algorithm)
    return {'dataset':dataset,'algorithm':algorithm,'experiment':experiment,'split':split,
            'threshold':float(threshold),'parameters':json.dumps(params,ensure_ascii=False),
            **{m:float(res[m]) for m in METRICS if m in res}}

all_rows=[]
selections={}
for variant, folder in DATASETS.items():
    print('\n===',variant,'===',flush=True)
    tr,va,te,features,le=load_splits(ROOT/'data'/folder)
    Xtr=tr[features].to_numpy(dtype=np.float32); ytr=tr['label'].to_numpy()
    Xva=va[features].to_numpy(dtype=np.float32); yva=va['label'].to_numpy()
    Xte=te[features].to_numpy(dtype=np.float32); yte=te['label'].to_numpy()
    assert len(features)==(174 if variant=='DDS' else 164)
    # Existing fitted baseline scores are recorded for both validation and test.
    baselines={}
    for algo,fn in [('RandomForest','randomforest'),('XGBoost','xgboost'),('LightGBM','lightgbm')]:
        suffix='_dds' if variant=='DDS' else ''
        with open(MODEL_DIR/f'{fn}{suffix}.pkl','rb') as f: model=pickle.load(f)
        params=getattr(model,'params',{})
        baselines[algo]=model
        for split,X,y in [('validation',Xva,yva),('test',Xte,yte)]:
            all_rows.append(score(variant,algo,'Baseline',params,model,split,X,y,le))
    # Candidate grids are selected by AP on validation only.
    grids={}
    for cand,params in XGB_GRID.items():
        clf=XGBClassifier(**params,random_state=42,n_jobs=-1,verbosity=0,eval_metric='logloss')
        clf.fit(Xtr,ytr); pred=clf.predict(Xva); proba=clf.predict_proba(Xva)
        res=evaluate(yva,pred,proba,le,model_name='XGBoost')
        grids[cand]={'model':clf,'params':params,'res':res}
    xb=max(grids,key=lambda k:grids[k]['res']['average_precision'])
    xgb=grids[xb]['model']; xgbp=grids[xb]['params']
    all_rows.append(score(variant,'XGBoost','Exp A: retuned',xgbp,xgb,'validation',Xva,yva,le))
    # Full class weights, based on AP-selected XGB hyperparameters.
    sw=compute_sample_weight('balanced',ytr)
    xgb_b=XGBClassifier(**xgbp,random_state=42,n_jobs=-1,verbosity=0,eval_metric='logloss')
    xgb_b.fit(Xtr,ytr,sample_weight=sw)
    all_rows.append(score(variant,'XGBoost','Exp B: balanced penuh',xgbp,xgb_b,'validation',Xva,yva,le))
    # Soft-balanced alpha is selected by AP validation, consistent with primary selection metric.
    alphas={}
    for alpha in [0.25,0.5,0.75]:
        clf=XGBClassifier(**xgbp,random_state=42,n_jobs=-1,verbosity=0,eval_metric='logloss')
        clf.fit(Xtr,ytr,sample_weight=1.0+alpha*(sw-1.0))
        proba=clf.predict_proba(Xva); pred=clf.predict(Xva)
        res=evaluate(yva,pred,proba,le,model_name='XGBoost')
        alphas[alpha]={'model':clf,'res':res}
    alpha=max(alphas,key=lambda a:alphas[a]['res']['average_precision'])
    xgb_f=alphas[alpha]['model']
    all_rows.append(score(variant,'XGBoost',f'Exp F: soft-balanced alpha={alpha}',{'base':xgbp,'alpha':alpha},xgb_f,'validation',Xva,yva,le))
    # Threshold is chosen using F1 macro on validation.
    thr_grid=np.linspace(0.01,0.99,197)
    pva=xgb_f.predict_proba(Xva)[:,1]
    f1s=[f1_score(yva,(pva>=t).astype(int),average='macro',zero_division=0) for t in thr_grid]
    threshold=float(thr_grid[int(np.argmax(f1s))])
    all_rows.append(score(variant,'XGBoost',f'Exp F + threshold={threshold:.3f}',{'base':xgbp,'alpha':alpha},xgb_f,'validation',Xva,yva,le,threshold))
    # LightGBM grid, with balanced class weighting throughout.
    lg={}
    for cand,params in LGBM_GRID.items():
        clf=LGBMClassifier(**params,random_state=42,n_jobs=-1,verbose=-1,class_weight='balanced')
        clf.fit(Xtr,ytr); proba=clf.predict_proba(Xva); pred=clf.predict(Xva)
        res=evaluate(yva,pred,proba,le,model_name='LightGBM')
        lg[cand]={'model':clf,'params':params,'res':res}
    lb=max(lg,key=lambda k:lg[k]['res']['average_precision'])
    lgbm=lg[lb]['model']; lgbmp=lg[lb]['params']
    all_rows.append(score(variant,'LightGBM','Exp C: retuned',lgbmp,lgbm,'validation',Xva,yva,le))
    # Random Forest grid, with balanced class weighting throughout.
    rf={}
    for cand,params in RF_GRID.items():
        clf=RandomForestClassifier(**params,class_weight='balanced',random_state=42,n_jobs=-1)
        clf.fit(Xtr,ytr); proba=clf.predict_proba(Xva); pred=clf.predict(Xva)
        res=evaluate(yva,pred,proba,le,model_name='RandomForest')
        rf[cand]={'model':clf,'params':params,'res':res}
    rb=max(rf,key=lambda k:rf[k]['res']['average_precision'])
    rfm=rf[rb]['model']; rfmp=rf[rb]['params']
    all_rows.append(score(variant,'RandomForest','Exp E: retuned',rfmp,rfm,'validation',Xva,yva,le))
    # Per-algorithm selected candidates: validation AP for model, F1 macro for threshold.
    chosen={'RandomForest':(rfm,rfmp,0.5, f'Exp E: {rb}'),
            'XGBoost':(xgb,xgbp,0.5,f'Exp A: {xb}'),
            'LightGBM':(lgbm,lgbmp,0.5,f'Exp C: {lb}')}
    for algo,(model,params,th,exp) in chosen.items():
        all_rows.append(score(variant,algo,exp,params,model,'test',Xte,yte,le,th))
    all_rows.append(score(variant,'XGBoost',f'Exp F + threshold={threshold:.3f}',{'base':xgbp,'alpha':alpha},xgb_f,'test',Xte,yte,le,threshold))
    selections[variant]={'xgb_grid_candidate':xb,'xgb_grid_params':xgbp,'lgbm_grid_candidate':lb,'lgbm_grid_params':lgbmp,
                        'rf_grid_candidate':rb,'rf_grid_params':rfmp,'best_softbalance_alpha_by_validation_ap':alpha,
                        'best_threshold_by_validation_f1_macro':threshold}
    print('Dipilih validasi berdasarkan AP - XGB:',xb,'LGBM:',lb,'RF:',rb,'alpha:',alpha,'threshold:',round(threshold,3),flush=True)

results=pd.DataFrame(all_rows)
results.to_csv(OUT/'nb06_paired_experiments_results.csv',index=False)
# Summary of validation accuracy maxima and test accuracy maxima (test candidates were selected by validation rules above).
val=results[results.split=='validation']
test=results[results.split=='test']
summary={'selection_protocol':{'model_and_hyperparameters':'average_precision on validation','threshold':'f1_macro on validation','test_used_for_selection':False},
         'selected_parameters_by_dataset':selections,
         'best_validation_accuracy':val.loc[val.accuracy.idxmax()].to_dict(),
         'best_test_accuracy_descriptive_only':test.loc[test.accuracy.idxmax()].to_dict(),
         'validation_candidate_count':int(len(val)),'test_candidate_count':int(len(test))}
(OUT/'nb06_paired_experiments_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('\nTop validation accuracy:')
print(val.sort_values('accuracy',ascending=False)[['dataset','algorithm','experiment','accuracy','average_precision']].head(10).to_string(index=False))
print('\nTop test accuracy among validation-selected candidate evaluations (descriptive only):')
print(test.sort_values('accuracy',ascending=False)[['dataset','algorithm','experiment','accuracy','average_precision']].head(10).to_string(index=False))
