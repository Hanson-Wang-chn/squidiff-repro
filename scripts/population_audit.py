"""Independent real-cell day classifier as a post-hoc population diagnostic.

All four true days are used ONLY to fit this diagnostic classifier. The diffusion
model is never updated. Real evaluation cell IDs are excluded from classifier fit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,confusion_matrix
from common import ROOT,write_json,record

def main(args):
    te=ad.read_h5ad(ROOT/args.data_root/'ipsc/test.h5ad')
    a=ad.read_h5ad(ROOT/'data/cache/ipsc_full.h5ad')
    pool=a[~a.obs_names.isin(te.obs_names)]
    rng=np.random.default_rng(20261113);ids=[]
    for g in range(4):ids.extend(rng.choice(pool.obs_names[pool.obs.Group==g].to_numpy(),1200,replace=False))
    ref=a[ids,te.var_names].copy();assert set(ref.obs_names).isdisjoint(te.obs_names)
    excluded=set(args.exclude_genes.split(',')) if args.exclude_genes else set()
    keep=~te.var_names.isin(excluded)
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=42))
    clf.fit(ref.X[:,keep],ref.obs.Group.to_numpy())
    labels=clf.predict(te.X[:,keep]);truth=te.obs.Group.to_numpy()
    out=ROOT/'results/ipsc'/args.run/'population_audit';out.mkdir(exist_ok=True)
    rows=[]
    for g in range(4):
        x=te.X[truth==g][:,keep];lab=clf.predict(x);prob=clf.predict_proba(x)
        rows.append(dict(kind='real_heldout',mode='real',target=f'day{g}',cells=len(x),
                         assigned_target_fraction=float((lab==g).mean()),target_probability=float(prob[:,g].mean()),
                         **{f'assigned_day{k}':float((lab==k).mean()) for k in range(4)}))
    path=ROOT/'results/ipsc'/args.run/'research_eval'/args.tag
    for p in path.glob('*_day[12]_rep0.npz'):
        parts=p.stem.split('_day');mode=parts[0];g=int(parts[1][0]);pred=np.load(p)['prediction']
        lab=clf.predict(pred[:,keep]);prob=clf.predict_proba(pred[:,keep])
        rows.append(dict(kind='generated',mode=mode,target=f'day{g}',cells=len(pred),
                         assigned_target_fraction=float((lab==g).mean()),target_probability=float(prob[:,g].mean()),
                         **{f'assigned_day{k}':float((lab==k).mean()) for k in range(4)}))
    label=args.tag+('_without_'+args.exclude_genes.replace(',','_') if args.exclude_genes else '')
    pd.DataFrame(rows).to_csv(out/f'{label}.csv',index=False)
    write_json({'classifier':'standardized multinomial logistic regression; C=1',
        'reference_cells_per_day':1200,'reference_evaluation_overlap':0,
        'excluded_genes':sorted(excluded),
        'real_heldout_accuracy':float(accuracy_score(truth,labels)),
        'real_confusion_matrix':confusion_matrix(truth,labels,labels=[0,1,2,3]).tolist(),
        'scope':'Post-hoc diagnostic classifier with reference labels; not a forecasting model or training signal'},
        out/f'{label}_method.json')
    print(pd.DataFrame(rows).to_string(index=False))
    record(f'群体状态诊断 ipsc/{args.run} 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='v1');p.add_argument('--tag',default='raw_controls');p.add_argument('--data-root',default='data/processed');p.add_argument('--exclude-genes',default='');main(p.parse_args())
