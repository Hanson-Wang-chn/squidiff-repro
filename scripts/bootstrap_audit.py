"""Audit the public notebook's use of target indices to resample control cells."""
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from sklearn.metrics import r2_score
from common import ROOT,write_json,record

def main(args):
    tr=ad.read_h5ad(ROOT/args.data_root/'norman/train.h5ad')
    te=ad.read_h5ad(ROOT/args.data_root/'norman/test.h5ad')
    c=tr.X[tr.obs.Group==0].astype(np.float64);truth=te.X.astype(np.float64)
    path=ROOT/'results/norman'/args.run/'research_eval'/args.tag
    rows=[]
    cases=[(file.stem,np.load(file)['prediction'].astype(np.float64)) for file in sorted(path.glob('*_AB_rep0.npz'))]
    cases.append(('exact_no_change',None))
    control_mean=c.mean(0)
    group=tr.obs.Group.to_numpy()
    additive=tr.X[group==1].astype(float).mean(0)+tr.X[group==2].astype(float).mean(0)-control_mean
    cases.append(('exact_additive_mean',np.broadcast_to(additive,(600,tr.n_vars))))
    for mode,pred in cases:
        rng=np.random.RandomState(42)
        for rep in range(args.repeats):
            n=len(pred) if pred is not None else 600
            ip=rng.choice(n,n,replace=True);it=rng.choice(len(truth),len(truth),replace=True)
            xp=pred[ip].mean(0) if pred is not None else control_mean
            yt=truth[it].mean(0)
            references={'released_same_indices_first_target_n_controls':c[it].mean(0),
                        'full_control_mean':control_mean,
                        'random_controls_from_entire_pool':c[rng.choice(len(c),len(truth),replace=True)].mean(0)}
            for method,ref in references.items():
                dx=xp-ref;dy=yt-ref
                rows.append(dict(run=args.run,mode=mode,repeat=rep,control_reference=method,
                    opposite_percent=float(np.mean(dx*dy<0)*100),delta_r2_truth_reference=float(r2_score(dy,dx)),
                    delta_r2_pred_reference=float(r2_score(dx,dy))))
    out=ROOT/'results/research/bootstrap_audit'/args.run;out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'bootstrap.csv',index=False)
    summary=df.groupby(['mode','control_reference'])[['opposite_percent','delta_r2_truth_reference']].agg(['mean','std'])
    summary.to_csv(out/'summary.csv');print(summary.to_string())
    write_json({'source':'squidiff_on_genepertb.ipynb cells 20 and 21',
       'issue':'indices2 is drawn over target cells, then reused to index control cells; it only accesses the first len(target) controls',
       'control_cells':len(c),'target_cells':len(truth),'repeats':args.repeats,
       'scope':'Audit of reference construction; bootstrap does not measure independent training or biological replication'},out/'method.json')
    record(f'Norman bootstrap参考审计 {args.run} 完成：比较原notebook只使用控制组前{len(truth)}个细胞与全部{len(c)}控制细胞。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='v1');p.add_argument('--tag',default='inference_controls');p.add_argument('--data-root',default='data/processed');p.add_argument('--repeats',type=int,default=100);main(p.parse_args())
