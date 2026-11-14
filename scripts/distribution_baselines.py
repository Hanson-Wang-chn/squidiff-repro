"""Parameter-free cell-distribution baselines, using training expression only."""
import argparse
import hashlib
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import r2_score
from common import ROOT, write_json
from evaluate import metrics


def main(args):
    data=ROOT/args.data_root/args.dataset
    tr=ad.read_h5ad(data/'train.h5ad');te=ad.read_h5ad(data/'test.h5ad')
    g=tr.obs.Group.to_numpy();groups={k:tr.X[g==k].astype(float) for k in np.unique(g)}
    means={k:v.mean(0) for k,v in groups.items()}
    out=ROOT/'results/research/distribution_baselines'/data.parent.name/args.dataset
    out.mkdir(parents=True,exist_ok=True);rows=[]
    scopes={'all_genes':np.arange(tr.n_vars)}
    if args.dataset=='norman' and tr.n_vars>203:
        ref=ad.read_h5ad(ROOT/'data/research/source_full/norman/test.h5ad')
        if set(ref.var_names)<=set(tr.var_names):scopes['shared_source203']=tr.var_names.get_indexer(ref.var_names)
    for rep in range(args.repeats):
        rng=np.random.default_rng(1000+rep)
        for target in (['AB'] if args.dataset=='norman' else ['day1','day2']):
            x0=groups[0][rng.choice(len(groups[0]),args.cells,replace=True)]
            if args.dataset=='norman':
                truth=te.X.astype(float);base=means[1]+means[2]-means[0]
                linear=x0+means[1]+means[2]-2*means[0]
                methods={'no_change_cells':x0,'additive_cells_unclipped':linear,
                         'additive_cells_clipped_zero':np.maximum(linear,0.)}
            else:
                day=int(target[-1]);truth=te.X[te.obs.Group==day].astype(float)
                x3=groups[3][rng.choice(len(groups[3]),args.cells,replace=True)]
                base=(1-day/3)*means[0]+day/3*means[3]
                methods={'no_change_day0_cells':x0,
                         'expression_linear_independent_cells':(1-day/3)*x0+day/3*x3}
            for method,pred in methods.items():
                for scope,idx in scopes.items():
                    row=dict(dataset=args.dataset,data_root=args.data_root,method=method,target=target,
                             repeat=rep,evaluation_scope=scope,**metrics(truth[:,idx],pred[:,idx],groups[0][:,idx]))
                    observed=truth[:,idx].mean(0);baseline=base[idx]
                    row.update(exact_linear_r2=float(r2_score(observed,baseline)),
                        exact_linear_mse=float(np.mean((observed-baseline)**2)))
                    if args.dataset=='norman':row['interaction_r2']=float(r2_score(observed-baseline,pred[:,idx].mean(0)-baseline))
                    else:
                        j=tr.var_names.get_loc('T');row['predicted_T_mean']=float(pred[:,j].mean())
                    rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    write_json(dict(args=vars(args),train_sha256=hashlib.sha256((data/'train.h5ad').read_bytes()).hexdigest(),
                    protocol='Training-only empirical cell draws. iPSC endpoints independently paired; Norman control cells shifted by full-training group means. No test information used to form predictions.',
                    caveat='Exact linear mean metrics are primary for mean prediction; finite sampled-cell metrics additionally assess distributions. Clipping changes the additive mean and is labelled separately.'),out/'method.json')
    print(frame.groupby(['method','target','evaluation_scope'])[['r2','mean_gene_wasserstein','energy_distance']].mean().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['ipsc','norman'])
    p.add_argument('--data-root',default='data/processed');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cells',type=int,default=600);main(p.parse_args())
