"""Conditional cell-sampling uncertainty of the non-additive residual (not biological replication)."""
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import norm,pearsonr
from sklearn.metrics import r2_score
from common import ROOT,write_json,record

def main(args):
    tr=ad.read_h5ad(ROOT/args.data_root/'norman/train.h5ad');te=ad.read_h5ad(ROOT/args.data_root/'norman/test.h5ad')
    g=tr.obs.Group.to_numpy();xs={i:tr.X[g==i].astype(np.float64) for i in [0,1,2]};xs[3]=te.X.astype(np.float64)
    means={k:v.mean(0) for k,v in xs.items()};signs={0:1,1:-1,2:-1,3:1}
    interaction=sum(signs[k]*means[k] for k in xs)
    se=np.sqrt(sum(v.var(0,ddof=1)/len(v) for v in xs.values()))
    z=interaction/np.maximum(se,1e-12);p=2*norm.sf(np.abs(z));order=np.argsort(p)
    q=np.empty_like(p);q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];q=np.minimum(q,1.)
    rng=np.random.default_rng(12345);rows=[]
    for rep in range(args.repeats):
        halves=[np.zeros(tr.n_vars),np.zeros(tr.n_vars)]
        for k,v in xs.items():
            ii=rng.permutation(len(v));n=len(ii)//2
            halves[0]+=signs[k]*v[ii[:n]].mean(0);halves[1]+=signs[k]*v[ii[n:]].mean(0)
        rows.append({'repeat':rep,'split_half_pearson':float(pearsonr(*halves).statistic),
                     'split_half_r2':float(r2_score(*halves))})
    dest=ROOT/'results/research/interaction_uncertainty'/args.run;dest.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame({'gene_id':tr.var_names,'gene_name':tr.var.gene_name.to_numpy(),
        'true_interaction':interaction,'cell_sampling_se':se,'z_approx':z,'fdr_bh':q})
    ev=ROOT/'results/norman'/args.run/'research_eval'/args.tag
    stats={}
    for path in ev.glob('*_AB_rep0.npz'):
        pred=np.load(path)['prediction'].astype(np.float64).mean(0)
        pi=pred-means[1]-means[2]+means[0]
        frame[path.stem+'_interaction']=pi
        mask=q<.05
        stats[path.stem]={'interaction_r2_all_genes':float(r2_score(interaction,pi)),
                         'interaction_pearson_all_genes':float(pearsonr(interaction,pi).statistic),
                         'fdr05_genes':int(mask.sum()),
                         'interaction_r2_fdr05':float(r2_score(interaction[mask],pi[mask])) if mask.sum()>1 else None}
    frame.to_csv(dest/'genes.csv',index=False);pd.DataFrame(rows).to_csv(dest/'split_halves.csv',index=False)
    out={'conditional_on_these_cells':True,'not_biological_replicates':True,'counts':{str(k):len(v) for k,v in xs.items()},
         'split_half_pearson_median':float(np.median([r['split_half_pearson'] for r in rows])),
         'split_half_pearson_5_95':np.quantile([r['split_half_pearson'] for r in rows],[.05,.95]).tolist(),
         'fdr05_interaction_genes':int((q<.05).sum()),'models':stats}
    write_json(out,dest/'summary.json');print(out)
    record(f'非加性交互信号审计 {args.run} 完成：四组分别独立拆半，评估残差信号的细胞抽样重复性；不替代生物学重复。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='v1');p.add_argument('--tag',default='inference_controls');p.add_argument('--data-root',default='data/processed');p.add_argument('--repeats',type=int,default=100);main(p.parse_args())
