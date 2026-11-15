"""Recompute author predictions, compare simple baselines, render evidence."""
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from common import ROOT,dense,write_json
from evaluate import metrics


def main():
    dest=ROOT/'results/round3/author/eval_1000'
    a=ad.read_h5ad('/home/whs/squidiff-ckpt/VO_trained_adata.h5ad')
    x=dense(a);g=a.obs.Major_type.to_numpy().astype(int)
    f=pd.read_csv(dest/'metrics.csv')
    rows=[];verified=[]
    for _,r in f.iterrows():
        p=dest/f'{r["mode"]}_group{r.group}_rep{r["repeat"]}.npz'
        b=np.load(p,allow_pickle=False)
        assert np.array_equal(b['genes'],a.var_names)
        truth=dense(a[b['truth_ids']]);pred=b['prediction']
        score=metrics(truth,pred)
        assert abs(score['r2']-r.r2)<1e-10
        assert np.isfinite(pred).all()
        verified.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if r['mode']=='encoded':
            mu=x[g==r.group].mean(0)
            baseline=np.repeat(mu[None,:],len(pred),axis=0)
            rng=np.random.default_rng(999)
            random_cells=x[g==r.group][rng.choice(np.sum(g==r.group),len(pred),replace=True)]
            for name,pp in [('encoded',pred),('class_mean',baseline),('same_class_resample',random_cells),
                            ('encoded_rows_permuted',pred[rng.permutation(len(pred))])]:
                m=metrics(truth,pp)
                row=dict(group=r.group,mode=name,**m,paired_mse=float(np.mean((truth-pp)**2)),
                    paired_flat_r2=float(r2_score(truth.flatten(),pp.flatten())),
                    variance_ratio=float(pp.var(0).sum()/truth.var(0).sum()))
                rows.append(row)
    b=pd.DataFrame(rows)
    b.to_csv(dest/'baseline_metrics.csv',index=False)
    write_json(dict(predictions=verified,count=len(verified),all_means_recomputed=True),dest/'integrity.json')
    names=['iPSC','EC','Fibroblast','Mural']
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for mode,label in [('encoded','Encoded own cells'),('center','Latent center'),
                       ('gaussian_0.2','Gaussian sigma=0.2'),('wrong_class','Wrong cell class')]:
        q=f[f['mode']==mode].sort_values('group')
        axs[0].plot(range(4),q.r2,'o-',label=label)
        axs[1].plot(range(4),q.energy_distance,'o-',label=label)
    axs[0].set_ylabel('Across-gene mean R²')
    axs[1].set_ylabel('Joint-expression energy (lower is better)')
    for mode,label in [('encoded','Encoded own cells'),('class_mean','Class expression mean'),
                       ('encoded_rows_permuted','Permuted reconstructed cells')]:
        q=b[b['mode']==mode].sort_values('group')
        axs[2].plot(range(4),q.paired_mse,'o-',label=label)
    axs[2].set_ylabel('Paired-cell expression MSE')
    for ax in axs:
        ax.set_xticks(range(4),names,rotation=20)
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Author BVO checkpoint: training-endpoint diagnostics, not temporal prediction',fontsize=11)
    fig.tight_layout();fig.savefig(dest/'author_controls.png',dpi=180);plt.close(fig)
    curves=pd.read_csv(dest/'descriptive_curves.csv')
    fig,axs=plt.subplots(1,3,figsize=(10,3.2))
    for ax,gene in zip(axs,['NANOG','CLDN5','LUM']):
        j=a.var_names.get_loc(gene)
        for group,label in [(1,'EC'),(2,'Fibroblast'),(3,'Mural')]:
            q=curves[(curves.group==group)&(curves.gene==gene)].sort_values('alpha')
            line=ax.plot(q.alpha,q['mean'],'o-',label=label)[0]
            ax.scatter([0,1],[x[g==0,j].mean(),x[g==group,j].mean()],
                marker='x',s=45,color=line.get_color())
        ax.set_title(gene);ax.set_xlabel('Latent interpolation coefficient')
        ax.set_ylabel('Expression mean');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('Author BVO model: descriptive curves; crosses = measured training endpoints',fontsize=10)
    fig.tight_layout();fig.savefig(dest/'author_curves.png',dpi=180);plt.close(fig)
    print(f[['group','mode','pearson','r2','energy_distance']].to_string(index=False))
    print(b[['group','mode','r2','paired_mse','variance_ratio']].to_string(index=False))


if __name__=='__main__':main()
