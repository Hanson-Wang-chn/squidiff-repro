"""Summarize all fixed final models on the same source panel and test cells."""
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT


def main():
    out=ROOT/'results/round3'
    configs={'native':ROOT/'results/round2/native_s42/eval_final_1000'}
    configs.update({v:out/f'{v}_s42/eval_final_1000' for v in ['xstart','minsnr','epsres']})
    frames=[];markers=[];decomposition=[]
    for v,folder in configs.items():
        f=pd.read_csv(folder/'metrics.csv');f=f[(f.method=='selected')&f.day.isin([1,2])]
        f['version']=v;frames.append(f)
        g=pd.read_csv(folder/'gene_means.csv');g=g[(g.method=='selected')&g.day.isin([1,2])]
        g['version']=v;markers.append(g[g.gene.isin(['T','DKK1','DKK4','FST','NANOG','GATA6'])])
        for day in [1,2]:
            q=g[g.day==day];e=(q.observed-q.predicted)**2
            transient=q.observed-q.endpoint_max>=.5
            top=q.gene.isin(['T','DKK1','DKK4'])
            decomposition.append(dict(version=v,day=day,
                true_transient_count=int(transient.sum()),
                transient_error_share=float(e[transient].sum()/e.sum()),
                T_DKK_error_share=float(e[top].sum()/e.sum())))
    f=pd.concat(frames,ignore_index=True);g=pd.concat(markers,ignore_index=True)
    f.to_csv(out/'selected_metrics.csv',index=False)
    g.to_csv(out/'marker_means.csv',index=False)
    pd.DataFrame(decomposition).to_csv(out/'error_decomposition.csv',index=False)
    variants=list(configs);colors=['#555555','#4c78a8','#f58518','#54a24b']
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for day,offset in [(1,-.18),(2,.18)]:
        q=f[f.day==day].set_index('version').loc[variants]
        axs[0,0].bar(np.arange(4)+offset,100*q.mse_gain,.34,label=f'day {day}')
        axs[1,1].bar(np.arange(4)+offset,q.energy_distance,.34,label=f'day {day}')
    axs[0,0].axhline(0,color='black',lw=.8)
    axs[0,0].set_ylabel('Mean MSE gain over linear baseline (%)')
    names=['T','DKK1','DKK4']
    for i,v in enumerate(['observed']+variants):
        vals=[]
        for gene in names:
            q=g[(g.version==('native' if v=='observed' else v))&(g.day==1)&(g.gene==gene)].iloc[0]
            vals.append(q.observed if v=='observed' else q.predicted)
        axs[0,1].bar(np.arange(3)+(i-2)*.16,vals,.15,label=v)
    axs[0,1].set_xticks(np.arange(3),names)
    axs[0,1].set_ylabel('Day1 expression mean (CSV units)')
    q=f[f.day==1].set_index('version').loc[variants]
    axs[1,0].bar(np.arange(4)-.18,q.transient_tp,.34,label='True positives')
    axs[1,0].bar(np.arange(4)+.18,q.transient_called-q.transient_tp,.34,label='False positives')
    axs[1,0].axhline(13,color='black',linestyle=':',label='13 actual day1 transient genes')
    axs[1,0].set_ylabel('Day1 transient upregulation calls')
    axs[1,1].set_ylabel('Joint-expression energy (lower is better)')
    for ax in [axs[0,0],axs[1,0],axs[1,1]]:
        ax.set_xticks(np.arange(4),variants)
    for ax in axs.flat:
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed final checkpoints, endpoint-selected sampling, identical 203 genes',fontsize=12)
    fig.tight_layout();fig.savefig(out/'controlled_comparison.png',dpi=180);plt.close(fig)
    print(f[['version','day','r2','mse_gain','T_pred','transient_tp','transient_called']].to_string(index=False))


if __name__=='__main__':main()
