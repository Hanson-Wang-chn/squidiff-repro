"""Collect completed evaluations and draw standalone result figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def main():
    frames=[]
    for file in sorted((ROOT/'results').glob('*/*/metrics_1000steps.csv')):
        f=pd.read_csv(file);frames.append(f)
    if frames:
        frame=pd.concat(frames,ignore_index=True)
        frame.to_csv(ROOT/'results/all_metrics.csv',index=False)
        cols=['pearson','r2','delta_pearson','delta_r2','mean_mse','mean_gene_wasserstein','energy_distance']
        summary=frame.groupby(['dataset','run','target','method'])[cols].agg(['mean','std'])
        summary.columns=[f'{metric}_{stat}' for metric,stat in summary.columns]
        summary.to_csv(ROOT/'results/summary.csv')
        print(summary.to_string())
    audits=[]
    for file in sorted((ROOT/'results').glob('*/*/gene_means_*_1000steps.csv')):
        f=pd.read_csv(file)
        for threshold in [0.,.01,.05,.1,.2,.5]:
            subset=f[f.observed_delta.abs()>=threshold]
            audits.append(dict(dataset=file.parents[1].name,run=file.parent.name,target=file.name,
                abs_observed_delta_threshold=threshold,genes=len(subset),
                opposite_direction_fraction=float((subset.observed_delta*subset.predicted_delta<0).mean()),
                mean_delta_mse=float(((subset.observed_delta-subset.predicted_delta)**2).mean())))
    if audits:pd.DataFrame(audits).to_csv(ROOT/'results/direction_audit_first_sample.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,4),squeeze=False)
    for ax,ds in zip(axs.flat,['norman','ipsc']):
        for file in sorted((ROOT/'results'/ds).glob('*/train.jsonl')):
            rows=[json.loads(s) for s in file.read_text().splitlines() if s]
            train=[r for r in rows if 'loss' in r]; val=[r for r in rows if 'val_loss' in r]
            if train:
                x=np.array([r['step'] for r in train]);y=np.array([r['loss'] for r in train])
                window=min(20,len(y));smooth=np.convolve(y,np.ones(window)/window,'valid')
                ax.plot(x[window-1:],smooth,label=f'{file.parent.name} train')
            if val:ax.plot([r['step'] for r in val],[r['val_loss'] for r in val],'o-',ms=3,label=f'{file.parent.name} validation')
        ax.set(title=ds,xlabel='Optimizer steps',ylabel='Noise MSE',yscale='log')
        if ax.get_legend_handles_labels()[0]:ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'results/training_curves.png',dpi=180)

if __name__=='__main__':main()
