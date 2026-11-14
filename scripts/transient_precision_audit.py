"""Derive transient-program precision and recall from frozen per-gene predictions."""
import json
import numpy as np
import pandas as pd
import anndata as ad
from common import ROOT,write_json


def main():
    rows=[];cache={}
    for path in sorted((ROOT/'results/ipsc').glob('*/research_eval/*/gene_means.csv')):
        if not (path.parent/'complete.json').exists():continue
        cfg=json.loads((path.parents[2]/'config.json').read_text())
        root=cfg.get('data_root','data/processed')
        if root not in cache:
            tr=ad.read_h5ad(ROOT/root/'ipsc/train.h5ad')
            mm={k:tr.X[tr.obs.Group==k].astype(float).mean(0) for k in [0,3]}
            cache[root]=(tr.var_names,np.maximum(mm[0],mm[3]))
        genes,envelope=cache[root]
        table=pd.read_csv(path)
        for (mode,target),subset in table.groupby(['mode','target']):
            subset=subset.set_index('gene_id').reindex(genes)
            real=subset.observed_mean.to_numpy()-envelope
            pred=subset.predicted_mean.to_numpy()-envelope
            for threshold in [.25,.5,1.]:
                truth=real>=threshold;called=pred>=threshold
                tp=int(np.sum(truth&called));fp=int(np.sum(~truth&called));fn=int(np.sum(truth&~called))
                precision=tp/(tp+fp) if tp+fp else None
                recall=tp/(tp+fn) if tp+fn else None
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
                rows.append(dict(run=cfg['run'],tag=path.parent.name,checkpoint_step=int(pd.read_csv(path.parent/'metrics.csv').step.iloc[0]),
                    mode=mode,target=target,threshold=threshold,true_transient_genes=int(truth.sum()),
                    predicted_transient_genes=int(called.sum()),true_positives=tp,false_positives=fp,false_negatives=fn,
                    precision=precision,recall=recall,f1=f1,
                    recovered_genes=';'.join(genes[truth&called]),missed_genes=';'.join(genes[truth&~called])))
    out=ROOT/'results/research/transient_programs';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'precision_recall.csv',index=False)
    write_json(dict(definition='A gene is transient when its day1/day2 mean exceeds max(real training day0 mean, real training day3 mean) by the stated threshold.',
        scope='Absolute expression excess, across all selected genes; not just recall on retrospectively chosen positive markers.',
        reference='Real training endpoint means, held fixed for both true and generated transient calls.',
        note='Oracle and all-days training remain diagnostic; source feature discovery itself includes intermediate-day expression.'),out/'method.json')
    print(f'{len(rows)} frozen transient-program evaluations written.')


if __name__=='__main__':main()
