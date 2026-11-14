"""Check the transient marker in the full public data, independent of model and cell split."""
import numpy as np
import pandas as pd
import anndata as ad
from common import ROOT, write_json


def main():
    a=ad.read_h5ad(ROOT/'data/cache/ipsc_full.h5ad',backed='r')
    meta=a.obs[['day','donor','Group']].copy()
    meta['T']=np.asarray(a[:,'T'].X).reshape(-1)
    original=a.var.loc['T'].to_dict()
    a.file.close()
    out=ROOT/'results/research/raw_dynamics';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for day, frame in meta.groupby('Group',observed=True):
        v=frame['T'].to_numpy()
        rows.append(dict(day=int(day),cells=len(v),donors=frame.donor.nunique(),mean=float(v.mean()),
            median=float(np.median(v)),p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95)),
            max=float(v.max()),fraction_above_0=float(np.mean(v>0)),
            fraction_above_1=float(np.mean(v>1)),fraction_above_3=float(np.mean(v>3))))
    pd.DataFrame(rows).to_csv(out/'T_all_cells_by_day.csv',index=False)
    donor=meta.groupby(['donor','Group'],observed=True)['T'].agg(['mean','count']).reset_index()
    donor.to_csv(out/'T_by_donor_day.csv',index=False)
    means=donor.pivot(index='donor',columns='Group',values='mean')
    counts=donor.pivot(index='donor',columns='Group',values='count')
    adequate=(counts[[0,1,3]]>=5).all(axis=1)
    excess=means.loc[adequate,1]-means.loc[adequate,[0,3]].max(axis=1)
    summary=dict(gene=original,units='Original public log-normalized values; no rescaling',
        donor_threshold='At least 5 observed cells at each of day0/day1/day3',
        adequate_donors=int(adequate.sum()),donors_with_positive_day1_T_excess=int((excess>0).sum()),
        donor_excess_median=float(excess.median()),donor_excess_min=float(excess.min()),
        donor_excess_max=float(excess.max()),scope='Observed-data robustness only, not an independent donor-held-out model test')
    write_json(summary,out/'summary.json');print(pd.DataFrame(rows).to_string(index=False));print(summary)


if __name__=='__main__':main()
