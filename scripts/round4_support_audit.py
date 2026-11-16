"""Describe exposure to endpoint programs without changing the frozen sampler."""
import json
import numpy as np
import pandas as pd
from common import ROOT,write_json
from round4 import load_data,OUT

def main():
    tr,va,te=load_data({'panel':'source'})
    full,_,_=load_data({'panel':'fullsource'})
    sampling=pd.read_csv(OUT/'support_sampling.csv')
    rows=[]
    for label,a,weights in [('source',tr,np.ones(tr.n_obs)/tr.n_obs),
                            ('supportbalance',tr,sampling.probability.to_numpy()),
                            ('fullsource',full,np.ones(full.n_obs)/full.n_obs)]:
        assert abs(weights.sum()-1)<1e-10
        for day in [0,3]:
            mask=a.obs.Group.to_numpy()==day
            w=weights[mask]/weights[mask].sum();x=np.asarray(a.X)[mask]
            row=dict(pool=label,day=day,rows=len(x),unique_ids=len(set(a.obs_names[mask])),
                effective_rows=float(1/np.sum(w*w)))
            for gene in ['T','DKK1','DKK4','FST','GATA6']:
                xx=x[:,a.var_names.get_loc(gene)]
                row[gene+'_mean']=float(np.sum(w*xx))
                row[gene+'_above3_mass']=float(np.sum(w*(xx>3)))
            idx=[a.var_names.get_loc(g) for g in ['T','DKK1','DKK4']]
            row['all_three_above3_mass']=float(np.sum(w*(x[:,idx]>3).all(1)))
            rows.append(row)
    # Targets are reported only after the intervention and its k were frozen.
    for day in [1,2]:
        x=np.asarray(te.X[te.obs.Group==day]);row=dict(pool='target_diagnostic',day=day,rows=len(x))
        for gene in ['T','DKK1','DKK4','FST','GATA6']:
            xx=x[:,te.var_names.get_loc(gene)];row[gene+'_mean']=float(xx.astype(float).mean())
            row[gene+'_above3_mass']=float((xx>3).mean())
        idx=[te.var_names.get_loc(g) for g in ['T','DKK1','DKK4']]
        row['all_three_above3_mass']=float((x[:,idx]>3).all(1).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'endpoint_support.csv',index=False)
    ids=sampling.groupby('cell_id').probability.sum().to_numpy()
    write_json(dict(rows=rows,supportbalance_effective_unique_cells=float(1/np.sum(ids*ids)),
        clustering='Eight clusters per endpoint; k and weights frozen before this target comparison',
        sampling_scope='Only source training endpoint expression used to fit sampling weights'),OUT/'endpoint_support.json')
    print(pd.DataFrame(rows)[['pool','day','rows','T_mean','T_above3_mass','all_three_above3_mass']].to_string(index=False))

if __name__=='__main__':main()
