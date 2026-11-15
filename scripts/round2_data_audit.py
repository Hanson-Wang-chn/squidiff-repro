"""Source metadata checks; descriptive diagnostics, never model selection."""
import json
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round2'
tr=ad.read_h5ad(ROOT/'data/round2/source/train.h5ad')
te=ad.read_h5ad(ROOT/'data/round2/source/test.h5ad')
markers=['T','DKK1','DKK4','NANOG','GATA6','FST']
support=[];frames=[]
for split,a in [('train',tr),('test',te)]:
    for gene in markers:
        j=a.var_names.get_loc(gene)
        for day in sorted(a.obs.Group.unique()):
            x=a.X[a.obs.Group==day,j]
            support.append(dict(gene=gene,split=split,day=int(day),mean=float(x.mean()),
                maximum=float(x.max()),fraction_above3=float((x>3).mean()),median=float(np.median(x))))
    frame=a.obs[['Group','donor','experiment']].copy()
    frame['cell_id']=a.obs_names
    for gene in markers:frame[gene]=a.X[:,a.var_names.get_loc(gene)]
    frames.append(frame)
pd.DataFrame(support).to_csv(OUT/'marker_support.csv',index=False)
f=pd.concat(frames).drop_duplicates('cell_id')
rows=[]
for stratifier,minimum in [('experiment',5),('donor',2)]:
    for name,g in f.groupby(stratifier,observed=True):
        counts=g.Group.value_counts()
        if not all(counts.get(d,0)>=minimum for d in [0,1,3]):continue
        for gene in ['T','DKK1','DKK4']:
            means=g.groupby('Group')[gene].mean()
            rows.append(dict(stratifier=stratifier,stratum=name,gene=gene,
                n0=int(counts[0]),n1=int(counts[1]),n3=int(counts[3]),
                day0=float(means[0]),day1=float(means[1]),day3=float(means[3]),
                day1_excess=float(means[1]-max(means[0],means[3]))))
f=pd.DataFrame(rows);f.to_csv(OUT/'marker_stratified.csv',index=False)
summary=[]
for (s,g),x in f.groupby(['stratifier','gene']):
    summary.append(dict(stratifier=s,gene=g,strata=len(x),positive_excess=int((x.day1_excess>.5).sum()),
                        median_excess=float(x.day1_excess.median())))
(OUT/'marker_stratified_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

# An unsuccessful provenance check, retained for transparency. Salmon abundance
# can be fractional, so non-integer recovery cannot establish a logarithm base.
a=np.asarray(tr.X,dtype=float);sf=tr.obs.size_factor.to_numpy(dtype=float)
integer_rows=[]
for base in ['log2','ln']:
    values=np.expm1(a*(np.log(2) if base=='log2' else 1))
    for mode,factor in [('size_factor',sf),('inverse_size_factor',1/sf),('unit',np.ones(len(sf)))]:
        v=(values*factor[:,None])[(a>0)&(a<5)]
        error=np.abs(v-np.round(v))
        integer_rows.append(dict(base=base,multiplier=mode,entries=len(v),
            median_integer_error=float(np.median(error)),fraction_within_1e_4=float((error<1e-4).mean()),
            fraction_within_1e_3=float((error<1e-3).mean())))
pd.DataFrame(integer_rows).to_csv(OUT/'log_base_integer_audit.csv',index=False)
