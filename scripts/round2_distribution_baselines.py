"""Endpoint-only population baselines in original expression coordinates."""
import numpy as np
import pandas as pd
import round2 as r

rows=[]
for panel in ['source','log2hvg']:
    cfg=dict(panel=panel)
    tr,va,te=r.load_data(cfg)
    x={d:tr.X[tr.obs.Group==d] for d in [0,3]}
    for rep in range(3):
        rng=np.random.RandomState(6000+rep)
        p0=x[0][rng.choice(len(x[0]),600)]
        p3=x[3][rng.choice(len(x[3]),600)]
        for day in [1,2]:
            alpha=day/3
            truth=te.X[te.obs.Group==day]
            for method in ['cell_linear','endpoint_mixture']:
                if method=='cell_linear':pred=(1-alpha)*p0+alpha*p3
                else:
                    switch=rng.rand(600)<alpha
                    pred=np.where(switch[:,None],p3,p0)
                row=dict(panel=panel,method=method,day=day,repeat=rep,
                         **r.metrics(truth,pred,x[0]))
                rows.append(row)
pd.DataFrame(rows).to_csv(r.OUT/'distribution_baselines.csv',index=False)
print(pd.DataFrame(rows).groupby(['panel','method','day'])[
    ['r2','mean_gene_wasserstein','energy_distance']].mean().to_string())
