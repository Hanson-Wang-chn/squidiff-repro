"""Compare population-mean and observed-source-conditioned expression baselines."""
import json
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import r2_score
from common import ROOT,write_json


def main():
    cache={};rows=[]
    for path in sorted((ROOT/'results/ipsc').glob('*/research_eval/*/metrics.csv')):
        if not (path.parent/'complete.json').exists():continue
        cfg=json.loads((path.parents[2]/'config.json').read_text())
        root=cfg.get('data_root','data/processed')
        if root not in cache:
            tr=ad.read_h5ad(ROOT/root/'ipsc/train.h5ad');te=ad.read_h5ad(ROOT/root/'ipsc/test.h5ad')
            means={g:tr.X[tr.obs.Group==g].astype(float).mean(0) for g in [0,3]}
            origin=te.X[te.obs.Group==0].astype(float).mean(0)
            cache[root]=(means,origin,{d:te.X[te.obs.Group==d].astype(float).mean(0) for d in [1,2]})
        means,origin,truth=cache[root]
        for _,r in pd.read_csv(path).iterrows():
            if r.target not in ['day1','day2']:continue
            day=int(r.target[-1]);frac=day/3;actual=truth[day]
            population=(1-frac)*means[0]+frac*means[3]
            conditioned=origin+frac*(means[3]-means[0])
            mse_pop=float(np.mean((actual-population)**2));mse_input=float(np.mean((actual-conditioned)**2))
            rows.append(dict(run=cfg['run'],tag=path.parent.name,step=int(r.step),mode=r['mode'],target=r.target,
                data_root=root,population_linear_r2=float(r2_score(actual,population)),
                input_conditioned_linear_r2=float(r2_score(actual,conditioned)),
                population_linear_mse=mse_pop,input_conditioned_linear_mse=mse_input,
                model_mean_mse=float(r.mean_mse),model_gain_vs_population=1-float(r.mean_mse)/mse_pop,
                model_gain_vs_input_conditioned=1-float(r.mean_mse)/mse_input,
                diagnostic_only=r['mode'] in ['oracle','target_centroid'] or 'all_days' in root))
    out=ROOT/'results/research'
    pd.DataFrame(rows).to_csv(out/'input_conditioned_baselines.csv',index=False)
    write_json(dict(population='(1-t)*mean(train day0)+t*mean(train day3)',
        input_conditioned='mean(observed source day0)+t*(mean(train day3)-mean(train day0))',
        purpose='Both use no middle-day expression. The second gives the baseline the same observed day0 source population available to the neural forecast.',
        scope='Mean-expression reference only; no clipping or test-target tuning.'),out/'input_baseline_method.json')
    print(f'{len(rows)} frozen forecasts compared with both input protocols.')


if __name__=='__main__':main()
