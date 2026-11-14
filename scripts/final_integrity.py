"""Read-only consistency check of completed runs, frozen data and prediction archives."""
import hashlib
import json
import time
import numpy as np
import pandas as pd
import anndata as ad
from common import ROOT,write_json


def main():
    jobs=[]
    for i in range(1,6):jobs+=json.loads((ROOT/f'configs/research_stage{i}.json').read_text())
    cache={};checks=[]
    for j in jobs:
        folder=ROOT/'results'/j['dataset']/j['run']
        cfg=json.loads((folder/'config.json').read_text());status=json.loads((folder/'status.json').read_text())
        assert status['status']=='trained' and status['step']==100000,(j,status)
        data=ROOT/cfg.get('data_root','data/processed')/j['dataset']
        if str(data) not in cache:
            te=ad.read_h5ad(data/'test.h5ad')
            cache[str(data)]=(hashlib.sha256((data/'train.h5ad').read_bytes()).hexdigest(),te)
        digest,te=cache[str(data)];assert cfg['train_data_sha256']==digest,j
        ev=folder/'research_eval/raw_controls';assert (ev/'complete.json').exists(),j
        m=pd.read_csv(ev/'metrics.csv');assert (m.step==100000).all(),j
        for path in ev.glob('*.npz'):
            with np.load(path,allow_pickle=False) as arrays:
                assert list(arrays['genes'])==list(te.var_names),path
                assert np.isfinite(arrays['prediction']).all(),path
        for name in ['latest.pt','step_050000.pt','step_100000.pt']:assert (folder/name).exists(),(j,name)
        checks.append(dict(dataset=j['dataset'],run=j['run'],step=100000,genes=cfg['genes'],
            model_variant=cfg.get('model_variant','source'),train_data_sha256_matches=True,
            prediction_gene_order_and_finiteness=True,full_checkpoint_and_milestones_present=True))
    assessment=json.loads((ROOT/'results/research/assessment.json').read_text())
    assert assessment['all_required_models_and_evaluations_complete'] and assessment['completed_research_jobs']==len(jobs)
    frame=pd.read_csv(ROOT/'results/research/all_controlled_metrics.csv')
    assert frame[frame.run=='source1000_fp32_s42'].sampling_stability_failure.all()
    write_json(dict(time=time.strftime('%Y-%m-%d %H:%M:%S Asia/Shanghai'),all_passed=True,new_runs=len(checks),
        runs=checks,failed_wide_model_flagged_and_retained=True,
        note='Generation failure and biological accuracy are separate from these integrity checks.'),ROOT/'results/research/final_integrity.json')
    print(f'PASS {len(checks)} runs: data hashes, gene orders, finite outputs, final steps and checkpoint retention.')


if __name__=='__main__':main()
