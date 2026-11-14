"""Choose latent noise using held-out endpoints only, before evaluating intermediate days."""
import argparse
import fcntl
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from common import ROOT,setup,seed_all,build_model,write_json,record


@torch.inference_mode()
def select(args):
    setup();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    data=ROOT/args.data_root/'ipsc'
    tr=ad.read_h5ad(data/'train.h5ad');va=ad.read_h5ad(data/'val.h5ad')
    assert set(tr.obs_names).isdisjoint(va.obs_names)
    dest=ROOT/'results/ipsc'/args.run/'noise_validation';dest.mkdir(exist_ok=True)
    state=torch.load(dest.parent/'latest.pt',map_location='cpu',weights_only=False)
    assert state['step']==100000
    m,d=build_model(tr.n_vars,1000);m.load_state_dict(state['model']);m.cuda().eval()
    z=m.encoder(torch.tensor(tr.X,device='cuda'));g=tr.obs.Group.to_numpy()
    groups={i:z[torch.tensor(g==i,device='cuda')] for i in [0,3]}
    means={i:v.mean(0) for i,v in groups.items()}
    rms=float(torch.stack([v.var(0,unbiased=True).mean() for v in groups.values()]).mean().sqrt())
    origin=m.encoder(torch.tensor(va.X[va.obs.Group==0],device='cuda')).mean(0)
    # Pre-specified scale grid: training within-group latent RMS, plus published 0.7.
    sigmas=sorted(set([rms*r for r in [0.,.5,.75,1.,1.25,1.5]]+[.7]))
    rows=[]
    for sigma in sigmas:
        losses=[];mse=[];r2=[]
        for day in [0,3]:
            seed_all(3000)
            latent=origin+day/3*(means[3]-means[0])+sigma*torch.randn((600,60),device='cuda')
            pred=d.ddim_sample_loop(m,(600,tr.n_vars),model_kwargs={'z_mod':latent}).cpu().numpy()
            truth=va.X[va.obs.Group==day]
            losses.append(float(np.mean([wasserstein_distance(truth[:,j],pred[:,j]) for j in range(tr.n_vars)])))
            mse.append(float(np.mean((truth.astype(float).mean(0)-pred.astype(float).mean(0))**2)))
            r2.append(float(r2_score(truth.astype(float).mean(0),pred.astype(float).mean(0))))
        row=dict(sigma=sigma,sigma_over_training_latent_rms=sigma/rms,
                 endpoint_validation_mean_gene_wasserstein=float(np.mean(losses)),
                 endpoint_validation_mean_mse=float(np.mean(mse)),endpoint_validation_mean_r2=float(np.mean(r2)))
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(dest/'candidates.csv',index=False)
    selected=min(rows,key=lambda r:r['endpoint_validation_mean_gene_wasserstein'])
    write_json(dict(args=vars(args),step=state['step'],training_within_group_latent_rms=rms,
        selection_metric='Mean gene-wise Wasserstein across independent day0/day3 validation cells',
        sigma=selected['sigma'],selected=selected,
        middle_day_information_used_for_noise_selection=False,
        scope='Inference adaptation; not the fixed-sigma released protocol',
        candidates=rows),dest/'selection.json')
    record(f'{args.run} 完成仅端点验证的噪声校准：sigma={selected["sigma"]:.4f}；未读取中间日测试数据选参数。')
    return selected['sigma']


def main(args):
    folder=ROOT/'results/ipsc'/args.run/'noise_validation';folder.mkdir(exist_ok=True)
    lock=open(folder/'.calibration.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    path=ROOT/'results/ipsc'/args.run/'noise_validation/selection.json'
    if path.exists():sigma=json.loads(path.read_text())['sigma']
    else:
        sigma=select(args)
        torch.cuda.empty_cache()
    done=ROOT/'results/ipsc'/args.run/'research_eval/endpoint_validated_noise/complete.json'
    if not done.exists():
        subprocess.run([sys.executable,'scripts/research_evaluate.py','ipsc','--run',args.run,
            '--data-root',args.data_root,'--tag','endpoint_validated_noise','--modes','centroid',
            '--sigmas',str(sigma),'--inference-math','ieee'],cwd=ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--data-root',required=True)
    main(p.parse_args())
