"""Select iPSC checkpoint and latent noise using endpoints only, before test access."""
import json
from pathlib import Path
import numpy as np
import torch
import anndata as ad
from scipy.stats import wasserstein_distance
from common import ROOT, build_model, seed_all, dense, atomic_save, write_json, record

@torch.inference_mode()
def choose_ipsc(dest):
    train=ad.read_h5ad(ROOT/'data/processed/ipsc/train.h5ad')
    val=ad.read_h5ad(ROOT/'data/processed/ipsc/val.h5ad')
    assert set(val.obs.Group)=={0,3} and set(train.obs.Group)=={0,3}
    x=torch.tensor(dense(val),device='cuda');tx=torch.tensor(dense(train),device='cuda')
    vg=val.obs.Group.to_numpy();tg=train.obs.Group.to_numpy()
    audit=[];models=[]
    def assess(m,d,z,g,seed=123):
        seed_all(seed)
        pred=d.ddim_sample_loop(m,(len(z),val.n_vars),model_kwargs={'z_mod':z}).cpu().numpy()
        truth=dense(val)[vg==g]
        return dict(mean_mse=float(np.mean((truth.mean(0)-pred.mean(0))**2)),
                    wasserstein=float(np.mean([wasserstein_distance(truth[:,j],pred[:,j]) for j in range(val.n_vars)])),
                    variance_mae=float(np.abs(truth.var(0)-pred.var(0)).mean()))
    seen=set()
    for name in ['best.pt','step_010000.pt','step_020000.pt','step_050000.pt','step_100000.pt']:
        path=dest/name
        if not path.exists():continue
        s=torch.load(path,map_location='cpu',weights_only=False)
        step=s['step']
        if step in seen:continue
        seen.add(step)
        m,d=build_model(val.n_vars,100);m.load_state_dict(s['model']);m.cuda().eval()
        z=m.encoder(x)
        rr=[]
        for g in [0,3]:
            r=assess(m,d,z[torch.as_tensor(vg==g,device='cuda')],g)
            audit.append(dict(stage='checkpoint_screen',checkpoint=name,step=step,day=g,**r));rr.append(r)
        score=float(np.mean([r['mean_mse'] for r in rr]))
        models.append((score,name,step))
        print('checkpoint calibration',step,score,flush=True)
        del m,d,s
    # Confirm the two strongest checkpoints with the full 1000-step sampler.
    confirmed=[]
    for _,name,step in sorted(models)[:2]:
        s=torch.load(dest/name,map_location='cpu',weights_only=False)
        m,d=build_model(val.n_vars,1000);m.load_state_dict(s['model']);m.cuda().eval()
        z=m.encoder(x);rr=[]
        for g in [0,3]:
            r=assess(m,d,z[torch.as_tensor(vg==g,device='cuda')],g)
            audit.append(dict(stage='checkpoint_confirm',checkpoint=name,step=step,day=g,**r));rr.append(r)
        confirmed.append((float(np.mean([r['mean_mse'] for r in rr])),name,step))
        del m,d,s
    _,name,step=min(confirmed)
    s=torch.load(dest/name,map_location='cpu',weights_only=False)
    atomic_save(s,dest/'best_generation.pt')
    m,d=build_model(val.n_vars,1000);m.load_state_dict(s['model']);m.cuda().eval()
    z=m.encoder(tx)
    noise_candidates=[]
    # The semantic scale is not identifiable during epsilon prediction. A fixed
    # absolute latent sigma need not transfer across random seeds or data splits.
    # Select sigma by endpoint distribution Wasserstein distance, never day1/day2.
    for sigma in [0.,.05,.1,.2,.3,.5,.7]:
        scores=[]
        for g in [0,3]:
            seed_all(321)
            center=z[torch.as_tensor(tg==g,device='cuda')].mean(0)
            zz=center+sigma*torch.randn((200,60),device='cuda')
            r=assess(m,d,zz,g)
            audit.append(dict(stage='latent_sigma',checkpoint=name,step=step,day=g,sigma=sigma,**r))
            scores.append(r['wasserstein'])
        noise_candidates.append((float(np.mean(scores)),sigma))
        print('latent noise calibration',sigma,float(np.mean(scores)),flush=True)
    score,sigma=min(noise_candidates)
    selection=dict(checkpoint='best_generation.pt',original_checkpoint=name,step=step,
                   latent_noise=sigma,endpoint_wasserstein=score,
                   selection_data='day0/day3 validation only; no day1/day2 test accessed',audit=audit)
    write_json(selection,dest/'generation_selection.json')
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(audit);frame.to_csv(dest/'generation_validation.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for stage,part in frame[frame.stage!='latent_sigma'].groupby('stage'):
        means=part.groupby('step').mean_mse.mean()
        axs[0].plot(means.index,means.values,'o-',label=stage)
    axs[0].set(xlabel='Training step',ylabel='Endpoint mean-expression MSE',yscale='log');axs[0].legend()
    means=frame[frame.stage=='latent_sigma'].groupby('sigma').wasserstein.mean()
    axs[1].plot(means.index,means.values,'o-');axs[1].axvline(sigma,color='r',ls='--')
    axs[1].set(xlabel='Latent noise sigma',ylabel='Endpoint gene Wasserstein distance')
    fig.tight_layout();fig.savefig(dest/'generation_validation.png',dpi=180);plt.close(fig)
    record(f'iPSC 仅用 day0/day3 验证生成结果选择第 {step} 步检查点，潜变量噪声 sigma={sigma}（原值0.7），端点平均Wasserstein={score:.4f}。原去噪损失选择与所有候选记录保留。')
    return selection
