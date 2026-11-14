"""Numerical checks against the unmodified public implementation, on the real model."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
import anndata as ad
from common import ROOT, setup, seed_all, build_model, write_json, atomic_save
from train import rng_state, restore_rng
from Squidiff.train_util import TrainLoop
from Squidiff import logger

def max_state_difference(a,b):
    return max(float((a[k].float()-b[k].float()).abs().max()) for k in a)

def main():
    setup();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    out={};seed_all(123)
    a=ad.read_h5ad(ROOT/'data/processed/ipsc/train.h5ad')
    x=torch.tensor(a.X[:64],device='cuda')
    m,d=build_model(a.n_vars);m.cuda().train();other=copy.deepcopy(m)
    fixed_t=torch.arange(64,device='cuda')*15
    class FixedSampler:
        def sample(self,n,device):return fixed_t[:n],torch.ones(n,device=device)
    logger.configure(dir=str(ROOT/'logs/native_equivalence'))
    native=TrainLoop(model=m,diffusion=d,data=[],batch_size=64,microbatch=-1,lr=1e-4,
        ema_rate='0.9999',log_interval=100,save_interval=100,resume_checkpoint='',
        lr_anneal_steps=100000,schedule_sampler=FixedSampler())
    opt=torch.optim.AdamW(other.parameters(),lr=1e-4,weight_decay=0.,fused=True)
    rows=[]
    for i in range(5):
        seed_all(100+i);native.step=i;native.run_step({'feature':x,'group':torch.zeros(64)})
        native_loss=float(native.loss_list[-1].detach())
        seed_all(100+i);opt.zero_grad(set_to_none=True)
        loss=d.training_losses(other,x,fixed_t,model_kwargs={'drug_dose':None,'control_feature':None})['loss'].mean()
        loss.backward();opt.step()
        for pg in opt.param_groups:pg['lr']=1e-4*(1-i/100000)
        rows.append(dict(step=i+1,native_loss=native_loss,local_loss=float(loss),
                         max_parameter_and_buffer_error=max_state_difference(m.state_dict(),other.state_dict())))
    out['native_vs_local_optimizer']=rows
    assert rows[-1]['max_parameter_and_buffer_error']<1e-4,rows
    # Analytic forward/reverse identities for every noise level, independent of the network.
    t=torch.tensor([0,25,100,500,999],device='cuda');xx=x[:5];eps=torch.randn_like(xx)
    q=d.q_sample(xx,t,noise=eps)
    abar=torch.tensor(d.alphas_cumprod,device='cuda',dtype=torch.float32)[t,None]
    qref=abar.sqrt()*xx+(1-abar).sqrt()*eps
    recovered=d._predict_xstart_from_eps(q,t,eps)
    out['diffusion_algebra']={'q_sample_max_error':float((q-qref).abs().max()),
                              'xstart_roundtrip_max_error':float((xx-recovered).abs().max()),
                              'time_map_identity':d.timestep_map==list(range(1000)),
                              'beta_start':float(d.betas[0]),'beta_end':float(d.betas[-1])}
    assert out['diffusion_algebra']['q_sample_max_error']<1e-4
    assert out['diffusion_algebra']['xstart_roundtrip_max_error']<1e-3
    # Official sampler.pred and local sampling must be identical given the same noise stream.
    sys.path.insert(0,str(ROOT/'upstream/Squidiff'))
    from sample_squidiff import sampler
    from evaluate import sample
    m.eval()
    with torch.inference_mode():
        z=m.encoder(x[:16]);s=sampler.__new__(sampler);s.model=m;s.sample_fn=d.ddim_sample_loop
        seed_all(999);official=s.pred(z,a.n_vars)
        seed_all(999);local=sample(m,d,z)
        err=float(np.max(np.abs(official.cpu().numpy()-local)))
    out['official_vs_local_sampler_max_error']=err;assert err==0.
    # Resume roundtrip must preserve the next updates, including random timesteps/noise.
    del native,other,opt,s
    m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.,fused=True)
    def update(model,optimizer):
        optimizer.zero_grad(set_to_none=True);tt=torch.randint(0,1000,(64,),device='cuda')
        loss=d.training_losses(model,x,tt,model_kwargs={'drug_dose':None,'control_feature':None})['loss'].mean()
        loss.backward();optimizer.step();return float(loss)
    for _ in range(3):update(m,opt)
    checkpoint=ROOT/'logs/native_equivalence/resume_test.pt'
    atomic_save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state()),checkpoint)
    expected=[update(m,opt) for _ in range(3)]
    actual=copy.deepcopy(m);actual_opt=torch.optim.AdamW(actual.parameters(),lr=1e-4,weight_decay=0.,fused=True)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    actual.load_state_dict(saved['model']);actual_opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    observed=[update(actual,actual_opt) for _ in range(3)]
    err=max_state_difference(m.state_dict(),actual.state_dict())
    out['resume']={'uninterrupted_losses':expected,'resumed_losses':observed,'max_state_error':err}
    assert err==0. and expected==observed,out['resume']
    checkpoint.unlink()
    out['checks_passed']=True
    write_json(out,ROOT/'results/research/implementation_audit.json')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
