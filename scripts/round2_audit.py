"""Numerical checks for round2's observation and standardized sampling changes."""
import copy
import json
import time
import torch
import numpy as np
import round2 as r
from torch.utils.data import DataLoader

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
config=dict(r.VARIANTS['native'],genes=203)
tr,va,te=r.load_data(config)
r.logger.configure(str(r.OUT/'numerical_audit_logs'),format_strs=[])

def path(with_validation):
    r.seed_all(4242)
    m,d=r.create(config);m.cuda().train()
    loop=r.TrainLoop(model=m,diffusion=d,data=None,batch_size=64,microbatch=-1,
        lr=1e-4,ema_rate='0.9999',log_interval=1000,save_interval=10000,
        resume_checkpoint='',lr_anneal_steps=100000)
    loader=DataLoader(r.Cells(tr,config),batch_size=64,shuffle=True)
    for i in range(5):
        loop.step=i
        loop.run_step(next(iter(loader)))
        if with_validation and i==1:
            r.validation(m,d,va,config)
            loop.loss_list.clear()
    result={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    state=r.rng()
    del loop,m,d
    torch.cuda.empty_cache()
    return result,state

a,ar=path(False);b,br=path(True)
err=max(float((a[k]-b[k]).abs().max()) for k in a)
assert err==0,err
assert torch.equal(ar['cpu'],br['cpu'])
assert all(torch.equal(x,y) for x,y in zip(ar['cuda'],br['cuda']))
assert np.array_equal(ar['numpy'][1],br['numpy'][1])

r.seed_all(111)
m,d=r.create(config,100);m.cuda().eval()
with torch.inference_mode():
    z=m.encoder(torch.tensor(tr.X[:8],device='cuda'))
    noise=torch.randn(8,203,device='cuda')
    p=d.ddim_sample_loop(m,(8,203),noise=noise.clone(),model_kwargs={'z_mod':z})
    # With center=0, scale=1, the corrected physical floor reduces exactly
    # to the original floor. Check every generated value with the same noise.
    r.signed_sampling(d,dict(center=[0.]*203,scale=[1.]*203))
    q=d.ddim_sample_loop(m,(8,203),noise=noise.clone(),model_kwargs={'z_mod':z})
    signed_err=float((p-q).abs().max())
    assert signed_err<1e-5,signed_err
def resumed_path(resume):
    r.seed_all(7878)
    cfg=dict(r.VARIANTS['compact'],genes=203)
    m,d=r.create(cfg);m.cuda().train()
    def make_loop(m,d):
        out=r.TrainLoop(model=m,diffusion=d,data=None,batch_size=64,microbatch=-1,
            lr=1e-4,ema_rate='0.9999',log_interval=1000,save_interval=10000,
            resume_checkpoint='',lr_anneal_steps=100000,weight_decay=cfg['wd'])
        return out
    loop=make_loop(m,d)
    loader=DataLoader(r.Cells(tr,cfg),batch_size=64,shuffle=True)
    for i in range(5):
        if resume and i==2:
            state=copy.deepcopy(dict(model=m.state_dict(),optimizer=loop.opt.state_dict(),
                ema=loop.mp_trainer.master_params_to_state_dict(loop.ema_params[0]),rng=r.rng()))
            m,d=r.create(cfg);m.cuda().train();loop=make_loop(m,d)
            m.load_state_dict(state['model']);loop.opt.load_state_dict(state['optimizer'])
            loop.ema_params=[[v.to('cuda') for v in loop.mp_trainer.state_dict_to_master_params(state['ema'])]]
            r.restore(state['rng'])
        loop.step=i;loop.run_step(next(iter(loader)))
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

ra=resumed_path(False);rb=resumed_path(True)
resume_err=max(float((ra[k]-rb[k]).abs().max()) for k in ra)
assert resume_err==0,resume_err
r.write_json(dict(validation_observer_parameter_max_error=err,
    native_optimizer_rng_resume_parameter_max_error=resume_err,
    cpu_cuda_numpy_rng_restored=True,standardized_floor_reduces_to_source_max_error=signed_err,
    full_source_run_loop_preserved=r.ObservedLoop.run_loop is r.TrainLoop.run_loop,
    full_source_run_step_preserved=r.ObservedLoop.run_step is r.TrainLoop.run_step,
    full_source_forward_backward_preserved=r.ObservedLoop.forward_backward is r.TrainLoop.forward_backward),
    r.OUT/'numerical_audit.json')
print((r.OUT/'numerical_audit.json').read_text())
