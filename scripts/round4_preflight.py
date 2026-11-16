"""Meaningful checks for round4 model interventions and the preserved update path."""
import copy
import json
import numpy as np
import torch
from common import ROOT, build_model, write_json
from round4 import create, VARIANTS, ResidualBlock, load_data
from Squidiff.MLPModel import MLPBlock

def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tr,va,te=load_data(dict(panel='source'))
    torch.manual_seed(901)
    m,d=build_model(tr.n_vars)
    x=torch.tensor(tr.X[:64]);noise=torch.randn_like(x)
    t=torch.arange(64,dtype=torch.long)*15
    cfg=dict(VARIANTS['fullsource'],genes=tr.n_vars)
    b,dd=create(cfg);b.load_state_dict(m.state_dict())
    # Native architecture and diffusion must retain the full gradient update.
    opts=[torch.optim.AdamW(q.parameters(),lr=1e-4,weight_decay=0.) for q in [m,b]]
    losses=[]
    for q,df,opt in zip([m,b],[d,dd],opts):
        q.train();opt.zero_grad()
        loss=df.training_losses(q,x,t,noise=noise,
            model_kwargs={'drug_dose':None,'control_feature':None})['loss'].mean()
        loss.backward();opt.step();losses.append(float(loss))
    error=max(float((m.state_dict()[k]-b.state_dict()[k]).abs().max()) for k in m.state_dict())
    assert error==0
    # Structural residual implements h + f(h), with no extra parameters.
    block=MLPBlock(32,32,32,60)
    residual=copy.deepcopy(block);residual.__class__=ResidualBlock
    h=torch.randn(8,32);emb=torch.randn(8,32);z=torch.randn(8,60)
    skip_error=float((residual(h,emb,z)-(h+block(h,emb,z))).abs().max())
    assert skip_error==0
    # LayerNorm is independent of companion cells, also while training.
    ln,_=create(dict(VARIANTS['layernorm'],genes=tr.n_vars));ln.train()
    a=ln.encoder(x[:1]);bb=ln.encoder(x)[:1]
    ln_error=float((a-bb).abs().max());assert ln_error<1e-5
    # All declared models receive finite gradients from the real endpoint task.
    variants=[]
    for name,c in VARIANTS.items():
        q,df=create(dict(c,genes=tr.n_vars));q.train()
        loss=df.training_losses(q,x,t,noise=noise,
            model_kwargs={'drug_dose':None,'control_feature':None})['loss'].mean()
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in q.parameters() if p.grad is not None)
        variants.append(dict(variant=name,loss=float(loss),parameters=sum(p.numel() for p in q.parameters())))
    write_json(dict(native_update_max_error=error,residual_algebra_max_error=skip_error,
        layernorm_companion_max_error=ln_error,variants=variants,
        scope='Engineering checks only, not biological results'),ROOT/'results/round4/preflight.json')
    print('PREFLIGHT_OK',error,ln_error,variants,flush=True)

if __name__=='__main__':main()
