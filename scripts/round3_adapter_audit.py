"""Analytic regression checks for the changed diffusion-output semantics."""
import copy
import numpy as np
import torch
from common import ROOT, write_json
from Squidiff.script_util import create_gaussian_diffusion
from diffusion_adapter import correct_reverse_target, min_snr_weight


def main():
    torch.set_num_threads(2)
    records=[]
    for count in [100,1000]:
        d=create_gaussian_diffusion(steps=1000,predict_xstart=True,
            timestep_respacing=f'ddim{count}')
        torch.manual_seed(123)
        x0=torch.rand(8,11)*8
        t=torch.tensor([0,1,2,5,10,20,count//2,count-1])
        xt=d.q_sample(x0,t,noise=torch.randn_like(x0))
        seen=[]
        def perfect(x,ts,**kw):
            seen.append(ts.clone())
            return x0
        broken=d.p_mean_variance(perfect,xt,t,clip_denoised=False)['pred_xstart']
        original_error=float((broken-x0).abs().max())
        assert original_error>1.,original_error
        correct_reverse_target(d)
        output=d.p_mean_variance(perfect,xt,t,clip_denoised=False)
        assert torch.equal(output['pred_xstart'],x0)
        assert torch.equal(seen[-1],torch.tensor(d.timestep_map)[t])
        mean,_,_=d.q_posterior_mean_variance(x0,xt,t)
        assert torch.equal(output['mean'],mean)
        # Verify the final sample across the complete deterministic DDIM chain.
        sample=d.ddim_sample_loop(perfect,x0.shape,
            noise=torch.randn_like(x0),device='cpu',clip_denoised=False)
        assert torch.equal(sample,x0)
        records.append(dict(steps=count,original_x0_max_error=original_error,
            corrected_x0_max_error=0.,final_ddim_error=0.,timestep_mapping_correct=True))
    e=create_gaussian_diffusion(steps=1000,timestep_respacing='ddim100')
    torch.manual_seed(321)
    xt=torch.randn(8,11)
    t=torch.full((8,),99,dtype=torch.long)
    def epsilon(x,ts,**kw): return .5*x+.1
    before=e.p_mean_variance(epsilon,xt,t)
    original=e.p_mean_variance
    correct_reverse_target(e)
    assert e.p_mean_variance==original
    after=e.p_mean_variance(epsilon,xt,t)
    assert all(torch.equal(before[k],after[k]) for k in before)
    torch.manual_seed(89)
    noise=torch.randn(8,11)
    original_draw=e.ddim_sample_loop(epsilon,xt.shape,noise=noise,device='cpu',clip_denoised=False)
    untouched=create_gaussian_diffusion(steps=1000,timestep_respacing='ddim100')
    other=untouched.ddim_sample_loop(epsilon,xt.shape,noise=noise,device='cpu',clip_denoised=False)
    assert torch.equal(original_draw,other)
    full=create_gaussian_diffusion(steps=1000)
    ti=torch.tensor([0,25,100,250,500,750,950,999])
    alpha=torch.tensor(full.alphas_cumprod,dtype=torch.float32)[ti]
    snr=alpha/(1-alpha)
    error=torch.linspace(.1,2.,8)
    a=min_snr_weight(full,ti)*snr*error.square()
    b=torch.minimum(snr,torch.full_like(snr,5.))*error.square()
    assert torch.allclose(a,b,rtol=1e-6,atol=1e-6)
    write_json(dict(xstart=records,epsilon_noop_bitwise=True,
        epsilon_ddim_bitwise=True,min_snr_equivalence_max_error=float((a-b).abs().max())),
        ROOT/'results/round3/adapter_audit.json')
    print('ADAPTER_AUDIT_OK',records)


if __name__=='__main__': main()
