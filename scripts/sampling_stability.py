"""Separate wide-model sampling failure from target-condition inference failure."""
import argparse
import json
import numpy as np
import pandas as pd
import anndata as ad
import torch
from sklearn.metrics import r2_score
from common import ROOT,setup,seed_all,build_model,write_json


@torch.inference_mode()
def main(args):
    setup();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    root=ROOT/args.data_root/'norman';tr=ad.read_h5ad(root/'train.h5ad');te=ad.read_h5ad(root/'test.h5ad')
    dest=ROOT/'results/norman'/args.run/('sampling_stability'+('_'+args.tag if args.tag else ''));dest.mkdir(exist_ok=True)
    examples={'training_control':tr.X[tr.obs.Group==0][:32], 'heldout_AB':te.X[:32]}
    rows=[];noise=[]
    cases=[(args.checkpoint,args.weights)] if args.checkpoint else [('step_050000.pt','model'),('latest.pt','model'),('latest.pt','ema')]
    for checkpoint,weights in cases:
        state=torch.load(dest.parent/checkpoint,map_location='cpu',weights_only=False)
        for nsteps in [int(s) for s in args.sampling_steps.split(',')]:
            m,d=build_model(tr.n_vars,nsteps,model_variant=state.get('config',{}).get('model_variant','source'));m.load_state_dict(state[weights]);m.cuda().eval()
            for label,x in examples.items():
                xx=torch.tensor(x,device='cuda');z=m.encoder(xx)
                seed_all(1000)
                pred=d.ddim_sample_loop(m,(len(x),tr.n_vars),model_kwargs={'z_mod':z}).cpu().numpy()
                row=dict(checkpoint=checkpoint,step=state['step'],weights=weights,sampling_steps=nsteps,
                    input=label,latent_mean_norm=float(z.mean(0).norm()),latent_std=float(z.std(0).square().mean().sqrt()),
                    observed_max=float(x.max()),generated_max=float(pred.max()),
                    generated_p99=float(np.quantile(pred,.99)),generated_mean=float(pred.mean()),
                    mean_r2=float(r2_score(x.astype(float).mean(0),pred.astype(float).mean(0))))
                rows.append(row);print(json.dumps(row),flush=True)
                if nsteps==1000:
                    for timestep in [25,100,500,999]:
                        seed_all(2000)
                        tt=torch.full((len(x),),timestep,device='cuda',dtype=torch.long)
                        loss=d.training_losses(m,xx,tt,model_kwargs={'drug_dose':None,'control_feature':None})['loss'].mean()
                        noise.append(dict(checkpoint=checkpoint,weights=weights,input=label,timestep=timestep,epsilon_mse=float(loss)))
                pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
            del m,d
        del state
    pd.DataFrame(noise).to_csv(dest/'conditional_noise_losses.csv',index=False)
    write_json(dict(args=vars(args),purpose='Only an implementation/stability diagnostic, not an interaction forecast benchmark'),dest/'complete.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='source1000_fp32_s42')
    p.add_argument('--data-root',default='data/research/source_1000_full')
    p.add_argument('--checkpoint');p.add_argument('--weights',choices=['model','ema'],default='model')
    p.add_argument('--sampling-steps',default='100,1000');p.add_argument('--tag',default='')
    main(p.parse_args())
