"""Post-evaluation day1 controls; never used for selecting the main prediction.

Remove only the mean direction while retaining the selected residual distribution
and identical x_T. Target-RNA oracle diagnoses conditional generation, not forecast.
"""
import json
import anndata as ad
import numpy as np
import pandas as pd
import torch
from common import ROOT,seed_all,write_json
from evaluate import metrics
import round3


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tr=ad.read_h5ad(ROOT/'data/round2/source/train.h5ad')
    te=ad.read_h5ad(ROOT/'data/round2/source/test.h5ad')
    truth=te.X[te.obs.Group==1]
    g=tr.obs.Group.to_numpy()
    envelope=np.maximum(tr.X[g==0].astype(float).mean(0),tr.X[g==3].astype(float).mean(0))
    actual=truth.astype(float).mean(0)-envelope>=.5
    rows=[];replays=[]
    out=ROOT/'results/round3/diagnostics';out.mkdir(exist_ok=True)
    for variant in ['xstart','minsnr','epsres']:
        folder=ROOT/f'results/round3/{variant}_s42'
        state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        cfg=state['config']
        m,d=round3.create(cfg,1000);m.load_state_dict(state['model']);m.cuda().eval()
        z=m.encoder(torch.tensor(tr.X,device='cuda'))
        zz={i:z[torch.tensor(g==i,device='cuda')] for i in [0,3]}
        means={i:zz[i].mean(0) for i in [0,3]}
        delta=means[3]-means[0]
        origin=m.encoder(torch.tensor(te.X[te.obs.Group==0],device='cuda'))
        selected=json.loads((folder/'eval_final_1000/selection.json').read_text())['selected']
        mode,sigma=selected['mode'],selected['sigma'];n=600;alpha=1/3
        seed_all(1000)
        if mode in ['fixed','isotropic']:
            target=origin.mean(0)+alpha*delta+sigma*torch.randn(n,60,device='cuda')
        elif mode=='empirical':
            a=zz[0][torch.randint(len(zz[0]),(n,),device='cuda')]-means[0]
            b=zz[3][torch.randint(len(zz[3]),(n,),device='cuda')]-means[3]
            residual=((1-alpha)*a+alpha*b)/np.sqrt((1-alpha)**2+alpha**2)
            target=origin.mean(0)+alpha*delta+residual
        elif mode=='paired':
            target=origin[torch.arange(n,device='cuda')%len(origin)]+alpha*delta
        elif mode=='ot':
            from scipy.optimize import linear_sum_assignment
            scale0=zz[0].std(0).clamp_min(1e-5);scale3=zz[3].std(0).clamp_min(1e-5)
            r0=(zz[0]-means[0])/scale0;r3=(zz[3]-means[3])/scale3
            ii,jj=linear_sum_assignment(torch.cdist(r0,r3).square().cpu().numpy())
            pairing=torch.tensor(jj[np.argsort(ii)],device='cuda')
            source=origin[torch.arange(n,device='cuda')%len(origin)]
            nearest=torch.cdist((source-means[0])/scale0,r0).argmin(1)
            target=(1-alpha)*source+alpha*zz[3][pairing[nearest]]
        else:raise ValueError(mode)
        noise=torch.randn(n,tr.n_vars,device='cuda')
        oracle=m.encoder(torch.tensor(truth,device='cuda'))
        for label,latent in [('selected_replay',target),('mean_direction_removed',target-alpha*delta),
                             ('target_rna_oracle',oracle)]:
            pred=d.ddim_sample_loop(m,noise.shape,noise=noise,model_kwargs={'z_mod':latent},
                clip_denoised=False).cpu().numpy()
            if label=='selected_replay':
                original=np.load(folder/'eval_final_1000/selected_day1_rep0.npz',allow_pickle=False)['prediction']
                difference=float(np.abs(original-pred).max())
                assert np.allclose(original,pred,rtol=2e-5,atol=2e-5),(variant,difference)
                replays.append(dict(variant=variant,selected_replay_max_error=difference))
            call=pred.astype(float).mean(0)-envelope>=.5
            j=tr.var_names.get_loc('T')
            row=dict(variant=variant,method=label,day=1,T_pred=float(pred[:,j].astype(float).mean()),
                transient_tp=int((call&actual).sum()),transient_fp=int((call&~actual).sum()),
                **metrics(truth,pred,tr.X[g==0]))
            rows.append(row)
            np.savez_compressed(out/f'{variant}_{label}.npz',prediction=pred,
                genes=tr.var_names.to_numpy(dtype=str))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(json.dumps(row),flush=True)
        del m,d,state
    write_json(dict(replays=replays,rows=len(rows),cells=600,sampling_steps=1000,
        main_predictions_changed=False,target_rna_used_only_by_oracle=True,
        control='Same selected residual distribution and xT; remove only alpha*mean_direction'),out/'complete.json')


if __name__=='__main__':main()
