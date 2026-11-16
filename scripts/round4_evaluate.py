"""Full-chain evaluation with endpoint-only selection and common random inputs.

The random input banks are fixed once per validation/test role. No training-seed
or sampling-seed sweep, and no use of day1/2 scores to select any configuration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score

from common import ROOT, write_json
from evaluate import metrics
from round4 import OUT, VARIANTS, create, load_data, folder, sha
from round4_bn_audit import recalibrate

CONTROLS=['native','bncal','ema','emacal','finalclip','paper_cellnoise']


def omit_step_floor(d):
    """Original epsilon reverse equations, omitting only the per-step x0 floor."""
    from Squidiff.diffusion import _extract_into_tensor, ModelMeanType
    assert d.model_mean_type==ModelMeanType.EPSILON
    def p_mean_variance(model,x,t,clip_denoised=False,denoised_fn=None,model_kwargs=None):
        wrapped=d._wrap_model(model) if hasattr(d,'_wrap_model') else model
        eps=wrapped(x,d._scale_timesteps(t),**(model_kwargs or {}))
        x0=d._predict_xstart_from_eps(x,t,eps)
        if denoised_fn is not None:x0=denoised_fn(x0)
        if clip_denoised:x0=x0.clamp(-1,1)
        mean,_,_=d.q_posterior_mean_variance(x0,x,t)
        var=np.append(d.posterior_variance[1],d.betas[1:])
        return dict(mean=mean,pred_xstart=x0,variance=_extract_into_tensor(var,t,x.shape),
            log_variance=_extract_into_tensor(np.log(var),t,x.shape))
    d.p_mean_variance=p_mean_variance
    return d


def bank(n,genes,seed):
    gen=torch.Generator(device='cuda');gen.manual_seed(seed)
    return dict(eta=torch.randn(n,60,device='cuda',generator=gen),
        noise=torch.randn(n,genes,device='cuda',generator=gen),
        u0=torch.rand(n,device='cuda',generator=gen),
        u3=torch.rand(n,device='cuda',generator=gen),
        mix=torch.rand(n,device='cuda',generator=gen))


def scores(truth,pred,tr,reference):
    result=metrics(truth,pred,tr.X[tr.obs.Group==0])
    y=truth.astype(float).mean(0);p=pred.astype(float).mean(0)
    env=np.maximum(*[reference.X[reference.obs.Group==g].astype(float).mean(0) for g in [0,3]])
    actual=y-env>=.5;called=p-env>=.5
    j=tr.var_names.get_loc('T')
    result.update(T_true=float(y[j]),T_pred=float(p[j]),T_above3=float((pred[:,j]>3).mean()),
        transient_actual=int(actual.sum()),transient_called=int(called.sum()),
        transient_tp=int((actual&called).sum()),transient_fp=int((~actual&called).sum()),
        transient_half_recall=float(np.mean((p-env)[actual]>=.5*(y-env)[actual])) if actual.any() else None,
        predicted_max=float(pred.max()),predicted_min=float(pred.min()))
    return result


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    name=args.model
    if name in CONTROLS:
        native=ROOT/'results/round2/native_s42'
        cfg=json.loads((native/'config.json').read_text())
        cfg.update(encoder_norm='batch',block_skip=False)
        path=native/('latest.pt' if name in ['ema','emacal'] else 'final.pt')
    else:
        dest=OUT/f'{name}_s42'
        cfg=json.loads((dest/'config.json').read_text())
        path=dest/'final.pt'
        assert json.loads((dest/'status.json').read_text())['status']=='trained'
    tr,va,te=load_data(cfg)
    reference,_,_=load_data(dict(panel='source'))
    assert tr.var_names.equals(reference.var_names) and tr.var_names.equals(te.var_names)
    assert sha(folder(cfg)/'train.h5ad')==cfg['train_data_sha256']
    for file,h in cfg['source_sha256'].items():assert sha(ROOT/'upstream/Squidiff/Squidiff'/file)==h
    state=torch.load(path,map_location='cpu',weights_only=False)
    m,d=create(cfg,args.sample_steps)
    m.load_state_dict(state['ema'] if name in ['ema','emacal'] else state['model'])
    step=state['step'];del state
    m.cuda().eval()
    x=torch.tensor(np.asarray(tr.X),device='cuda')
    if name in ['bncal','emacal']:recalibrate(m,x)
    if name=='finalclip':omit_step_floor(d)
    output_root=Path(args.output_root) if args.output_root else OUT
    dest=output_root/f'{name}_s42' if name in VARIANTS else output_root/'inference_controls'/name
    out=dest/f'eval_final_{args.sample_steps}'
    if (out/'complete.json').exists():raise RuntimeError('Completed evaluation exists; refusing overwrite')
    out.mkdir(parents=True,exist_ok=True)
    z=m.encoder(x);groups=tr.obs.Group.to_numpy()
    zz={g:z[torch.tensor(groups==g,device='cuda')] for g in [0,3]}
    means={g:zz[g].mean(0) for g in [0,3]}
    rms=float(torch.stack([v.var(0).mean() for v in zz.values()]).mean().sqrt())
    delta=means[3]-means[0]
    n=args.cells
    # Bounded-size train-only OT pairing. Whole-pool means/covariances retained.
    count=min(1200,len(zz[0]),len(zz[3]))
    subset={g:zz[g][torch.linspace(0,len(zz[g])-1,count,device='cuda').long()] for g in [0,3]}
    scales={g:zz[g].std(0).clamp_min(1e-5) for g in [0,3]}
    r0=(subset[0]-means[0])/scales[0]
    r3=(subset[3]-means[3])/scales[3]
    ii,jj=linear_sum_assignment(torch.cdist(r0,r3).square().cpu().numpy())
    pair=torch.tensor(jj[np.argsort(ii)],device='cuda')
    def latent(mode,sigma,alpha,origin,bb):
        center=origin.mean(0)+alpha*delta
        if mode in ['fixed','isotropic']:return center+sigma*bb['eta']
        a=zz[0][(bb['u0']*len(zz[0])).long()]-means[0]
        b=zz[3][(bb['u3']*len(zz[3])).long()]-means[3]
        if mode=='empirical':return center+((1-alpha)*a+alpha*b)/np.sqrt((1-alpha)**2+alpha**2)
        source=origin[torch.arange(n,device='cuda')%len(origin)]
        if mode=='paired':return source+alpha*delta
        if mode=='paired_noise':return source+alpha*delta+sigma*bb['eta']
        if mode=='ot':
            nearest=torch.cdist((source-means[0])/scales[0],r0).argmin(1)
            return (1-alpha)*source+alpha*subset[3][pair[nearest]]
        raise ValueError(mode)
    def draw(conditions,bb):
        pred=d.ddim_sample_loop(m,(n,tr.n_vars),noise=bb['noise'].clone(),
            model_kwargs={'z_mod':conditions},clip_denoised=False).cpu().numpy()
        if name=='finalclip':pred=np.maximum(pred,0)
        if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite generation')
        return pred
    valbank=bank(n,tr.n_vars,3000)
    testbank=bank(n,tr.n_vars,1000)
    origin=m.encoder(torch.tensor(np.asarray(va.X[va.obs.Group==0]),device='cuda'))
    candidates=[('fixed',.7),('isotropic',0.),('isotropic',.5*rms),('isotropic',rms),
        ('empirical',1.),('paired',1.),('ot',1.)]
    if name=='paper_cellnoise':
        candidates=[('paired_noise',s) for s in [0.,.5*rms,rms,.7]]
    if args.use_native_selection:
        assert name in CONTROLS and name!='native'
        frozen=json.loads((OUT/'inference_controls/native/eval_final_1000/selection.json').read_text())['selected']
        candidates=[(frozen['mode'],frozen['sigma'])]
    rows=[]
    for mode,sigma in candidates:
        ws=[];ms=[]
        for day in [0,3]:
            pred=draw(latent(mode,sigma,day/3,origin,valbank),valbank)
            truth=np.asarray(va.X[va.obs.Group==day])
            ws.append(np.mean([wasserstein_distance(truth[:,j],pred[:,j]) for j in range(tr.n_vars)]))
            ms.append(np.mean((truth.astype(float).mean(0)-pred.astype(float).mean(0))**2))
        row=dict(mode=mode,sigma=sigma,wasserstein=float(np.mean(ws)),mse=float(np.mean(ms)))
        rows.append(row);pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False)
        print('VALIDATE',name,json.dumps(row),flush=True)
    selected=min(rows,key=lambda r:r['wasserstein'])
    criterion=('Frozen native protocol; endpoint quality checked, not retuned' if args.use_native_selection
               else 'Independent endpoint gene-wise Wasserstein')
    write_json(dict(selected=selected,criterion=criterion,
        checkpoint_step=step,intermediate_RNA_used=False,common_random_bank=True),out/'selection.json')
    # Freeze selection before encoding any target-day RNA for diagnostics.
    origin=m.encoder(torch.tensor(np.asarray(te.X[te.obs.Group==0]),device='cuda'))
    trmeans={g:tr.X[tr.obs.Group==g].astype(float).mean(0) for g in [0,3]}
    refmeans={g:reference.X[reference.obs.Group==g].astype(float).mean(0) for g in [0,3]}
    records=[];gene_rows=[]
    for method in ['fixed','selected']:
        mode,sigma=('fixed',.7) if method=='fixed' else (selected['mode'],selected['sigma'])
        if name=='paper_cellnoise' and method=='fixed':mode='paired_noise'
        for day in [0,1,2,3]:
            conditions=latent(mode,sigma,day/3,origin,testbank)
            pred=draw(conditions,testbank)
            truth=np.asarray(te.X[te.obs.Group==day])
            row=dict(model=name,method=method,day=day,step=step,mode=mode,sigma=sigma,
                **scores(truth,pred,tr,reference))
            y=truth.astype(float).mean(0)
            baseline=(1-day/3)*trmeans[0]+day/3*trmeans[3]
            refbase=(1-day/3)*refmeans[0]+day/3*refmeans[3]
            row.update(baseline_r2=float(r2_score(y,baseline)),
                baseline_mse=float(np.mean((y-baseline)**2)),
                mse_gain=float(1-row['mean_mse']/np.mean((y-baseline)**2)),
                reference_baseline_r2=float(r2_score(y,refbase)),
                reference_mse_gain=float(1-row['mean_mse']/np.mean((y-refbase)**2)))
            records.append(row)
            np.savez_compressed(out/f'{method}_day{day}.npz',prediction=pred,genes=np.array(tr.var_names,dtype=str))
            gene_rows.extend(dict(method=method,day=day,gene=g,observed=float(y[j]),
                predicted=float(pred[:,j].astype(float).mean()),baseline=float(baseline[j])) for j,g in enumerate(tr.var_names))
            pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
            pd.DataFrame(gene_rows).to_csv(out/'gene_means.csv',index=False)
            print('TEST',name,method,day,row['r2'],row['T_pred'],row['transient_tp'],row['transient_fp'],flush=True)
    # Keep the identical day1 residuals and xT and remove only alpha * delta.
    day=1;alpha=1/3
    truth=np.asarray(te.X[te.obs.Group==day])
    chosen=latent(selected['mode'],selected['sigma'],alpha,origin,testbank)
    diagnostic=[('mean_direction_removed',chosen-alpha*delta)]
    if args.oracle:
        encoded=m.encoder(torch.tensor(truth,device='cuda'))
        diagnostic.append(('target_rna_oracle',encoded[torch.arange(n,device='cuda')%len(encoded)]))
    for method,conditions in diagnostic:
        pred=draw(conditions,testbank)
        row=dict(model=name,method=method,day=1,step=step,**scores(truth,pred,tr,reference))
        np.savez_compressed(out/f'{method}_day1.npz',prediction=pred,genes=np.array(tr.var_names,dtype=str))
        records.append(row);pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
        print('DIAGNOSTIC',name,method,row['r2'],row['T_pred'],flush=True)
    # Baselines use this model's complete training pool and the same endpoint data.
    baselines=[]
    aa=tr.X[groups==0][(testbank['u0'].cpu().numpy()*sum(groups==0)).astype(int)]
    bb=tr.X[groups==3][(testbank['u3'].cpu().numpy()*sum(groups==3)).astype(int)]
    for day in [1,2]:
        alpha=day/3;truth=np.asarray(te.X[te.obs.Group==day])
        for method,pred in [('cell_linear',(1-alpha)*aa+alpha*bb),
            ('endpoint_mixture',np.where((testbank['mix'].cpu().numpy()<alpha)[:,None],bb,aa))]:
            baselines.append(dict(model=name,day=day,method=method,**scores(truth,pred,tr,reference)))
    pd.DataFrame(baselines).to_csv(out/'distribution_baselines.csv',index=False)
    write_json(dict(args=vars(args),config=cfg,weight_sha256=sha(path),
        train_data_sha256=sha(folder(cfg)/'train.h5ad'),selected=selected,
        rows=len(records),source_rows=tr.n_obs,ot_pairs=count,
        sampling_inputs='Common fixed CUDA random input banks per role; no seed search',
        scope='Repeatedly inspected reproduction/development data; oracle is a diagnostic only'),out/'complete.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(VARIANTS)+CONTROLS,required=True)
    p.add_argument('--sample-steps',type=int,default=1000);p.add_argument('--cells',type=int,default=600)
    p.add_argument('--oracle',action='store_true')
    p.add_argument('--use-native-selection',action='store_true',help='Hold the native latent protocol fixed for a paired inference ablation')
    p.add_argument('--output-root',help='Separate output directory for replay; weights and frozen data still come from the original run')
    evaluate(p.parse_args())
