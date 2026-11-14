"""Frozen scientific diagnostics for controlled runs. Never overwrites v1 outputs."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from common import ROOT, setup, seed_all, build_model, write_json, atomic_save, record
from evaluate import metrics

@torch.inference_mode()
def run(args):
    setup()
    if args.inference_math=='ieee':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    data=ROOT/args.data_root/args.dataset
    tr=ad.read_h5ad(data/'train.h5ad');te=ad.read_h5ad(data/'test.h5ad')
    dest=ROOT/'results'/args.dataset/args.run
    outdir=dest/'research_eval'/args.tag;outdir.mkdir(parents=True,exist_ok=True)
    s=torch.load(dest/args.checkpoint,map_location='cpu',weights_only=False)
    weights=s[args.weights] if args.weights in s else s
    m,d=build_model(tr.n_vars,args.sample_steps,model_variant=s.get('config',{}).get('model_variant','source'));m.load_state_dict(weights);m.cuda().eval()
    if args.bn_recalibrate:
        # Only training cells are used to refresh encoder BN statistics.
        bns=[x for x in m.encoder.modules() if isinstance(x,torch.nn.BatchNorm1d)]
        for b in bns:b.reset_running_stats();b.momentum=None
        m.encoder.train();seed_all(777)
        for idx in torch.randperm(tr.n_obs).split(64):
            if len(idx)>1:m.encoder(torch.tensor(tr.X[idx.numpy()],device='cuda'))
        m.encoder.eval()
    seed_all(123)
    x=torch.tensor(tr.X,dtype=torch.float32,device='cuda')
    z=m.encoder(x);g=tr.obs.Group.to_numpy();tg=te.obs.Group.to_numpy()
    zz={int(k):z[torch.as_tensor(g==k,device='cuda')] for k in np.unique(g)}
    zm={k:v.mean(0) for k,v in zz.items()}
    latent_stats={str(k):{'cells':len(v),'mean_norm':float(v.mean(0).norm()),
                    'within_rms_std':float(v.std(0).square().mean().sqrt()),
                    'mean_radius':float((v-v.mean(0)).norm(dim=1).mean())} for k,v in zz.items()}
    if args.dataset=='ipsc':latent_stats['endpoint_mean_distance']=float((zm[3]-zm[0]).norm())
    write_json(latent_stats,outdir/'latent_statistics.json')
    means={int(k):tr.X[g==k].astype(np.float64).mean(0) for k in np.unique(g)}
    control=tr.X[g==0]
    rows=[];gene_rows=[]
    def draw(latent):
        result=[]
        for part in latent.split(1024):
            result.append(d.ddim_sample_loop(m,(len(part),tr.n_vars),model_kwargs={'z_mod':part}).cpu().numpy())
        return np.concatenate(result)
    def evaluate(label,target,latent,truth,base,rep,extra=None):
        pred=draw(latent);truth=truth.astype(np.float64);base=np.asarray(base,dtype=np.float64)
        row=dict(dataset=args.dataset,run=args.run,tag=args.tag,mode=label,target=target,repeat=rep,
                 step=s.get('step'),sample_steps=args.sample_steps,generated_cells=len(pred),**metrics(truth,pred,control))
        observed=truth.mean(0);pm=pred.astype(np.float64).mean(0)
        bmse=float(np.mean((observed-base)**2))
        row.update(exact_linear_r2=float(r2_score(observed,base)),exact_linear_mse=bmse,
                   relative_mse_gain=1-row['mean_mse']/bmse if bmse>0 else None)
        if args.dataset=='norman':
            real_i=observed-base;pred_i=pm-base
            row.update(interaction_r2=float(r2_score(real_i,pred_i)),
                       interaction_pearson=float(pearsonr(real_i,pred_i).statistic))
        else:
            j=tr.var_names.get_loc('T');row.update(true_T_mean=float(observed[j]),predicted_T_mean=float(pm[j]))
            envelope=np.maximum(means[0],means[3]);real_excess=observed-envelope;pred_excess=pm-envelope
            for threshold in [.25,.5,1.]:
                mask=real_excess>=threshold
                row[f'transient_genes_at_{threshold}']=int(mask.sum())
                row[f'transient_half_amplitude_recall_at_{threshold}']=float(np.mean(pred_excess[mask]>=.5*real_excess[mask])) if mask.any() else None
        if extra:row.update(extra)
        rows.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(outdir/f'{label}_{target}_rep{rep}.npz',prediction=pred,genes=np.array(tr.var_names,dtype=str))
        if rep==0:
            for j,name in enumerate(tr.var_names):
                gene_rows.append(dict(mode=label,target=target,gene_id=name,
                    gene_name=str(tr.var.gene_name.iloc[j]) if 'gene_name' in tr.var else name,
                    observed_mean=float(observed[j]),predicted_mean=float(pm[j]),exact_linear_mean=float(base[j])))
        pd.DataFrame(rows).to_csv(outdir/'metrics.csv',index=False)
    for rep in range(args.repeats):
        if args.dataset=='norman':
            truth=te.X;baseline=means[1]+means[2]-means[0]
            for mode in args.modes.split(','):
                seed_all(1000+rep)
                if mode=='centroid':latent=(zm[1]+zm[2]-zm[0]).expand(args.cells,-1)
                elif mode=='cell_shift':latent=zz[0][torch.arange(args.cells,device='cuda')%len(zz[0])]+zm[1]+zm[2]-2*zm[0]
                elif mode=='literal':
                    # Exact released notebook shape bug: a one-dimensional control
                    # mean is averaged to a scalar; len(z_ctrl)==60 cells result.
                    latent=(zm[0].mean()+zm[1]+zm[2]-zm[0]).expand(60,-1)
                elif mode=='oracle':latent=m.encoder(torch.tensor(truth,device='cuda'))
                elif mode=='target_centroid':
                    if 3 not in zm:continue
                    latent=zm[3].expand(args.cells,-1)
                else:raise ValueError(mode)
                evaluate(mode,'AB',latent,truth,baseline,rep)
        else:
            z0=m.encoder(torch.tensor(te.X[tg==0],device='cuda'))
            for day in [1,2]:
                truth=te.X[tg==day];baseline=(1-day/3)*means[0]+day/3*means[3]
                for mode in args.modes.split(','):
                    for sigma in ([float(v) for v in args.sigmas.split(',')] if mode=='centroid' else [0.]):
                        seed_all(1000+rep)
                        if mode=='centroid':latent=z0.mean(0)+day/3*(zm[3]-zm[0])+sigma*torch.randn((len(z0),60),device='cuda')
                        elif mode=='cell_shift':latent=z0+day/3*(zm[3]-zm[0])
                        elif mode=='oracle':latent=m.encoder(torch.tensor(truth,device='cuda'))
                        elif mode=='target_centroid':
                            if day not in zm:continue
                            latent=zm[day].expand(len(z0),-1)
                        else:raise ValueError(mode)
                        label=f'{mode}_sigma{sigma:g}' if mode=='centroid' else mode
                        evaluate(label,f'day{day}',latent,truth,baseline,rep,{'sigma':sigma})
    pd.DataFrame(gene_rows).to_csv(outdir/'gene_means.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['mode','target'])[['pearson','r2','relative_mse_gain']].mean().to_dict('index')
    summary={str(k):v for k,v in summary.items()}
    write_json({'args':vars(args),'checkpoint_step':s.get('step'),'summary':summary,
                'oracle_is_not_forecasting':'Real target expression is encoded for oracle diagnostic only'},outdir/'complete.json')
    record(f'研究评估 {args.dataset}/{args.run}/{args.tag} 完成：'+json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['norman','ipsc']);p.add_argument('--run',required=True)
    p.add_argument('--data-root',default='data/processed');p.add_argument('--checkpoint',default='latest.pt')
    p.add_argument('--weights',choices=['model','ema'],default='model');p.add_argument('--tag',default='raw')
    p.add_argument('--modes',default='centroid,cell_shift,oracle');p.add_argument('--sigmas',default='0,0.5,0.7')
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--cells',type=int,default=600)
    p.add_argument('--sample-steps',type=int,default=1000);p.add_argument('--bn-recalibrate',action='store_true')
    p.add_argument('--inference-math',choices=['tf32','ieee'],default='tf32')
    run(p.parse_args())
