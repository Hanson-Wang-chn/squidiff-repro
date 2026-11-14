"""Held-out predictions and baselines; reports true-target R2 and notebook R2."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from common import ROOT, setup, seed_all, build_model, dense, write_json, record

def metrics(truth,pred,control=None):
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise FloatingPointError('Non-finite observed or predicted gene expression')
    a,b = np.asarray(truth,dtype=np.float64).mean(0), np.asarray(pred,dtype=np.float64).mean(0)
    m=dict(pearson=float(pearsonr(a,b).statistic),r2=float(r2_score(a,b)),
           notebook_reverse_r2=float(r2_score(b,a)), mean_mse=float(np.mean((a-b)**2)),
           variance_mae=float(np.mean(np.abs(np.var(truth,0)-np.var(pred,0)))),
           mean_gene_wasserstein=float(np.mean([wasserstein_distance(truth[:,j],pred[:,j]) for j in range(len(a))])))
    m['predicted_negative_fraction']=float(np.mean(pred<0))
    # Public notebook MMD compares the distribution of per-gene means, gamma=.05.
    m['notebook_mean_mmd']=float(rbf_kernel(a[:,None],a[:,None],gamma=.05).mean()+
        rbf_kernel(b[:,None],b[:,None],gamma=.05).mean()-2*rbf_kernel(a[:,None],b[:,None],gamma=.05).mean())
    # Energy distance in joint expression space; finite-sample V-statistic.
    m['energy_distance']=float(max(0.,2*cdist(truth,pred).mean()-cdist(truth,truth).mean()-cdist(pred,pred).mean()))
    if control is not None:
        c=np.asarray(control,dtype=np.float64).mean(0)
        da,db=a-c,b-c
        m.update(delta_pearson=float(pearsonr(da,db).statistic) if np.std(db)>0 else None,
                 delta_r2=float(r2_score(da,db)),
                 opposite_direction_fraction=float(np.mean(da*db<0)))
        # Differential genes ranked only for evaluation, never for fitting.
        top=np.argsort(np.abs(da))[-20:]
        m['top20_delta_mse']=float(np.mean((da[top]-db[top])**2))
        m['top20_delta_pearson']=float(pearsonr(da[top],db[top]).statistic) if np.std(db[top])>0 else None
    return m

@torch.inference_mode()
def sample(model,diffusion,z,batch_size=1024):
    outputs=[]
    for zz in z.split(batch_size):
        outputs.append(diffusion.ddim_sample_loop(model, (len(zz),model.input_layer.in_features),
                       model_kwargs={'z_mod':zz},clip_denoised=False).float().cpu().numpy())
    return np.concatenate(outputs)

@torch.inference_mode()
def main(args):
    setup();seed_all(42)
    data=ROOT/'data/processed'/args.dataset
    dest=ROOT/'results'/args.dataset/args.run
    if args.dataset=='ipsc' and args.checkpoint=='best.pt' and not args.no_calibration:
        selection_path=dest/'generation_selection.json'
        if selection_path.exists():selection=json.loads(selection_path.read_text())
        else:
            from calibrate import choose_ipsc
            selection=choose_ipsc(dest)
        args.checkpoint=selection['checkpoint'];args.latent_noise=selection['latent_noise']
    train=ad.read_h5ad(data/'train.h5ad');test=ad.read_h5ad(data/'test.h5ad')
    checkpoint=dest/args.checkpoint
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    weights=state.get('model',state)
    model,diffusion=build_model(train.n_vars,args.sample_steps)
    model.load_state_dict(weights);model.cuda().eval()
    x=torch.from_numpy(dense(train)).cuda(); tx=dense(test)
    z=model.encoder(x)
    group=train.obs.Group.to_numpy();tg=test.obs.Group.to_numpy()
    zmean={int(g):z[torch.as_tensor(group==g,device='cuda')].mean(0) for g in np.unique(group)}
    control=dense(train)[group==0]
    rows=[]; arrays={}; baselines={}; truths={}
    for rep in range(args.repeats):
        seed_all(1000+rep)
        if args.dataset=='norman':
            ctrl=z[torch.as_tensor(group==0,device='cuda')]
            ids=torch.arange(args.cells,device='cuda')%len(ctrl)
            # Intended paper equation: E(ctrl) + Delta(PTPN12) + Delta(ZBTB25),
            # retaining control cell heterogeneity.
            zz=ctrl[ids]+zmean[1]+zmean[2]-2*zmean[0]
            pred=sample(model,diffusion,zz)
            targets={'PTPN12+ZBTB25':(tx,pred)}
        else:
            source=test[tg==0]
            z0=model.encoder(torch.from_numpy(dense(source)).cuda())
            z3=model.encoder(torch.from_numpy(tx[tg==3]).cuda())
            direction=zmean[3]-zmean[0]
            targets={}
            for d in range(4):
                if d==0:zz=z0
                elif d==3:zz=z3
                else:
                    zz=z0.mean(0)+d/3*direction+args.latent_noise*torch.randn((len(z0),60),device='cuda')
                pred=sample(model,diffusion,zz)
                targets[f'day{d}']=(tx[tg==d],pred)
        for target,(truth,pred) in targets.items():
            row=dict(dataset=args.dataset,run=args.run,repeat=rep,target=target,
                     method='Squidiff',sample_steps=args.sample_steps,
                     checkpoint_step=state.get('step'),**metrics(truth,pred,control))
            rows.append(row);print(json.dumps(row),flush=True)
            if rep==0:
                arrays[target]=pred;truths[target]=truth
        pd.DataFrame(rows).to_csv(dest/f'metrics_{args.sample_steps}steps.csv',index=False)
    for target,truth in truths.items():
        nochange=control[:min(len(control),args.cells)]
        if args.dataset=='norman':
            additive=control[:args.cells]+dense(train)[group==1].mean(0)+dense(train)[group==2].mean(0)-2*control.mean(0)
        else:
            d=int(target[-1]);a=control[:600];b=dense(train)[group==3][:600]
            additive=(1-d/3)*a+d/3*b
        for label,pred in [('no_change',nochange),('expression_linear',additive)]:
            rows.append(dict(dataset=args.dataset,run=args.run,repeat=0,target=target,
                             method=label,**metrics(truth,pred,control)))
    df=pd.DataFrame(rows);df.to_csv(dest/f'metrics_{args.sample_steps}steps.csv',index=False)
    np.savez_compressed(dest/f'predictions_{args.sample_steps}steps.npz',**arrays,
                        genes=train.var_names.to_numpy().astype(str))
    for target,pred in arrays.items():
        truth=truths[target]
        gene_table=pd.DataFrame(dict(gene_id=train.var_names,
            gene_name=train.var['gene_name'].astype(str).to_numpy() if 'gene_name' in train.var else train.var_names,
            observed_mean=truth.mean(0),predicted_mean=pred.mean(0),
            observed_std=truth.std(0),predicted_std=pred.std(0),
            observed_delta=truth.mean(0)-control.mean(0),predicted_delta=pred.mean(0)-control.mean(0)))
        gene_table.to_csv(dest/f'gene_means_{target}_{args.sample_steps}steps.csv',index=False)
    fig,axs=plt.subplots(1,len(truths),figsize=(4*len(truths),4),squeeze=False)
    for ax,(target,truth) in zip(axs.flat,truths.items()):
        pred=arrays[target];m=metrics(truth,pred)
        a,b=truth.mean(0),pred.mean(0);ax.scatter(a,b,s=9,alpha=.7)
        low,high=min(a.min(),b.min()),max(a.max(),b.max());ax.plot([low,high],[low,high],'k--',lw=.8)
        ax.set(xlabel='Observed mean',ylabel='Predicted mean',title=f'{target}\nr={m["pearson"]:.3f}, R²={m["r2"]:.3f}')
    fig.tight_layout();fig.savefig(dest/f'mean_expression_{args.sample_steps}steps.png',dpi=180);plt.close(fig)
    pca=PCA(n_components=2,random_state=42)
    allx=np.concatenate([*truths.values(),*arrays.values()]);coords=pca.fit_transform(allx)
    fig,axs=plt.subplots(1,2,figsize=(10,4));offset=0
    for idx,d in enumerate([truths,arrays]):
        for label,arr in d.items():
            co=coords[offset:offset+len(arr)];offset+=len(arr)
            axs[idx].scatter(co[:,0],co[:,1],s=5,alpha=.4,label=label)
        axs[idx].set_title(['Observed','Squidiff'][idx]);axs[idx].legend(markerscale=2)
    fig.tight_layout();fig.savefig(dest/f'pca_{args.sample_steps}steps.png',dpi=180);plt.close(fig)
    if args.dataset=='ipsc':
        symbols=list(train.var_names);markers=[g for g in ['NANOG','T','GATA6'] if g in symbols]
        fig,axs=plt.subplots(1,len(markers),figsize=(4*len(markers),3),squeeze=False)
        for ax,g in zip(axs.flat,markers):
            j=symbols.index(g)
            ax.plot(range(4),[truths[f'day{d}'][:,j].mean() for d in range(4)],'o-',label='Observed')
            ax.plot(range(4),[arrays[f'day{d}'][:,j].mean() for d in range(4)],'o-',label='Predicted')
            ax.set(title=g,xlabel='Day',ylabel='Mean expression');ax.legend()
        fig.tight_layout();fig.savefig(dest/'marker_trends.png',dpi=180);plt.close(fig)
    summary=df[df.method=='Squidiff'].groupby('target')[['pearson','r2','notebook_reverse_r2','delta_pearson','delta_r2']].mean().to_dict('index')
    write_json(dict(checkpoint=str(checkpoint),step=state.get('step'),repeats=args.repeats,
                    sample_steps=args.sample_steps,latent_noise=args.latent_noise,metrics=summary),dest/f'evaluation_{args.sample_steps}steps.json')
    record(f'{args.dataset}/{args.run} 评估完成（DDIM {args.sample_steps} 步，{args.repeats} 次采样）：'+json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['norman','ipsc']);p.add_argument('--run',default='v1')
    p.add_argument('--checkpoint',default='best.pt');p.add_argument('--sample-steps',type=int,default=100)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--cells',type=int,default=600)
    p.add_argument('--latent-noise',type=float,default=.7)
    p.add_argument('--no-calibration',action='store_true',help='Use the originally requested checkpoint and latent sigma directly')
    main(p.parse_args())
