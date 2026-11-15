"""Isolated issue-8 reproduction: native training and endpoint-only model selection.

Never modifies previous results or upstream sources. All scores use original CSV
units, even for models trained on ln or z-score input. See report-2.md.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/Squidiff'))
from Squidiff import logger
from Squidiff.MLPModel import MLPModel
from Squidiff.train_util import TrainLoop
from common import build_model, seed_all, atomic_save, write_json
from evaluate import metrics

OUT = ROOT / 'results/round2'
VARIANTS = {
    'native': dict(panel='source', transform='identity', width=2048, latent_noise=0., wd=0.),
    'ln': dict(panel='source', transform='ln', width=2048, latent_noise=0., wd=0.),
    'log2hvg': dict(panel='log2hvg', transform='identity', width=2048, latent_noise=0., wd=0.),
    'compact': dict(panel='source', transform='identity', width=512, latent_noise=.1, wd=1e-4),
    'zscore': dict(panel='source', transform='zscore', width=2048, latent_noise=0., wd=0.),
}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/research/source/ipsc'
    genes = json.loads((ROOT / 'docs/round2/hvg_features.json').read_text())['declare_log2']
    a = ad.read_h5ad(ROOT / 'data/cache/ipsc_full.h5ad')
    audit = {}
    for panel in ['source', 'log2hvg']:
        dest = ROOT / 'data/round2' / panel
        dest.mkdir(parents=True, exist_ok=True)
        parts = {}
        for part in ['train', 'val', 'test']:
            original = ad.read_h5ad(source / f'{part}.h5ad')
            b = original.copy() if panel == 'source' else a[original.obs_names, genes].copy()
            b.write_h5ad(dest / f'{part}.h5ad', compression='gzip')
            parts[part] = b
        tr, va, te = (parts[x] for x in ['train', 'val', 'test'])
        assert set(tr.obs_names).isdisjoint(va.obs_names)
        assert set(tr.obs_names).isdisjoint(te.obs_names[te.obs.Group.isin([1, 2])])
        audit[panel] = dict(genes=list(tr.var_names), train_rows=tr.n_obs,
            train_unique=len(set(tr.obs_names)), val_rows=va.n_obs, test_rows=te.n_obs,
            train_test_overlap=len(set(tr.obs_names)&set(te.obs_names)),
            scope='Source cells fixed; intermediate-day discovery used by original HVG protocol',
            hashes={p:hashlib.sha256((dest/f'{p}.h5ad').read_bytes()).hexdigest() for p in parts})
        write_json(audit[panel], dest/'manifest.json')
    write_json(audit, OUT/'data_audit.json')
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def load_data(config):
    folder = ROOT / 'data/round2' / config['panel']
    return tuple(ad.read_h5ad(folder/f'{p}.h5ad') for p in ['train','val','test'])


def transform(x, config):
    x = np.asarray(x, dtype=np.float32)
    if config['transform'] == 'ln':
        return x * np.float32(np.log(2))
    if config['transform'] == 'zscore':
        return (x-np.array(config['center'],dtype=np.float32))/np.array(config['scale'],dtype=np.float32)
    return x


def inverse(x, config):
    if config['transform'] == 'ln':
        return x / np.log(2)
    if config['transform'] == 'zscore':
        # Undo finite precision roundoff at the physical expression-zero floor.
        return np.maximum(x*np.array(config['scale'])+np.array(config['center']),0.)
    return x


def create(config, steps=1000):
    m, d = build_model(config['genes'], steps)
    if config['width'] != 2048:
        w = config['width']
        m = MLPModel(config['genes'], config['genes'], num_layers=3,
                     hidden_sizes=w, time_pos_dim=w, time_embed_dim=w, use_encoder=True)
    if config['latent_noise']:
        original = d.training_losses
        def noisy(model, x_start, t, model_kwargs=None, noise=None):
            kw = dict(model_kwargs or {})
            z = model.encoder(x_start)
            # Training-only local regularizer, scaled by batch latent spread.
            sigma = z.detach().std(0).square().mean().sqrt() * config['latent_noise']
            kw['z_mod'] = z + torch.randn_like(z)*sigma if model.training else z
            return original(model, x_start, t, model_kwargs=kw, noise=noise)
        d.training_losses = noisy
    return m, d


def signed_sampling(d, config):
    """For z-scores, clip to expression zero in standardized coordinates.

    Keep the source fixed-variance reverse equations, replacing only the wrong
    zero floor. This is explicitly an inference correction, not source protocol.
    """
    from Squidiff.diffusion import _extract_into_tensor
    floor=torch.tensor(-np.array(config['center'])/np.array(config['scale']),device='cuda',dtype=torch.float32)
    def p_mean_variance(model,x,t,clip_denoised=False,denoised_fn=None,model_kwargs=None):
        eps=model(x,d._scale_timesteps(t),**(model_kwargs or {}))
        x0=d._predict_xstart_from_eps(x,t,eps)
        if denoised_fn is not None:x0=denoised_fn(x0)
        x0=torch.maximum(x0,floor)
        mean,_,_=d.q_posterior_mean_variance(x0,x,t)
        variance=np.append(d.posterior_variance[1],d.betas[1:])
        return dict(mean=mean,pred_xstart=x0,
                    variance=_extract_into_tensor(variance,t,x.shape),
                    log_variance=_extract_into_tensor(np.log(variance),t,x.shape))
    # SpacedDiffusion wraps the model before invoking this method; assigning to
    # the instance bypasses that wrapper, so do the same wrapping explicitly.
    d.p_mean_variance=lambda model,*a,**kw:p_mean_variance(d._wrap_model(model),*a,**kw)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu']); torch.cuda.set_rng_state_all(s['cuda'])


@torch.no_grad()
def validation(m, d, val, config):
    saved = rng(); seed_all(12345); m.eval()
    v = torch.tensor(transform(val.X, config), device='cuda')
    rows = []
    for group in [0,3]:
        for chunk in v[torch.tensor(val.obs.Group.to_numpy()==group,device='cuda')].split(128):
            for t in [25,100,250,500,750,950]:
                tt = torch.full((len(chunk),), t, device='cuda', dtype=torch.long)
                loss = d.training_losses(m, chunk, tt, model_kwargs={'drug_dose':None,'control_feature':None})['loss']
                rows.extend(loss.cpu().tolist())
    m.train(); restore(saved)
    return float(np.mean(rows))


class Cells(Dataset):
    def __init__(self, a, config):
        self.x = torch.tensor(transform(a.X,config))
        self.g = a.obs.Group.to_numpy()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return dict(feature=self.x[i], group=self.g[i])


class ObservedLoop(TrainLoop):
    """Keep native run_loop/run_step/forward_backward; add read-only observation.

    save() only changes artifact format. log_step() validation restores every RNG
    and train/eval mode. Clearing the already-backpropagated loss history saves RAM.
    """
    def log_step(self):
        super().log_step()
        n = self.step + 1
        if n % 1000 == 0:
            value = float(torch.stack([x.detach() for x in self.loss_list]).mean())
            elapsed = time.monotonic()-self.last_time
            row = dict(step=n, loss=value, steps_per_second=1000/elapsed)
            self.last_time = time.monotonic(); self.loss_list.clear()
            if n % 10000 == 0 or n == self.target:
                row['val_loss'] = validation(self.model,self.diffusion,self.val,self.config)
                state = dict(model=self.model.state_dict(),step=n,config=self.config)
                if row['val_loss'] < self.best:
                    self.best = row['val_loss']; atomic_save(state,self.dest/'best.pt')
                if n in [10000,50000,self.target]:
                    atomic_save(state,self.dest/f'step_{n:06d}.pt')
            with open(self.dest/'train.jsonl','a') as f:
                f.write(json.dumps(row)+'\n')
            write_json(dict(step=n, target=self.target, status='training',pid=os.getpid()),self.dest/'status.json')
            print(json.dumps(row),flush=True)

    def save(self):
        # Native run_loop saves at index 0, 10000, ... and after incrementing
        # self.step on exit. Record the actual number of optimizer updates.
        n = min(self.step+1,self.target)
        atomic_save(dict(model=self.model.state_dict(),
            ema=self.mp_trainer.master_params_to_state_dict(self.ema_params[0]),
            optimizer=self.opt.state_dict(),rng=rng(),step=n,
            config=self.config,best_val=self.best),self.dest/'latest.pt')


def train(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    seed_all(args.seed)
    config = dict(VARIANTS[args.variant],variant=args.variant,seed=args.seed,steps=args.steps,
                  batch_size=64,lr=1e-4,math='IEEE FP32',sampling='native CPU loader + NumPy timesteps')
    tr,va,_ = load_data(config)
    config['genes'] = tr.n_vars
    if config['transform']=='zscore':
        config['center']=tr.X.astype(float).mean(0).tolist()
        config['scale']=np.maximum(tr.X.astype(float).std(0),1e-6).tolist()
    dest = OUT / f'{args.variant}_s{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'status.json').exists() and not args.resume:
        raise RuntimeError(f'Existing run: {dest}; use a new seed/directory, never overwrite')
    config['source_sha256']={name:hashlib.sha256((ROOT/'upstream/Squidiff/Squidiff'/name).read_bytes()).hexdigest()
                             for name in ['train_util.py','MLPModel.py','diffusion.py']}
    config['train_data_sha256']=hashlib.sha256((ROOT/'data/round2'/config['panel']/'train.h5ad').read_bytes()).hexdigest()
    if args.resume:
        saved_config=json.loads((dest/'config.json').read_text())
        for key in ['variant','seed','steps','genes','panel','transform','width','latent_noise',
                    'wd','lr','batch_size','math','sampling','train_data_sha256','source_sha256']:
            assert saved_config[key]==config[key],key
    m,d = create(config); m.cuda().train()
    config['parameters']=sum(p.numel() for p in m.parameters())
    write_json(config,dest/'config.json')
    logger.configure(str(dest/'native_logs'),format_strs=['json'])
    loop = ObservedLoop(model=m,diffusion=d,
        data=DataLoader(Cells(tr,config),batch_size=64,shuffle=True),batch_size=64,microbatch=-1,
        lr=1e-4,ema_rate='0.9999',log_interval=1000,save_interval=10000,
        resume_checkpoint=str(dest),lr_anneal_steps=args.steps,weight_decay=config['wd'])
    loop.dest=dest; loop.config=config; loop.val=va; loop.target=args.steps
    if args.variant != 'native':
        # These two norms are logging only in FP32. Avoid ~140 device syncs per
        # update for sensitivity runs; the native reference keeps them intact.
        loop.mp_trainer._compute_norms = lambda grad_scale=1.: (0., 0.)
        config['logging_norms_disabled'] = True
        write_json(config,dest/'config.json')
    loop.best=float('inf'); loop.last_time=time.monotonic()
    if args.resume:
        state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(state['model']);loop.opt.load_state_dict(state['optimizer'])
        loop.ema_params=[[v.to('cuda') for v in loop.mp_trainer.state_dict_to_master_params(state['ema'])]]
        loop.step=state['step'];loop.best=state['best_val'];restore(state['rng'])
        print('RESUMED',loop.step,flush=True)
        del state
    config['cuda_mps_pipe_directory']=os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
    write_json(config,dest/'config.json')
    started=time.monotonic()
    loop.run_loop()
    assert loop.step==args.steps
    atomic_save(dict(model=m.state_dict(),step=loop.step,config=config),dest/'final.pt')
    write_json(dict(step=loop.step,target=args.steps,status='trained',pid=os.getpid(),
                    seconds=time.monotonic()-started,best_val=loop.best),dest/'status.json')
    print('TRAINED',dest,flush=True)


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    dest=OUT/f'{args.variant}_s{args.seed}'
    config=json.loads((dest/'config.json').read_text())
    tr,va,te=load_data(config)
    assert hashlib.sha256((ROOT/'data/round2'/config['panel']/'train.h5ad').read_bytes()).hexdigest()==config['train_data_sha256'], 'Training artifact changed'
    for name,sha in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/'upstream/Squidiff/Squidiff'/name).read_bytes()).hexdigest()==sha, name
    ckpt=torch.load(dest/args.checkpoint,map_location='cpu',weights_only=False)
    m,d=create(config,args.sample_steps);m.load_state_dict(ckpt['model']);m.cuda().eval()
    if args.signed:
        assert config['transform']=='zscore'
        signed_sampling(d,config)
    base_out=dest/f'eval_{Path(args.checkpoint).stem}_{args.sample_steps}{"_signed" if args.signed else ""}'
    out=Path(str(base_out)+('_mse' if args.criterion=='mse' else ''))
    out.mkdir(exist_ok=True)
    x=torch.tensor(transform(tr.X,config),device='cuda')
    z=m.encoder(x);g=tr.obs.Group.to_numpy()
    zz={i:z[torch.tensor(g==i,device='cuda')] for i in [0,3]}
    means={i:zz[i].mean(0) for i in [0,3]}
    rms=float(torch.stack([v.var(0).mean() for v in zz.values()]).mean().sqrt())
    dim=z.shape[1]
    endpoint_distance=float((means[3]-means[0]).norm())
    write_json(dict(within_group_rms=rms,endpoint_mean_distance=endpoint_distance,
        endpoint_distance_over_rms_radius=endpoint_distance/(np.sqrt(dim)*max(rms,1e-12)),
        center_nearest_training_distance={str(alpha):float(torch.cdist(
            ((1-alpha)*means[0]+alpha*means[3])[None,:],z).min()) for alpha in [0.,1/3,2/3,1.]},
        scope='Descriptive encoder geometry; latent coordinates have reparameterization freedom'),out/'latent_geometry.json')
    candidates=[('fixed',.7),('isotropic',0.),('isotropic',.5*rms),('isotropic',rms),
                ('empirical',1.),('paired',1.),('ot',1.)]
    # One-to-one endpoint pairing in standardized semantic residual coordinates.
    # All fitting uses training endpoints; target-day expression is never encoded.
    from scipy.optimize import linear_sum_assignment
    scale0=zz[0].std(0).clamp_min(1e-5)
    scale3=zz[3].std(0).clamp_min(1e-5)
    r0=(zz[0]-means[0])/scale0;r3=(zz[3]-means[3])/scale3
    cost=torch.cdist(r0,r3).square().cpu().numpy()
    ii,jj=linear_sum_assignment(cost)
    pairing=torch.tensor(jj[np.argsort(ii)],device='cuda')
    def latent(mode,sigma,alpha,origin,n):
        if mode in ['isotropic','fixed']:
            return origin.mean(0)+alpha*(means[3]-means[0])+sigma*torch.randn(n,dim,device='cuda')
        if mode=='empirical':
            a=zz[0][torch.randint(len(zz[0]),(n,),device='cuda')]-means[0]
            b=zz[3][torch.randint(len(zz[3]),(n,),device='cuda')]-means[3]
            # Preserve within-endpoint covariance and its natural scale.
            residual=((1-alpha)*a+alpha*b)/np.sqrt((1-alpha)**2+alpha**2)
            return origin.mean(0)+alpha*(means[3]-means[0])+residual
        if mode=='paired':
            # Cell-wise semantic translation; preserve the source residuals.
            delta=means[3]-means[0]
            return origin[torch.arange(n,device='cuda')%len(origin)]+alpha*delta
        if mode=='ot':
            source=origin[torch.arange(n,device='cuda')%len(origin)]
            nearest=torch.cdist((source-means[0])/scale0,r0).argmin(1)
            return (1-alpha)*source+alpha*zz[3][pairing[nearest]]
        raise ValueError(mode)
    def draw(z,noise=None):
        p=[]
        offset=0
        for chunk in z.split(600):
            p.append(d.ddim_sample_loop(m,(len(chunk),tr.n_vars),
                noise=None if noise is None else noise[offset:offset+len(chunk)],
                model_kwargs={'z_mod':chunk},clip_denoised=False).cpu().numpy())
            offset+=len(chunk)
        return inverse(np.concatenate(p),config)
    candidates_rows=[]
    vo=m.encoder(torch.tensor(transform(va.X[va.obs.Group==0],config),device='cuda'))
    if args.reuse_candidates:
        previous=json.loads((base_out/'complete.json').read_text())
        for key in ['variant','seed','checkpoint','sample_steps','cells','signed']:
            assert previous['args'].get(key,False)==getattr(args,key),key
        assert previous['config']['train_data_sha256']==config['train_data_sha256']
        candidates_rows=pd.read_csv(base_out/'candidates.csv').to_dict('records')
        pd.DataFrame(candidates_rows).to_csv(out/'candidates.csv',index=False)
    else:
        for mode,sigma in candidates:
            ws=[];ms=[]
            for day in [0,3]:
                seed_all(3000)
                pred=draw(latent(mode,sigma,day/3,vo,args.cells))
                truth=va.X[va.obs.Group==day]
                ws.append(float(np.mean([wasserstein_distance(truth[:,j],pred[:,j]) for j in range(tr.n_vars)])))
                ms.append(float(np.mean((truth.astype(float).mean(0)-pred.mean(0))**2)))
            row=dict(mode=mode,sigma=sigma,wasserstein=float(np.mean(ws)),mse=float(np.mean(ms)))
            candidates_rows.append(row); print('VALIDATE',json.dumps(row),flush=True)
            pd.DataFrame(candidates_rows).to_csv(out/'candidates.csv',index=False)
    selected=min(candidates_rows,key=lambda r:r[args.criterion])
    write_json(dict(selected=selected,within_latent_rms=rms,checkpoint_step=ckpt['step'],
                    criterion='Independent endpoints only; '+args.criterion,
                    middle_days_used=False),out/'selection.json')
    # Test is accessed for scoring only after endpoint candidate selection.
    origin=m.encoder(torch.tensor(transform(te.X[te.obs.Group==0],config),device='cuda'))
    expression_means={i:tr.X[tr.obs.Group==i].astype(float).mean(0) for i in [0,3]}
    common_panel=json.loads((OUT/'data_audit.json').read_text())
    common=set(common_panel['source']['genes'])&set(common_panel['log2hvg']['genes'])
    common_idx=np.array([i for i,s in enumerate(tr.var_names) if s in common])
    rows=[];gene_rows=[]
    methods=([('selected',selected['sigma'])] if args.selected_only else [('fixed',.7),('selected',selected['sigma'])])
    inverted=None
    if args.full:
        methods += [('empirical',1.),('paired',1.),('ot',1.),('inversion',1.),('no_direction',selected['sigma']),('oracle',0.)]
        source_x=torch.tensor(transform(te.X[te.obs.Group==0],config),device='cuda')
        source_ids=torch.arange(args.cells,device='cuda')%len(source_x)
        inverted=source_x[source_ids].clone()
        source_z=origin[source_ids]
        for i in range(d.num_timesteps-1):
            tt=torch.full((len(inverted),),i,dtype=torch.long,device='cuda')
            inverted=d.ddim_reverse_sample(m,inverted,tt,model_kwargs={'z_mod':source_z})['sample']
    for label,sigma in methods:
        for day in ([0,1,2,3] if args.full else [1,2]):
            for rep in range(args.repeats):
                seed_all(1000+rep)
                truth=te.X[te.obs.Group==day]
                mode=selected['mode'] if label in ['selected','no_direction'] else label
                alpha=0. if label=='no_direction' else day/3
                if label=='oracle':
                    target=m.encoder(torch.tensor(transform(truth,config),device='cuda'))
                elif label=='inversion':
                    target=origin[torch.arange(args.cells,device='cuda')%len(origin)]+day/3*(means[3]-means[0])
                else:
                    target=latent(mode,sigma,alpha,origin,args.cells)
                pred=draw(target,noise=inverted if label=='inversion' else None)
                baseline=(1-day/3)*expression_means[0]+day/3*expression_means[3]
                y=truth.astype(float).mean(0);p=pred.astype(float).mean(0)
                envelope=np.maximum(expression_means[0],expression_means[3])
                actual=y-envelope>=.5;called=p-envelope>=.5
                j=tr.var_names.get_loc('T')
                row=dict(variant=args.variant,seed=args.seed,step=ckpt['step'],checkpoint=args.checkpoint,
                    sample_steps=args.sample_steps,method=label,latent_mode=mode,sigma=sigma,day=day,repeat=rep,
                    **metrics(truth,pred,tr.X[tr.obs.Group==0]))
                row.update(baseline_r2=float(r2_score(y,baseline)),mse_gain=1-row['mean_mse']/np.mean((y-baseline)**2),
                    common_genes=len(common_idx),common_r2=float(r2_score(y[common_idx],p[common_idx])),
                    common_baseline_r2=float(r2_score(y[common_idx],baseline[common_idx])),
                    T_true=float(y[j]),T_pred=float(p[j]),T_above3=float((pred[:,j]>3).mean()),
                    transient_actual=int(actual.sum()),transient_called=int(called.sum()),transient_tp=int((called&actual).sum()),
                    transient_half_recall=float(np.mean((p-envelope)[actual]>=.5*(y-envelope)[actual])) if actual.any() else None,
                    predicted_max=float(pred.max()))
                rows.append(row)
                np.savez_compressed(out/f'{label}_day{day}_rep{rep}.npz',prediction=pred,genes=np.array(tr.var_names,dtype=str))
                if rep==0:
                    gene_rows.extend(dict(method=label,day=day,gene=name,observed=y[k],predicted=p[k],baseline=baseline[k],endpoint_max=envelope[k])
                                     for k,name in enumerate(tr.var_names))
                pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
                pd.DataFrame(gene_rows).to_csv(out/'gene_means.csv',index=False)
                print('TEST',json.dumps(row),flush=True)
    write_json(dict(args=vars(args),config=config,selected=selected,rows=len(rows)),out/'complete.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','evaluate'])
    p.add_argument('--variant',choices=list(VARIANTS),default='native');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--checkpoint',default='final.pt')
    p.add_argument('--sample-steps',type=int,default=100);p.add_argument('--cells',type=int,default=600)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--full',action='store_true')
    p.add_argument('--signed',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--criterion',choices=['wasserstein','mse'],default='wasserstein')
    p.add_argument('--reuse-candidates',action='store_true')
    p.add_argument('--selected-only',action='store_true')
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='train':train(args)
    else:evaluate(args)
