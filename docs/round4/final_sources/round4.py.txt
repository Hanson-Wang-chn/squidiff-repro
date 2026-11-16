"""Last controlled endpoint-only day1 attempt; immutable upstream and old results.

All variants use the same source203 panel, final 100k updates and seed42.
Changes are declared interventions, not claims about the author's original run.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.cluster import KMeans

from common import ROOT, build_model, seed_all, atomic_save, write_json
import round2 as previous
from Squidiff import logger
from Squidiff.MLPModel import MLPBlock
from Squidiff.train_util import TrainLoop

OUT = ROOT/'results/round4'
DATA = ROOT/'data/round4'
VARIANTS = {
    'layernorm': dict(panel='source',encoder_norm='layer',block_skip=False,sampler='native'),
    'resblock': dict(panel='source',encoder_norm='batch',block_skip=True,sampler='native'),
    'fullsource': dict(panel='fullsource',encoder_norm='batch',block_skip=False,sampler='native'),
    'supportbalance': dict(panel='source',encoder_norm='batch',block_skip=False,sampler='endpoint_cluster_balanced'),
}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


class ResidualBlock(MLPBlock):
    def forward(self,x,emb,z_sem):
        return x + super().forward(x,emb,z_sem)


def create(config, steps=1000):
    m,d = build_model(config['genes'],steps)
    if config['encoder_norm']=='layer':
        m.encoder.bn1=nn.LayerNorm(m.hidden_sizes)
        m.encoder.bn2=nn.LayerNorm(m.hidden_sizes)
    if config['block_skip']:
        for block in m.mlp_blocks:
            block.__class__=ResidualBlock
    return m,d


def folder(config):
    return ROOT/'data/round2/source' if config['panel']=='source' else DATA/config['panel']


def load_data(config):
    p=folder(config)
    return tuple(ad.read_h5ad(p/f'{part}.h5ad') for part in ['train','val','test'])


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    full_dir=DATA/'fullsource'
    if (full_dir/'manifest.json').exists():
        raise RuntimeError('Prepared round4 data already exists')
    full_dir.mkdir(parents=True,exist_ok=True)
    tr,va,te=load_data(dict(panel='source'))
    full=ad.read_h5ad(ROOT/'data/cache/ipsc_full.h5ad')
    excluded=set(va.obs_names)|set(te.obs_names)
    mask=full.obs.Group.isin([0,3])&~full.obs_names.isin(excluded)
    expanded=full[mask,tr.var_names].copy()
    expanded.uns.clear();expanded.layers.clear()
    assert expanded.obs_names.is_unique
    assert set(expanded.obs_names).isdisjoint(va.obs_names)
    assert set(expanded.obs_names).isdisjoint(te.obs_names)
    for part,a in [('train',expanded),('val',va),('test',te)]:
        a.write_h5ad(full_dir/f'{part}.h5ad',compression='gzip')
    write_json(dict(rows=expanded.n_obs,groups=expanded.obs.Group.value_counts().to_dict(),
        genes=tr.var_names.tolist(),validation_and_test_fixed=True,
        training_only_endpoints=True,train_test_id_overlap=0,
        feature_scope='Identical historical source203 panel; four-day HVG discovery inherited, not a new endpoint-only feature claim',
        hashes={p:sha(full_dir/f'{p}.h5ad') for p in ['train','val','test']}),full_dir/'manifest.json')
    # Unsupervised support balancing, fixed before inspecting middle-day results.
    # Fit 8 clusters within each endpoint using training-only standardized RNA.
    x=np.asarray(tr.X,dtype=float)
    center=x.mean(0);scale=np.maximum(x.std(0),1e-3)
    xx=(x-center)/scale
    weights=np.zeros(len(x),dtype=float)
    clusters=np.zeros(len(x),dtype=int)
    labels=tr.obs.Group.to_numpy()
    support=[]
    for day in [0,3]:
        ids=np.flatnonzero(labels==day)
        km=KMeans(n_clusters=8,random_state=42,n_init=10).fit(xx[ids])
        cc=km.labels_;counts=np.bincount(cc,minlength=8)
        weights[ids]=1./(16*counts[cc])
        clusters[ids]=cc+8*(day==3)
        for c,n in enumerate(counts):
            support.append(dict(day=day,cluster=c,cells=int(n),sampling_mass=1/16))
    assert abs(weights.sum()-1)<1e-12
    pd.DataFrame(dict(row=np.arange(len(x)),cell_id=tr.obs_names,day=labels,
        cluster=clusters,probability=weights)).to_csv(OUT/'support_sampling.csv',index=False)
    write_json(dict(clusters=support,k_per_endpoint=8,features='training endpoint RNA only',
        middle_days_used=False,training_data_sha256=sha(ROOT/'data/round2/source/train.h5ad')),
        OUT/'support_sampling.json')
    # Data hashes and fixed protocol are written before any training is launched.
    write_json(dict(variants=VARIANTS,seed=42,updates=100000,batch_size=64,lr=1e-4,
        criterion='Endpoint validation gene-wise Wasserstein only',checkpoint='final100k',
        gene_panel=tr.var_names.tolist(),source_hashes={p:sha(ROOT/'data/round2/source'/f'{p}.h5ad') for p in ['train','val','test']},
        full_hashes={p:sha(full_dir/f'{p}.h5ad') for p in ['train','val','test']},
        no_seed_search=True,no_intermediate_training=True,
        historical_target_reuse='Development/reproduction diagnostics; middle days have been inspected in previous rounds, not an independent final test'),OUT/'frozen_protocol.json')
    print('PREPARED',expanded.shape,support,flush=True)


class ObservedLoop(previous.ObservedLoop):
    def log_step(self):
        TrainLoop.log_step(self)
        n=self.step+1
        if n%1000:return
        loss=float(torch.stack([x.detach() for x in self.loss_list]).mean())
        self.loss_list.clear()
        row=dict(step=n,loss=loss,steps_per_second=1000/(time.monotonic()-self.last_time))
        self.last_time=time.monotonic()
        if n%10000==0 or n==self.target:
            row['val_loss']=previous.validation(self.model,self.diffusion,self.val,self.config)
            if row['val_loss']<self.best:
                self.best=row['val_loss']
                atomic_save(dict(model=self.model.state_dict(),step=n,config=self.config),self.dest/'best.pt')
        with (self.dest/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write_json(dict(step=n,target=self.target,status='training',pid=os.getpid()),self.dest/'status.json')
        print(json.dumps(row),flush=True)


def train(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    seed_all(42)
    config=dict(VARIANTS[args.variant],variant=args.variant,seed=42,steps=args.steps,
        transform='identity',width=2048,latent_noise=0.,wd=0.,genes=203,
        batch_size=64,lr=1e-4,math='IEEE FP32',target='epsilon',
        optimizer='native default AdamW',diffusion_steps=1000,
        source_sha256={p:sha(ROOT/'upstream/Squidiff/Squidiff'/p) for p in ['MLPModel.py','diffusion.py','train_util.py']},
        adapter_sha256={p:sha(ROOT/'scripts'/p) for p in ['round4.py','round2.py','common.py']})
    tr,va,_=load_data(config)
    config['genes']=tr.n_vars
    config['train_data_sha256']=sha(folder(config)/'train.h5ad')
    dest=OUT/f'{args.variant}_s42'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'status.json').exists() and not args.resume:
        raise RuntimeError('Existing run; refusing overwrite')
    if args.resume:
        old=json.loads((dest/'config.json').read_text())
        for key in config:
            assert old[key]==config[key],key
    m,d=create(config);m.cuda().train()
    config['parameters']=sum(p.numel() for p in m.parameters())
    dataset=previous.Cells(tr,config)
    if config['sampler']=='endpoint_cluster_balanced':
        tab=pd.read_csv(OUT/'support_sampling.csv')
        assert np.array_equal(tab.cell_id.to_numpy(),tr.obs_names.to_numpy())
        sampler=WeightedRandomSampler(torch.tensor(tab.probability.to_numpy(),dtype=torch.double),num_samples=len(tr),replacement=True)
        loader=DataLoader(dataset,batch_size=64,sampler=sampler)
        config['sampling_weights_sha256']=sha(OUT/'support_sampling.csv')
    else:loader=DataLoader(dataset,batch_size=64,shuffle=True)
    logger.configure(str(dest/'native_logs'),format_strs=['json'])
    loop=ObservedLoop(model=m,diffusion=d,data=loader,batch_size=64,microbatch=-1,
        lr=1e-4,ema_rate='0.9999',log_interval=1000,save_interval=10000,
        resume_checkpoint=str(dest),lr_anneal_steps=args.steps,weight_decay=0.)
    loop.dest=dest;loop.config=config;loop.val=va;loop.target=args.steps
    loop.best=float('inf');loop.last_time=time.monotonic()
    loop.mp_trainer._compute_norms=lambda grad_scale=1.:(0.,0.)
    config['logging_norms_disabled']=True
    config['cuda_mps_pipe_directory']=os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
    if args.resume:
        s=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(s['model']);loop.opt.load_state_dict(s['optimizer'])
        loop.ema_params=[[v.cuda() for v in loop.mp_trainer.state_dict_to_master_params(s['ema'])]]
        loop.step=s['step'];loop.best=s['best_val'];previous.restore(s['rng'])
        del s
    write_json(config,dest/'config.json')
    start=time.monotonic()
    loop.run_loop()
    assert loop.step==args.steps
    atomic_save(dict(model=m.state_dict(),step=loop.step,config=config),dest/'final.pt')
    write_json(dict(step=loop.step,target=args.steps,status='trained',pid=os.getpid(),
        seconds=time.monotonic()-start,best_val=loop.best),dest/'status.json')
    print('TRAINED',args.variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','train'])
    p.add_argument('--variant',choices=list(VARIANTS))
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action=='prepare':prepare()
    else:train(args)
