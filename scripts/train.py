"""Single GPU training of the original Squidiff network with resumable state."""
import argparse
import copy
import fcntl
import json
import os
import random
import signal
import time
from pathlib import Path
import anndata as ad
import numpy as np
import torch
from common import ROOT, setup, seed_all, build_model, atomic_save, write_json, record, dense

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu']); torch.cuda.set_rng_state_all(s['cuda'])

@torch.no_grad()
def validate(model, diffusion, x, groups, precision):
    saved = rng_state()
    seed_all(12345)
    model.eval()
    scores = []
    for g in np.unique(groups):
        gx = x[torch.as_tensor(groups==g, device='cuda')]
        losses = []
        for chunk in gx.split(128):
            for timestep in [25,100,250,500,750,950]:
                t = torch.full((len(chunk),), timestep, device='cuda', dtype=torch.long)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
                    out = diffusion.training_losses(model, chunk, t,
                         model_kwargs={'drug_dose':None, 'control_feature':None})
                losses.extend(out['loss'].float().cpu().tolist())
        scores.append(float(np.mean(losses)))
    model.train()
    restore_rng(saved)
    return float(np.mean(scores)), dict(zip([str(x) for x in np.unique(groups)], scores))

def main(args):
    setup(); seed_all(args.seed)
    if args.math_mode=='ieee':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    dest = ROOT / 'results' / args.dataset / args.run
    dest.mkdir(parents=True, exist_ok=True)
    # Prevent two controllers from resuming and writing the same run concurrently.
    run_lock = open(dest / '.training.lock', 'a')
    fcntl.flock(run_lock, fcntl.LOCK_EX)
    data = ROOT / args.data_root / args.dataset
    train = ad.read_h5ad(data/'train.h5ad'); val = ad.read_h5ad(data/'val.h5ad')
    if args.validation_source=='heldout':
        assert set(train.obs_names).isdisjoint(val.obs_names)
    elif args.validation_source=='training':
        val=train[:min(800,train.n_obs)].copy()
    x = torch.from_numpy(dense(train)).cuda()
    vx = torch.from_numpy(dense(val)).cuda()
    model, diffusion = build_model(train.n_vars,model_variant=args.model_variant)
    model.cuda().train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    params, ep = list(model.parameters()), list(ema.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0., fused=args.optimizer=='fused')
    if args.sampler in ['native','balanced_native']:
        from torch.utils.data import DataLoader, TensorDataset
        from Squidiff.resample import UniformSampler
        native_loader=DataLoader(TensorDataset(torch.from_numpy(dense(train))),batch_size=args.batch_size,shuffle=True)
        native_sampler=UniformSampler(diffusion)
        if args.sampler=='balanced_native':
            cpu_features=torch.from_numpy(dense(train))
            group_values=train.obs.Group.to_numpy()
            cell_weights=torch.tensor([1./np.sum(group_values==g) for g in group_values],dtype=torch.float64)
    import hashlib
    data_hash=hashlib.sha256((data/'train.h5ad').read_bytes()).hexdigest()
    config = dict(vars(args), train_data_sha256=data_hash, genes=train.n_vars, model='upstream MLP 2048x3, latent 60',
                  diffusion_steps=1000, noise_schedule='linear 0.0001 to 0.02',
                  upstream_commit='abdfc27d84947dcccd745d1067c0840a41d32eb8',
                  train_cells=train.n_obs, val_cells=val.n_obs)
    config['cuda_mps_pipe_directory'] = os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
    if args.model_variant!='source':config['model']+='; explicit epsilon input residual and zero output initialization'
    step, best = 0, float('inf')
    checkpoint = dest/'latest.pt'
    resume_path = checkpoint if checkpoint.exists() else dest/'previous.pt'
    if resume_path.exists():
        s = torch.load(resume_path, map_location='cpu', weights_only=False)
        for k in ['seed','batch_size','lr','horizon','precision','genes']:
            assert config[k]==s['config'][k], (k,config[k],s['config'][k])
        for k,default in [('math_mode','tf32'),('data_root','data/processed')]:
            assert config[k]==s['config'].get(k,default),(k,config[k],s['config'].get(k,default))
        for k,default in [('lr_timing','standard'),('sampler','gpu'),('optimizer','fused'),('validation_source','heldout')]:
            assert config[k]==s['config'].get(k,default),(k,config[k],s['config'].get(k,default))
        assert args.model_variant==s['config'].get('model_variant','source')
        if 'train_data_sha256' in s['config']:
            assert config['train_data_sha256']==s['config']['train_data_sha256']
        model.load_state_dict(s['model']); ema.load_state_dict(s['ema'])
        opt.load_state_dict(s['optimizer']); step=s['step']; best=s['best_val']
        restore_rng(s['rng'])
        record(f'{args.dataset}/{args.run} 从第 {step} 步完整恢复。')
        del s
    else:
        record(f'开始训练 {args.dataset}/{args.run}：{train.n_obs} 个细胞、{train.n_vars} 个基因；作者原始 2048 隐层网络，batch={args.batch_size}，lr={args.lr}，{args.precision}，计划 {args.steps} 步。')
    write_json(config, dest/'config.json')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        print(f'received signal {signum}; saving at next completed optimizer step', flush=True)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def save():
        state = dict(step=step, config=config, model=model.state_dict(), ema=ema.state_dict(),
                     optimizer=opt.state_dict(), best_val=best, rng=rng_state())
        # Keep the last two complete optimizer states, never replace a valid file with a partial write.
        if checkpoint.exists():
            (dest/'previous.pt').unlink(missing_ok=True)
            os.link(checkpoint, dest/'previous.pt')
        atomic_save(state, checkpoint)
        atomic_save(model.state_dict(), dest/'model.pt')
        write_json(dict(step=step, target=args.steps, best_val=best,
                        status='interrupted' if stopped else 'running', pid=os.getpid(),
                        updated=time.strftime('%Y-%m-%d %H:%M:%S')), dest/'status.json')
    loss_sum = torch.zeros((),device='cuda')
    loss_count = 0
    started = time.monotonic(); report_at = step
    save()
    log = open(dest/'train.jsonl', 'a', buffering=1)
    try:
        while step < args.steps and not stopped:
            # The author creates a fresh shuffled loader each step. Sampling a fresh
            # permutation on GPU preserves that distribution without loader overhead.
            if args.sampler=='native':
                batch=next(iter(native_loader))[0].cuda()
                t,_=native_sampler.sample(len(batch),'cuda')
            elif args.sampler=='balanced_native':
                ids=torch.multinomial(cell_weights,args.batch_size,replacement=True)
                batch=cpu_features[ids].cuda()
                t,_=native_sampler.sample(len(batch),'cuda')
            else:
                ids = torch.randperm(len(x), device='cuda')[:args.batch_size]
                batch = x[ids]
                t = torch.randint(0,1000,(len(batch),),device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision=='bf16'):
                losses = diffusion.training_losses(model,batch,t,
                         model_kwargs={'drug_dose':None,'control_feature':None})
                loss = losses['loss'].mean()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(params,args.grad_clip)
            # Match the upstream linear learning-rate schedule.
            lr_index=max(0,step-1) if args.lr_timing=='source' else step
            lr = args.lr * max(0.,1.-lr_index/args.horizon)
            for pg in opt.param_groups: pg['lr']=lr
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(ep,params,1.-args.ema)
                for eb,b in zip(ema.buffers(),model.buffers()): eb.copy_(b)
                loss_sum += loss.detach()
                loss_count += 1
            step += 1
            if step % args.log_every == 0:
                value = float(loss_sum.item()/loss_count)
                if not np.isfinite(value): raise FloatingPointError('Non-finite training loss')
                row=dict(step=step,loss=value,loss_average_steps=loss_count,lr=lr,
                         steps_per_second=(step-report_at)/(time.monotonic()-started))
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                loss_sum.zero_();loss_count=0;started=time.monotonic();report_at=step
            if step % args.val_every == 0 or step == args.steps:
                value, by_group = validate(model,diffusion,vx,val.obs.Group.to_numpy(),args.precision)
                row=dict(step=step,val_loss=value,val_by_group=by_group)
                if value < best:
                    best=value
                    atomic_save(dict(model=model.state_dict(),step=step,config=config,val_loss=value), dest/'best.pt')
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save()
            if step in [10000,20000,50000,100000]:
                atomic_save(dict(model=model.state_dict(),step=step,config=config),dest/f'step_{step:06d}.pt')
    finally:
        save();log.close()
    status='interrupted' if stopped else 'trained'
    write_json(dict(step=step,target=args.steps,best_val=best,status=status,pid=os.getpid()),dest/'status.json')
    record(f'{args.dataset}/{args.run} {status}，已训练 {step} 步，验证集去噪 MSE={best:.6f}。')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('dataset', choices=['norman','ipsc']);p.add_argument('--run',default='v1')
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--horizon',type=int,default=100000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--ema',type=float,default=.9999)
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--grad-clip',type=float,default=0.)
    p.add_argument('--math-mode',choices=['tf32','ieee'],default='tf32')
    p.add_argument('--data-root',default='data/processed')
    p.add_argument('--sampler',choices=['gpu','native','balanced_native'],default='gpu')
    p.add_argument('--optimizer',choices=['fused','source'],default='fused')
    p.add_argument('--model-variant',choices=['source','epsilon_residual'],default='source')
    p.add_argument('--lr-timing',choices=['standard','source'],default='standard')
    p.add_argument('--validation-source',choices=['heldout','training'],default='heldout')
    p.add_argument('--save-every',type=int,default=2000);p.add_argument('--val-every',type=int,default=2000)
    p.add_argument('--log-every',type=int,default=100)
    main(p.parse_args())
