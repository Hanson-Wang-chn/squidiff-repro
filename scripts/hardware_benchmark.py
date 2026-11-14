"""Check independent-process throughput without changing the training algorithm."""
import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    import torch
    import anndata as ad
    from torch.utils.data import DataLoader, TensorDataset
    from common import setup, seed_all, build_model
    from Squidiff.resample import UniformSampler
    setup(); seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    a = ad.read_h5ad(ROOT / 'data/research/source/ipsc/train.h5ad')
    loader = DataLoader(TensorDataset(torch.from_numpy(a.X)), batch_size=64, shuffle=True)
    m, d = build_model(a.n_vars); m.cuda().train()
    ema = copy.deepcopy(m).requires_grad_(False)
    params, ep = list(m.parameters()), list(ema.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
    sampler = UniformSampler(d)
    last = []
    for step in range(args.steps + 30):
        if step == 30:
            torch.cuda.synchronize(); start = time.monotonic()
        x = next(iter(loader))[0].cuda(); t, _ = sampler.sample(len(x), 'cuda')
        opt.zero_grad(set_to_none=True)
        loss = d.training_losses(m, x, t, model_kwargs={'drug_dose':None,'control_feature':None})['loss'].mean()
        loss.backward(); opt.step()
        with torch.no_grad(): torch._foreach_lerp_(ep, params, .0001)
        if step >= args.steps + 25: last.append(float(loss))
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    digest = hashlib.sha256()
    for value in m.state_dict().values(): digest.update(value.detach().cpu().numpy().tobytes())
    Path(args.worker).write_text(json.dumps(dict(steps=args.steps, seconds=elapsed,
        steps_per_second=args.steps/elapsed, model_sha256=digest.hexdigest(), final_losses=last)))


def main(args):
    dest = ROOT / 'results/research/hardware_benchmark'
    dest.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/squidiff-mps', CUDA_MPS_LOG_DIRECTORY=str(ROOT/'logs/cuda_mps'))
    Path(env['CUDA_MPS_PIPE_DIRECTORY']).mkdir(exist_ok=True)
    Path(env['CUDA_MPS_LOG_DIRECTORY']).mkdir(exist_ok=True)
    os.kill(args.pause_pid, signal.SIGSTOP)
    out = {}
    try:
        r = subprocess.run(['nvidia-cuda-mps-control','-d'], env=env, capture_output=True, text=True)
        out['daemon_start'] = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        if r.returncode: return
        for n in [1, 2, 3]:
            processes = []
            for i in range(n):
                path = dest/f'concurrency{n}_worker{i}.json'
                log = open(dest/f'concurrency{n}_worker{i}.log','w')
                p = subprocess.Popen([sys.executable,__file__,'--worker',str(path),'--steps',str(args.steps)],
                                     env=env, stdout=log,stderr=subprocess.STDOUT)
                processes.append((p,log,path))
            for p,log,path in processes:
                if p.wait(): raise RuntimeError(f'Benchmark failed; see {path.with_suffix(".log")}')
                log.close()
            rows = [json.loads(path.read_text()) for _,_,path in processes]
            out[str(n)] = dict(workers=rows, aggregate_steps_per_second=sum(r['steps_per_second'] for r in rows))
            print(n,out[str(n)]['aggregate_steps_per_second'],flush=True)
    finally:
        os.kill(args.pause_pid,signal.SIGCONT)
        (dest/'summary.json').write_text(json.dumps(out,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--steps',type=int,default=300)
    p.add_argument('--pause-pid',type=int)
    args=p.parse_args()
    worker(args) if args.worker else main(args)
