"""One-off local scheduler: preserve full states, then share the A100 with MPS.

PIDs must be supplied explicitly. Only these owned training workers are stopped.
No external service is used. No training/hyperparameter selection reads test data.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round2'
LOG=ROOT/'logs/round2'

def main(a):
    while not a.resume_queue:
        status=OUT/'native_s42/status.json'
        if status.exists() and json.loads(status.read_text())['step']>=20000:
            time.sleep(2)
            state=torch.load(OUT/'native_s42/latest.pt',map_location='cpu',weights_only=False)
            ready=state['step']>=20001
            del state
            if ready:break
        time.sleep(2)
    stopped=[]
    for pid in ([] if a.resume_queue else [a.native_pid,a.compact_pid,a.ln_pid]):
        os.kill(pid,signal.SIGTERM)
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
        stopped.append(pid)
    # Do not resume until all prior CUDA contexts are gone.
    for pid in stopped:
        for _ in range(100):
            if not Path(f'/proc/{pid}').exists():break
            time.sleep(.2)
        else:raise RuntimeError(f'PID {pid} did not terminate')
    env=os.environ.copy()
    env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/squidiff-round2-mps',
               CUDA_MPS_LOG_DIRECTORY=str(LOG/'mps'))
    for p in [env['CUDA_MPS_PIPE_DIRECTORY'],env['CUDA_MPS_LOG_DIRECTORY']]:
        Path(p).mkdir(exist_ok=True,parents=True)
    subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
    (LOG/'scheduling.json').write_text(json.dumps(dict(stopped_pids=stopped,
        reason='Avoid independent CUDA context switching; full RNG+optimizer resume',
        variants=['native','compact','ln','log2hvg','zscore'],env={k:v for k,v in env.items() if k.startswith('CUDA_MPS')}),indent=2))
    waiting=[(v,(OUT/f'{v}_s42/latest.pt').exists()) for v in ['native','compact','ln','log2hvg','zscore']]
    running=[]
    try:
        while waiting or running:
            while waiting and len(running)<4:
                variant,resume=waiting.pop(0)
                f=open(LOG/f'{variant}.log','a',buffering=1)
                cmd=[sys.executable,str(ROOT/'scripts/round2.py'),'train','--variant',variant]
                if resume:cmd+=['--resume']
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                running.append((p,f,variant))
                print('START',variant,p.pid,flush=True)
            for p,f,v in running[:]:
                if p.poll() is not None:
                    f.close();running.remove((p,f,v))
                    if p.returncode:raise RuntimeError(f'{v} exited {p.returncode}')
                    print('DONE',v,flush=True)
            time.sleep(2)
    finally:
        # On a failure, leave explicit run logs and stop owned workers cleanly.
        for p,f,v in running:
            p.terminate();p.wait();f.close()
        # Keep this private MPS daemon for the following evaluation queue.
        # The root task shuts it down after all GPU work and integrity checks.
        (LOG/'training_queue_finished.json').write_text(json.dumps(dict(
            time=time.time(),all_training_completed=not waiting and not running)))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['native','compact','ln']:p.add_argument(f'--{n}-pid',type=int)
    p.add_argument('--resume-queue',action='store_true')
    main(p.parse_args())
