"""Private GPU queue: finish author verification, then three native training jobs.

No external communication or global GPU setting changes. MPS uses a private pipe.
Every worker exit is checked; evaluation never proceeds from a failed training.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import ROOT, write_json

LOG=ROOT/'logs/round3'
OUT=ROOT/'results/round3'


def main():
    LOG.mkdir(exist_ok=True,parents=True)
    while not (OUT/'author/eval_1000/complete.json').exists():
        time.sleep(2)
    assert shutil.disk_usage(ROOT).free > 12*1024**3
    env=os.environ.copy()
    env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/squidiff-round3-mps',
               CUDA_MPS_LOG_DIRECTORY=str(LOG/'mps'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for key in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']:
        Path(env[key]).mkdir(exist_ok=True,parents=True)
    subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
    workers=[]
    start=time.time()
    try:
        for v in ['xstart','minsnr','epsres']:
            f=open(LOG/f'{v}_train.log','a',buffering=1)
            cmd=[sys.executable,str(ROOT/'scripts/round3.py'),'train','--variant',v]
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            workers.append((p,f,v))
            print('TRAIN_START',v,p.pid,flush=True)
        write_json(dict(start=start,pids={v:p.pid for p,f,v in workers},
            steps_each=100000,variants=[v for p,f,v in workers],
            authorization='User explicitly requested retraining several variants',
            env={k:env[k] for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']}),
            LOG/'queue.json')
        running=list(workers)
        while running:
            for p,f,v in running[:]:
                if p.poll() is not None:
                    if p.returncode: raise RuntimeError(f'{v} training exited {p.returncode}')
                    running.remove((p,f,v));f.close()
                    print('TRAIN_DONE',v,flush=True)
            time.sleep(2)
        write_json(dict(start=start,end=time.time(),all_success=True),LOG/'training_complete.json')
        # First check the full reverse chain on each fixed final model. All seven
        # latent candidates are selected solely using the endpoint validation set.
        for v in ['xstart','minsnr','epsres']:
            cmd=[sys.executable,str(ROOT/'scripts/round3.py'),'evaluate','--variant',v,
                 '--sample-steps','1000','--cells','600']
            with open(LOG/f'{v}_evaluate.log','a',buffering=1) as f:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            print('EVALUATE_DONE',v,flush=True)
        write_json(dict(start=start,end=time.time(),all_success=True),LOG/'queue_complete.json')
    finally:
        for p,f,v in workers:
            if p.poll() is None: p.terminate();p.wait()
            if not f.closed:f.close()
        subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,check=False)
        write_json(dict(time=time.time(),pipe=env['CUDA_MPS_PIPE_DIRECTORY']),LOG/'mps_shutdown.json')


if __name__=='__main__':main()
