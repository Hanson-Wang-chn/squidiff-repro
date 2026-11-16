"""Run the four frozen jobs under a private MPS daemon; check every exit."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import ROOT, write_json
from round4 import VARIANTS, sha

def main():
    log=ROOT/'logs/round4';log.mkdir(parents=True,exist_ok=True)
    assert json.loads((ROOT/'results/round4/preflight.json').read_text())['native_update_max_error']==0
    assert shutil.disk_usage(ROOT).free>20*1024**3
    env=os.environ.copy()
    env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/squidiff-round4-mps',
        CUDA_MPS_LOG_DIRECTORY=str(log/'mps'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for name in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']:Path(env[name]).mkdir(parents=True,exist_ok=True)
    subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
    workers=[];started=time.time()
    try:
        for name in VARIANTS:
            f=(log/f'{name}_train.log').open('a',buffering=1)
            command=[sys.executable,str(ROOT/'scripts/round4.py'),'train','--variant',name]
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            workers.append((p,f,name));print('START',name,p.pid,flush=True)
        write_json(dict(start=started,commands={v:[sys.executable,'scripts/round4.py','train','--variant',v] for p,f,v in workers},
            pids={v:p.pid for p,f,v in workers},seed=42,steps_each=100000,
            frozen_protocol_sha256=sha(ROOT/'results/round4/frozen_protocol.json'),
            env={k:env[k] for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']}),log/'queue.json')
        remaining=list(workers)
        while remaining:
            for p,f,v in remaining[:]:
                if p.poll() is not None:
                    if p.returncode:raise RuntimeError(f'{v} exited {p.returncode}')
                    remaining.remove((p,f,v));f.close();print('DONE',v,flush=True)
            time.sleep(2)
        write_json(dict(start=started,end=time.time(),exit_codes={v:p.returncode for p,f,v in workers},
            updates=400000,all_success=True),log/'training_complete.json')
    finally:
        for p,f,v in workers:
            if p.poll() is None:p.terminate();p.wait()
            if not f.closed:f.close()
        shutdown=subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env)
        write_json(dict(time=time.time(),pipe=env['CUDA_MPS_PIPE_DIRECTORY'],exit_code=shutdown.returncode),log/'mps_shutdown.json')

if __name__=='__main__':main()
