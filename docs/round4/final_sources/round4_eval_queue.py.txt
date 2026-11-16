"""Frozen native inference ablations, followed by completed new model evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,write_json
from round4 import VARIANTS

def main():
    log=ROOT/'logs/round4'
    env=os.environ.copy()
    env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/squidiff-round4-mps',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    jobs=[];start=time.time()
    native=ROOT/'results/round4/inference_controls/native/eval_final_1000/complete.json'
    while not native.exists():time.sleep(2)
    for name in ['bncal','ema','emacal','finalclip']:
        command=[sys.executable,str(ROOT/'scripts/round4_evaluate.py'),'--model',name,
            '--oracle','--use-native-selection']
        with (log/f'{name}_evaluate.log').open('a',buffering=1) as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(model=name,exit_code=result.returncode,command=command))
        write_json(dict(start=start,jobs=jobs),log/'evaluation_progress.json')
        if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}')
        print('EVALUATED',name,flush=True)
    # The training supervisor owns MPS; subsequent evaluations use the normal
    # CUDA context after its workers and daemon have exited.
    while not (log/'training_complete.json').exists():time.sleep(2)
    while not (log/'mps_shutdown.json').exists():time.sleep(1)
    env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_LOG_DIRECTORY',None)
    for name in VARIANTS:
        command=[sys.executable,str(ROOT/'scripts/round4_evaluate.py'),'--model',name,'--oracle']
        with (log/f'{name}_evaluate.log').open('a',buffering=1) as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(model=name,exit_code=result.returncode,command=command))
        write_json(dict(start=start,jobs=jobs),log/'evaluation_progress.json')
        if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}')
        print('EVALUATED',name,flush=True)
    write_json(dict(start=start,end=time.time(),jobs=jobs,all_success=True),log/'evaluation_complete.json')

if __name__=='__main__':main()
