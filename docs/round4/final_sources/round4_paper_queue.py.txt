"""Run the supplementary cell-wise formula after the other native ablations."""
import os
import subprocess
import sys
import time
from common import ROOT,write_json

def main():
    log=ROOT/'logs/round4'
    gate=ROOT/'results/round4/inference_controls/finalclip/eval_final_1000/complete.json'
    while not gate.exists():time.sleep(2)
    env=os.environ.copy()
    # A normal context makes this independent of the training daemon's shutdown.
    env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_LOG_DIRECTORY',None)
    command=[sys.executable,str(ROOT/'scripts/round4_evaluate.py'),'--model','paper_cellnoise','--oracle']
    start=time.time()
    with (log/'paper_cellnoise_evaluate.log').open('a',buffering=1) as f:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    write_json(dict(start=start,end=time.time(),command=command,exit_code=result.returncode),log/'paper_evaluation_complete.json')
    if result.returncode:raise RuntimeError(f'Cell-wise formula evaluation exited {result.returncode}')
    print('PAPER_FORMULA_EVALUATED',flush=True)

if __name__=='__main__':main()
