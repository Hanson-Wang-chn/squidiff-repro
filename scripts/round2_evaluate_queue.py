"""Run fixed evaluation jobs after all five training runs have actually finished."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
variants=['native','ln','log2hvg','compact','zscore']
jobs=[]
for v in variants:
    jobs.append((v,'final.pt',v in ['native','compact'],False))
    if v in ['native','ln','compact']:
        jobs.append((v,'best.pt',False,False))
jobs.append(('zscore','final.pt',False,True))
env=os.environ.copy()
env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/squidiff-round2-mps',CUDA_MPS_LOG_DIRECTORY=str(ROOT/'logs/round2/mps'))
while jobs:
    ready=[]
    for job in jobs:
        variant=job[0]
        p=ROOT/f'results/round2/{variant}_s42/status.json'
        s=json.loads(p.read_text()) if p.exists() else {}
        if s.get('status')=='trained' and s.get('step')==100000:ready.append(job)
    if not ready:
        time.sleep(5);continue
    variant,checkpoint,full,signed=ready[0]
    jobs.remove(ready[0])
    cmd=[sys.executable,str(ROOT/'scripts/round2.py'),'evaluate','--variant',variant,
         '--checkpoint',checkpoint,'--sample-steps','1000']
    if full:cmd+=['--full']
    if signed:cmd+=['--signed']
    tag=f'{variant}_{Path(checkpoint).stem}_1000'+('_signed' if signed else '')
    folder=f'eval_{Path(checkpoint).stem}_1000'+('_signed' if signed else '')
    complete=ROOT/f'results/round2/{variant}_s42/{folder}/complete.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if old['args'].get('full',False)==full:
            print('EXISTING_COMPLETE',tag,flush=True);continue
    print('EVALUATE',tag,flush=True)
    with open(ROOT/f'logs/round2/{tag}.log','w') as f:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/round2_summary.py')],cwd=ROOT,check=True)
print('EVALUATION_QUEUE_COMPLETE',flush=True)
