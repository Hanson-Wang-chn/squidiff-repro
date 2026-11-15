"""Exercise the real CLI after interrupting the first atomic checkpoint."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import round2 as r

env=os.environ.copy()
env['CUDA_MPS_PIPE_DIRECTORY']='/tmp/squidiff-round2-mps'
env['CUDA_MPS_LOG_DIRECTORY']=str(r.ROOT/'logs/round2/mps')
dest=r.OUT/'compact_s7777'
assert not dest.exists()
cmd=[sys.executable,str(r.ROOT/'scripts/round2.py'),'train','--variant','compact','--seed','7777','--steps','2000']
with open(r.ROOT/'logs/round2/cli_interrupted.log','w') as f:
    p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=r.ROOT,env=env)
    while not (dest/'latest.pt').exists():
        if p.poll() is not None:raise RuntimeError('CLI exited before checkpoint')
        time.sleep(.1)
    p.terminate();p.wait()
shutil.move(str(dest),str(r.OUT/'cli_interrupted'))
with open(r.ROOT/'logs/round2/cli_reference.log','w') as f:
    subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=r.ROOT,env=env,check=True)
reference=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
shutil.move(str(dest),str(r.OUT/'cli_reference'))
shutil.move(str(r.OUT/'cli_interrupted'),str(dest))
with open(r.ROOT/'logs/round2/cli_resumed.log','w') as f:
    subprocess.run(cmd+['--resume'],stdout=f,stderr=subprocess.STDOUT,cwd=r.ROOT,env=env,check=True)
resumed=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
err=max(float((reference['model'][k]-resumed['model'][k]).abs().max()) for k in reference['model'])
assert err==0,err
r.write_json(dict(steps=2000,seed=7777,full_cli_resume_max_parameter_buffer_error=err,
                 official_experiment=False),r.OUT/'cli_resume_audit.json')
print((r.OUT/'cli_resume_audit.json').read_text())
