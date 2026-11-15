"""Avoid resampling an exactly identical native early-stop checkpoint twice."""
import json
from pathlib import Path
import shutil
import time
import torch
import pandas as pd

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/'results/round2/native_s42'
while True:
    s=json.loads((dest/'status.json').read_text())
    if s.get('status')=='trained':break
    time.sleep(5)
best=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
early=torch.load(dest/'step_010000.pt',map_location='cpu',weights_only=False)
equal=best['step']==early['step'] and all(torch.equal(v,early['model'][k]) for k,v in best['model'].items())
source=dest/'eval_step_010000_1000'
target=dest/'eval_best_1000'
if equal and (source/'complete.json').exists() and not target.exists():
    shutil.copytree(source,target)
    f=pd.read_csv(target/'metrics.csv');f['checkpoint']='best.pt';f.to_csv(target/'metrics.csv',index=False)
    done=json.loads((target/'complete.json').read_text())
    done['args']['checkpoint']='best.pt'
    done['exact_checkpoint_reuse']=dict(source=str(source.relative_to(ROOT)),
        all_parameters_and_buffers_bitwise_equal=True,new_sampling_performed=False)
    (target/'complete.json').write_text(json.dumps(done,indent=2))
    (target/'reuse.json').write_text(json.dumps(done['exact_checkpoint_reuse'],indent=2))
print('EXACT_CHECKPOINT_REUSE',equal,flush=True)
