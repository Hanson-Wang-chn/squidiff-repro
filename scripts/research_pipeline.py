"""Durable single-GPU experiment queue with explicit hypotheses and disk retention."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable
def call(cmd,name):
    with open(ROOT/'logs'/f'{name}.log','a',buffering=1) as f:
        print(time.strftime('%Y-%m-%d %H:%M:%S'),cmd,flush=True)
        p=subprocess.Popen([PY,'-u',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        (ROOT/'logs/research_child.json').write_text(json.dumps({'pid':p.pid,'command':cmd}))
        if p.wait():raise RuntimeError(f'{cmd} failed; see logs/{name}.log')
def compact(ds,run):
    dest=ROOT/'results'/ds/run;status=json.loads((dest/'status.json').read_text())
    if status['status']!='trained':return
    removed=[]
    for name in ['previous.pt','model.pt','step_010000.pt','step_020000.pt']:
        path=dest/name
        if path.exists():
            removed.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size});path.unlink()
    with open(ROOT/'logs/checkpoint_retention.jsonl','a') as f:
        f.write(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'removed':removed,
                            'kept':'latest full optimizer/RNG state, final/50k model weights and best model'})+'\n')
def main(args):
    if args.wait_pid:
        while psutil.pid_exists(args.wait_pid):
            try:
                if psutil.Process(args.wait_pid).status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(5)
    if args.prerequisite:
        for j in json.loads((ROOT/args.prerequisite).read_text()):
            p=ROOT/'results'/j['dataset']/j['run']
            status=json.loads((p/'status.json').read_text())
            assert status.get('step',0)>=j.get('steps',100000),(j,status)
            assert (p/'research_eval/raw_controls/complete.json').exists(),j
    jobs=json.loads((ROOT/args.plan).read_text())
    for job in jobs:
        ds,run=job['dataset'],job['run'];dest=ROOT/'results'/ds/run
        if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; retention review required')
        status=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {}
        target=job.get('steps',100000)
        train=['scripts/train.py',ds,'--run',run,'--precision','fp32','--steps',str(target),
               '--seed',str(job.get('seed',42)),'--data-root',job.get('data_root','data/processed'),
               '--math-mode',job.get('math_mode','tf32'),'--sampler',job.get('sampler','gpu'),
               '--optimizer',job.get('optimizer','fused'),'--lr-timing',job.get('lr_timing','standard'),
               '--validation-source',job.get('validation_source','heldout'),
               '--model-variant',job.get('model_variant','source'),
               '--val-every','10000','--save-every','5000']
        if status.get('step',0)<target:
            for attempt in range(2):
                try:call(train,f'train_{ds}_{run}');break
                except RuntimeError:
                    if attempt:raise
                    time.sleep(5)
        ev=['scripts/research_evaluate.py',ds,'--run',run,'--data-root',job.get('data_root','data/processed'),
            '--tag','raw_controls','--modes',job.get('modes','centroid,cell_shift,oracle'),
            '--sigmas','0,0.5,0.7','--inference-math','ieee']
        if not (dest/'research_eval/raw_controls/complete.json').exists():call(ev,f'research_eval_{ds}_{run}')
        if job.get('positive_control'):
            import pandas as pd
            m=pd.read_csv(dest/'research_eval/raw_controls/metrics.csv')
            r=m[(m['mode']=='oracle')&(m.target=='day1')].iloc[0]
            if (r.predicted_T_mean/r.true_T_mean<.7 or r.r2<.9) and target<100000:
                train[train.index('--steps')+1]='100000'
                call(train,f'train_{ds}_{run}')
                call(ev,f'research_eval_{ds}_{run}_100k')
        compact(ds,run)
        result={'completed_job':job,'time':time.strftime('%Y-%m-%d %H:%M:%S'),
                'free_gib':shutil.disk_usage(ROOT).free/1024**3}
        with open(ROOT/'logs/research_completed.jsonl','a') as f:f.write(json.dumps(result)+'\n')
    print('Research queue complete',args.plan,flush=True)
    (ROOT/'logs'/f'{Path(args.plan).stem}_complete.json').write_text(json.dumps({'plan':args.plan,'time':time.strftime('%Y-%m-%d %H:%M:%S')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--wait-pid',type=int);p.add_argument('--prerequisite');main(p.parse_args())
