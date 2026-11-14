"""Persistent sequential training/evaluation supervisor for the single local GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable

def run(command,logfile):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),command,flush=True)
    with open(ROOT/'logs'/logfile,'a',buffering=1) as log:
        result=subprocess.run([PY,'-u',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{command} exited {result.returncode}; inspect logs/{logfile}')

def main(args):
    if args.wait_pid:
        while psutil.pid_exists(args.wait_pid):
            try:
                if psutil.Process(args.wait_pid).status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(5)
    for ds in ['norman','ipsc']:
        if not (ROOT/'data/processed'/ds/'manifest.json').exists():
            raise RuntimeError(f'{ds} preparation is not complete')
        dest=ROOT/'results'/ds/'v1'
        state=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {}
        if state.get('step',0)<100000:
            # Retry an interrupted training command using its fully saved state.
            for attempt in range(3):
                try:
                    run(['scripts/train.py',ds,'--steps','100000'],f'train_{ds}_v1.log')
                    final_state=json.loads((dest/'status.json').read_text())
                    if final_state.get('step',0)<100000:
                        raise RuntimeError(f'{ds}: training stopped before 100000 steps')
                    break
                except RuntimeError:
                    if attempt==2:raise
                    time.sleep(5)
        evalpath=dest/'evaluation_1000steps.json'
        if not evalpath.exists():
            run(['scripts/evaluate.py',ds,'--sample-steps','1000','--repeats',str(10 if ds=='norman' else 6)],
                f'evaluate_{ds}_v1.log')
    (ROOT/'logs/pipeline_complete.json').write_text(json.dumps(dict(status='finished',time=time.strftime('%Y-%m-%d %H:%M:%S'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);main(p.parse_args())
