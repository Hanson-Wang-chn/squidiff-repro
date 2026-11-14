"""Three isolated training queues sharing the A100 through a private CUDA MPS server."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    lock = open(ROOT / 'logs/.research_parallel.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/squidiff-mps',
               CUDA_MPS_LOG_DIRECTORY=str(ROOT / 'logs/cuda_mps'),
               OPENBLAS_NUM_THREADS='4')
    Path(env['CUDA_MPS_PIPE_DIRECTORY']).mkdir(parents=True,exist_ok=True)
    Path(env['CUDA_MPS_LOG_DIRECTORY']).mkdir(parents=True,exist_ok=True)
    controller_pid = Path(env['CUDA_MPS_PIPE_DIRECTORY']) / 'nvidia-cuda-mps-control.pid'
    if not controller_pid.exists() or not psutil.pid_exists(int(controller_pid.read_text())):
        subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True)
    jobs = []
    for plan in [f'configs/research_stage{i}.json' for i in range(1,6)]:
        if (ROOT / plan).exists():jobs.extend(json.loads((ROOT / plan).read_text()))
    children = []
    for i in range(args.workers):
        plan = f'configs/research_worker{i}.json'
        (ROOT / plan).write_text(json.dumps(jobs[i::args.workers], indent=2))
        log = open(ROOT / f'logs/research_worker{i}.log', 'a', buffering=1)
        child = subprocess.Popen([sys.executable, '-u', 'scripts/research_pipeline.py', '--plan', plan],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((child, log, plan))
    (ROOT / 'logs/research_parallel.json').write_text(json.dumps(dict(
        pid=os.getpid(), workers=[dict(pid=p.pid, plan=plan) for p, _, plan in children],
        env={k:env[k] for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']},
        started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=2))
    codes = []
    for child, log, _ in children:
        codes.append(child.wait()); log.close()
    assert codes == [0] * args.workers, codes
    with open(ROOT / 'logs/research_followup.log', 'a') as log:
        subprocess.run([sys.executable,'-u','scripts/research_followup.py'],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    (ROOT / 'logs/research_parallel_complete.json').write_text(json.dumps(dict(
        finished=time.strftime('%Y-%m-%d %H:%M:%S'), training_jobs=len(jobs), status='complete')))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);main(p.parse_args())
