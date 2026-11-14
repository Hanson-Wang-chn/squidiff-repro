"""Read-only view of durable progress, useful after terminal/session interruption."""
import json
import shutil
import psutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in sorted((ROOT/'results').glob('*/*/status.json')):
    if (p.parent/'EXCLUDED.json').exists():
        print(str(p.parent.relative_to(ROOT)),'EXCLUDED; see EXCLUDED.json')
        continue
    print(str(p.parent.relative_to(ROOT)),json.loads(p.read_text()))
    log=p.parent/'train.jsonl'
    if log.exists():
        rows=log.read_text().splitlines()
        print('  latest:',rows[-1] if rows else '(empty)')
    for ev in sorted(p.parent.glob('evaluation_*.json')):
        print('  evaluation:',ev.name,json.loads(ev.read_text()).get('metrics'))
for name in ['pipeline','research_parallel','research_stage3','research_stage4','research_stage5']:
    pid=ROOT/f'logs/{name}.pid'
    if pid.exists():
        value=int(pid.read_text())
        try:alive=psutil.pid_exists(value) and psutil.Process(value).status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:alive=False
        print(name,'supervisor PID',value,'alive',alive)
for name in ['pipeline','research_parallel','research_followup']:
    done=ROOT/f'logs/{name}_complete.json'
    if done.exists():print(name+':',json.loads(done.read_text()))
jobs=[]
for name in [f'research_stage{i}' for i in range(1,6)]:
    plan=ROOT/f'configs/{name}.json'
    if plan.exists():jobs.extend(json.loads(plan.read_text()))
if jobs:
    trained=evaluated=0
    for job in jobs:
        folder=ROOT/'results'/job['dataset']/job['run']
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        trained+=status.get('status')=='trained' and status.get('step',0)>=job.get('steps',100000)
        evaluated+=(folder/'research_eval/raw_controls/complete.json').exists()
    print(f'Research planned jobs: trained {trained}/{len(jobs)}, evaluated {evaluated}/{len(jobs)}')
print('Workspace free GiB:',round(shutil.disk_usage(ROOT).free/1024**3,2))
