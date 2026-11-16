"""Complete the frozen postprocessing without waiting for interactive polling."""
import json
import os
import subprocess
import sys
import time
from common import ROOT,write_json

def main():
    log=ROOT/'logs/round4';start=time.time()
    for path in [log/'evaluation_complete.json',log/'paper_evaluation_complete.json']:
        while not path.exists():time.sleep(2)
    assert json.loads((log/'paper_evaluation_complete.json').read_text())['exit_code']==0
    env=os.environ.copy();env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_LOG_DIRECTORY',None)
    jobs=[]
    for script in ['round4_endpoint_tail.py','round4_integrity.py','round4_report.py']:
        command=[sys.executable,str(ROOT/'scripts'/script)]
        with (log/(script.removesuffix('.py')+'.log')).open('a',buffering=1) as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(script=script,exit_code=result.returncode,command=command))
        write_json(dict(start=start,jobs=jobs),log/'finalization_progress.json')
        if result.returncode:raise RuntimeError(f'{script} exited {result.returncode}')
        print('POSTPROCESS_DONE',script,flush=True)
    write_json(dict(start=start,end=time.time(),jobs=jobs,all_success=True,
        remaining='Human review of plots, final interpretation and delivery manifest'),log/'numeric_finalization_complete.json')

if __name__=='__main__':main()
