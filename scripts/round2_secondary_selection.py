"""Post-hoc endpoint-MSE selection; never select by intermediate-day scores.

Uses the already measured endpoint candidates. Reuses identical draws whenever
possible and samples only previously unevaluated candidates (two target days).
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd
import round2 as r

while True:
    p=r.ROOT/'logs/round2/evaluation_queue.log'
    if p.exists() and 'EVALUATION_QUEUE_COMPLETE' in p.read_text():break
    time.sleep(5)
rows=[]
for variant,signed in [(v,False) for v in r.VARIANTS]+[('zscore',True)]:
    source=r.OUT/f'{variant}_s42'/('eval_final_1000'+('_signed' if signed else ''))
    candidates=pd.read_csv(source/'candidates.csv')
    winner=candidates.loc[candidates.mse.idxmin()]
    main=pd.read_csv(source/'metrics.csv')
    selection=json.loads((source/'selection.json').read_text())['selected']
    available=None;reused=True
    if winner['mode']==selection['mode'] and math.isclose(winner.sigma,selection['sigma'],abs_tol=1e-10):
        available=main[main.method=='selected']
    elif winner['mode'] in set(main.method) and winner['mode'] in ['fixed','paired','empirical','ot']:
        available=main[main.method==winner['mode']]
    if available is None:
        reused=False
        cmd=[sys.executable,str(r.ROOT/'scripts/round2.py'),'evaluate','--variant',variant,
             '--sample-steps','1000','--criterion','mse','--reuse-candidates','--selected-only']
        if signed:cmd+=['--signed']
        log_name=f"logs/round2/{variant}_secondary_mse{('_signed' if signed else '')}.log"
        with open(r.ROOT/log_name,'w') as f:
            subprocess.run(cmd,cwd=r.ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        available=pd.read_csv(Path(str(source)+'_mse')/'metrics.csv')
    for day in [1,2]:
        a=main[(main.method=='selected')&(main.day==day)].iloc[0]
        b=available[available.day==day].iloc[0]
        rows.append(dict(variant=variant,signed=signed,day=day,mse_selected_mode=winner['mode'],sigma=float(winner.sigma),
            reused_existing_draw=reused,primary_r2=float(a.r2),mse_selected_r2=float(b.r2),
            primary_T=float(a.T_pred),mse_selected_T=float(b.T_pred),mse_selected_mse_gain=float(b.mse_gain),
            true_transients=int(b.transient_actual),true_positives=int(b.transient_tp),false_positives=int(b.transient_called-b.transient_tp)))
    pd.DataFrame(rows).to_csv(r.OUT/'secondary_endpoint_mse.csv',index=False)
    print(variant,signed,winner.to_dict(),flush=True)
subprocess.run([sys.executable,str(r.ROOT/'scripts/round2_summary.py')],cwd=r.ROOT,check=True)
print('SECONDARY_SELECTION_COMPLETE',flush=True)
