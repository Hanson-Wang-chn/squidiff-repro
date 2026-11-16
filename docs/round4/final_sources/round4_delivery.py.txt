"""Seal the reviewed local delivery; do not stage, commit, or publish anything."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pandas as pd
from common import ROOT,write_json
from round4 import OUT,VARIANTS,sha

def main():
    checks=json.loads((OUT/'final_integrity.json').read_text())
    assert checks['total_updates']==400000 and checks['total_predictions']==100 and checks['supplementary_predictions']==10
    assert json.loads((ROOT/'logs/round4/numeric_finalization_complete.json').read_text())['all_success']
    report=ROOT/'report-4.md';body=report.read_text()
    for stale in ['状态：正在运行','待本轮受控结果完成后判断','尚待四个正式训练']:assert stale not in body
    sources=sorted(set((ROOT/'scripts').glob('round4*.py'))|{ROOT/'scripts/train_source_safe.py'})
    snapshots=ROOT/'docs/round4/final_sources';snapshots.mkdir(exist_ok=True)
    entries=[]
    for p in sources:
        dest=snapshots/(p.name+'.txt');dest.write_bytes(p.read_bytes())
        entries.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),snapshot=str(dest.relative_to(ROOT))))
    write_json(dict(note='Final reviewed sources for replay. Actual training hashes remain bound in each model config and training_source_manifest.json.',files=entries),ROOT/'docs/round4/source_manifest.json')
    f=pd.read_csv(OUT/'selected_metrics.csv');day1=f[f.day==1]
    new=day1[day1.model.isin(VARIANTS)].copy()
    assert len(new)==4 and (new.transient_tp==0).all() and float(new.T_pred.max())<.281
    now=datetime.now().isoformat(timespec='seconds')
    assessment=dict(status='complete',completed_local=now,new_formal_training_runs=4,
        new_formal_optimizer_updates=400000,engineering_cli_smoke_updates=2,
        formal_training_seed=42,seed_search=False,primary_cells=600,ddim_steps=1000,
        primary_prediction_files=100,capacity_diagnostic_files=10,genes=203,
        issue_entries_read=16,actual_issues_read=15,pr_entries_read=1,comments_read=23,
        day1_reproduced=False,conclusion='Accurate day1 transient state was not reproduced; stop the current public-protocol reproduction route. This is not a proof of mathematical impossibility.',
        day1_metrics=new[['model','r2','baseline_r2','gain_percent','common_gain_percent','T_pred','T_above3','transient_tp','transient_fp','energy_distance']].to_dict('records'),
        actual_day1_T=6.2500641773144405,actual_day1_transient_genes=13,
        source_train_rows=2400,source_unique_training_cells=2237,fullsource_train_cells=16580,
        limitations=['One training per new variant','Historically transductive source203 HVG panel','Repeatedly inspected development targets','No donor holdout','Exact author Fig2 checkpoint/split/gene table unavailable','Expanded-data model underfits high-T training tail'],
        numerical_checks_passed=True,historical_archive_files_unchanged=checks['historical_archive_files_verified'],
        retired_optimizer_snapshots=15,retired_bytes=12673484910,free_disk_bytes=shutil.disk_usage(ROOT).free,
        figures_reviewed=['controlled_results.png','marker_trajectories.png','training_curves.png'],
        review_actor='Codex',report_sha256=sha(report))
    write_json(assessment,OUT/'assessment.json')
    write_json(dict(completed=now,actor='Codex',plots_reviewed=True,interpretation_written=True,
        all_report_placeholders_resolved=True,numeric_stage_remaining_tasks_completed=True),ROOT/'logs/round4/review_complete.json')
    # Large local artifacts are indexed separately, not included as portable files.
    large=[]
    for p in sorted(OUT.rglob('*.pt')):
        large.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),role='local checkpoint'))
    for p in sorted((ROOT/'data/round4').rglob('*.h5ad')):
        large.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),role='local data'))
    paths={report,*sources}
    for d in [ROOT/'docs/round4',ROOT/'logs/round4',OUT,ROOT/'data/round4']:
        for p in d.rglob('*'):
            if not p.is_file() or p.suffix in ['.pt','.h5ad','.pyc','.tmp']:continue
            if p==OUT/'delivery_manifest.json':continue
            paths.add(p)
    portable=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    delivery=dict(completed=now,note='Local round4 delivery, separate from the immutable historical archive; manifest excludes itself.',
        files=portable,file_count=len(portable),total_bytes=sum(p['bytes'] for p in portable),large_local_artifacts=large)
    write_json(delivery,OUT/'delivery_manifest.json')
    for target in re.findall(r'\]\(([^)]+)\)',body):
        if target.startswith(('https://','http://','#')):continue
        assert (ROOT/target).exists(),target
    for entry in portable:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    print('DELIVERY_OK',len(portable),'portable files',len(large),'local large artifacts',
        'free_GiB',shutil.disk_usage(ROOT).free/1024**3,flush=True)

if __name__=='__main__':main()
