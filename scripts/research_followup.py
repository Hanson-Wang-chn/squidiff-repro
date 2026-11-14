"""Run the pre-specified diagnostics after both single-GPU training queues finish."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import psutil
from research_pipeline import call

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    if args.wait_pid:
        while psutil.pid_exists(args.wait_pid):
            try:
                if psutil.Process(args.wait_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(5)
    jobs = []
    for plan in [f'configs/research_stage{i}.json' for i in range(1,6)]:
        if (ROOT / plan).exists():jobs.extend(json.loads((ROOT / plan).read_text()))
    for extra_name in ['research_stage3','research_stage4','research_stage5']:
        extra = ROOT / f'logs/{extra_name}.pid'
        if extra.exists():
            pid = int(extra.read_text())
            while psutil.pid_exists(pid):
                try:
                    proc=psutil.Process(pid)
                    if proc.status() == psutil.STATUS_ZOMBIE:break
                    command=' '.join(proc.cmdline())
                    if 'research_pipeline.py' not in command or f'{extra_name}.json' not in command:break
                except psutil.NoSuchProcess:break
                time.sleep(5)
    for job in jobs:
        dest = ROOT / 'results' / job['dataset'] / job['run']
        assert (dest / 'research_eval/raw_controls/complete.json').exists(), job
        assert json.loads((dest / 'status.json').read_text())['step'] == 100000, job
    # These intervals keep training data fixed; do not mislabel them as biological CIs.
    for job in jobs:
        ds, run, data = job['dataset'], job['run'], job.get('data_root', 'data/processed')
        done = ROOT / 'results/research/conditional_uncertainty' / ds / run / 'raw_controls.csv'
        if not done.exists():
            call(['scripts/conditional_uncertainty.py', ds, '--run', run, '--data-root', data],
                 f'uncertainty_{ds}_{run}')
    for seed in [42, 43, 44]:
        run = f'source_fp32_s{seed}'
        dest = ROOT / 'results/ipsc' / run
        if not (dest / 'sampling_nulls_source_repeats/complete.json').exists():
            call(['scripts/sampling_nulls.py', '--run', run, '--data-root', 'data/research/source',
                  '--sigmas', '0.7', '--repeats', '3', '--tag', 'source_repeats'],
                 f'nulls_{run}')
        # Diagnostic classifier is fitted to different real cell IDs; T is excluded.
        if not (dest / 'population_audit/raw_controls_without_T.csv').exists():
            call(['scripts/population_audit.py', '--run', run, '--data-root', 'data/research/source',
                  '--exclude-genes', 'T'], f'population_{run}')
        if not (ROOT / 'results/research/bootstrap_audit' / run / 'method.json').exists():
            call(['scripts/bootstrap_audit.py', '--run', run, '--tag', 'raw_controls',
                  '--data-root', 'data/research/source_full'], f'bootstrap_{run}')
        if not (ROOT / 'results/research/interaction_uncertainty' / run / 'summary.json').exists():
            call(['scripts/interaction_audit.py', '--run', run, '--tag', 'raw_controls',
                  '--data-root', 'data/research/source_full'], f'interaction_{run}')
        # Remove any effect of the first-600-control subsample in the cell-wise variant.
        all_controls=ROOT/'results/norman'/run/'research_eval/all_control_cells/complete.json'
        if not all_controls.exists():
            call(['scripts/research_evaluate.py','norman','--run',run,'--data-root','data/research/source_full',
                  '--tag','all_control_cells','--modes','cell_shift','--cells','7353',
                  '--inference-math','ieee'],f'all_controls_{run}')
        all_controls_interval=ROOT/'results/research/conditional_uncertainty/norman'/run/'all_control_cells.csv'
        if not all_controls_interval.exists():
            call(['scripts/conditional_uncertainty.py','norman','--run',run,'--data-root','data/research/source_full',
                  '--tag','all_control_cells'],f'uncertainty_all_controls_{run}')
    # Diagnose generation on known cells before using wide-model scores biologically.
    for run in ['source1000_fp32_s42','source1000_epsres_s42']:
        stability=ROOT/'results/norman'/run/'sampling_stability/complete.json'
        if not stability.exists():
            call(['scripts/sampling_stability.py','--run',run],f'sampling_stability_{run}')
    # EMA is a diagnostic alternative; the original raw final model remains primary.
    for run,data in [('balanced_fp32_s42','data/research/source_full'),
                     ('source1000_fp32_s42','data/research/source_1000_full'),
                     ('source1000_epsres_s42','data/research/source_1000_full')]:
        stability=ROOT/'results/norman'/run/'sampling_stability/metrics.csv'
        if stability.exists():
            import pandas as pd
            sm=pd.read_csv(stability)
            sm=sm[(sm.checkpoint=='latest.pt')&(sm.weights=='model')&(sm.input=='training_control')&(sm.sampling_steps==1000)]
            if len(sm) and ((sm.mean_r2 < -1)&(sm.generated_max>5*sm.observed_max)).any():continue
        done=ROOT/'results/norman'/run/'research_eval/all_control_cells/complete.json'
        if not done.exists():
            call(['scripts/research_evaluate.py','norman','--run',run,'--data-root',data,
                  '--tag','all_control_cells','--modes','cell_shift','--cells','7353',
                  '--inference-math','ieee'],f'all_controls_{run}')
        ci=ROOT/'results/research/conditional_uncertainty/norman'/run/'all_control_cells.csv'
        if not ci.exists():
            call(['scripts/conditional_uncertainty.py','norman','--run',run,'--data-root',data,
                  '--tag','all_control_cells'],f'uncertainty_all_controls_{run}')
    for ds, data, modes in [('ipsc', 'data/research/source', 'centroid,cell_shift,oracle'),
                            ('norman', 'data/research/source_full', 'centroid,cell_shift,literal,oracle')]:
        dest = ROOT / 'results' / ds / 'source_fp32_s42/research_eval/ema_controls/complete.json'
        if not dest.exists():
            call(['scripts/research_evaluate.py', ds, '--run', 'source_fp32_s42', '--data-root', data,
                  '--tag', 'ema_controls', '--weights', 'ema', '--modes', modes,
                  '--sigmas', '0,0.5,0.7', '--inference-math', 'ieee'], f'ema_{ds}_source_s42')
        early = ROOT / 'results' / ds / 'source_fp32_s42/research_eval/step050000_controls/complete.json'
        if not early.exists():
            call(['scripts/research_evaluate.py', ds, '--run', 'source_fp32_s42', '--data-root', data,
                  '--tag', 'step050000_controls', '--checkpoint', 'step_050000.pt', '--modes', modes,
                  '--sigmas', '0,0.5,0.7', '--inference-math', 'ieee'], f'50k_{ds}_source_s42')
        official = ROOT / 'results' / ds / 'source_fp32_s42/official_api/complete.json'
        needs_official=not official.exists()
        if ds=='ipsc' and official.exists():
            needs_official=not json.loads(official.read_text()).get('includes_no_shift_null_and_forward_day3',False)
        if needs_official:
            call(['scripts/official_inference.py',ds,'--run','source_fp32_s42','--data-root',data],f'official_api_{ds}_source_s42')
    for run,data in [(f'source_fp32_s{s}','data/research/source') for s in [42,43,44]]+[
            ('fresh_endpoints_fp32_s42','data/research/fresh_endpoints'),
            ('full_endpoints_fp32_s42','data/research/full_endpoints')]:
        done=ROOT/'results/ipsc'/run/'research_eval/endpoint_validated_noise/complete.json'
        if not done.exists():
            call(['scripts/calibrate_research_noise.py','--run',run,'--data-root',data],f'noise_calibration_{run}')
        population=ROOT/'results/ipsc'/run/'population_audit/endpoint_validated_noise_without_T.csv'
        if not population.exists():
            call(['scripts/population_audit.py','--run',run,'--data-root',data,
                  '--tag','endpoint_validated_noise','--exclude-genes','T'],f'population_calibrated_{run}')
    call(['scripts/protocol_audit.py'], 'protocol_audit_final')
    call(['scripts/research_report.py'], 'research_report_final')
    call(['scripts/transient_precision_audit.py'], 'transient_precision_final')
    call(['scripts/input_baseline_audit.py'], 'input_baseline_final')
    call(['scripts/scientific_figures.py'], 'scientific_figures_final')
    (ROOT / 'logs/research_followup_complete.json').write_text(json.dumps(dict(
        time=time.strftime('%Y-%m-%d %H:%M:%S'), jobs=len(jobs), status='complete')))
    call(['scripts/assessment_snapshot.py'], 'assessment_snapshot_final')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int)
    main(p.parse_args())
