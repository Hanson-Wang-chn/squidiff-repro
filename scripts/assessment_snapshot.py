"""Machine-readable evidence inventory; scientific interpretation stays in README."""
import json
import shutil
import time
import numpy as np
import pandas as pd
from common import ROOT,write_json


def describe(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return None
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std(ddof=1)) if len(a)>1 else None,
                min=float(a.min()),max=float(a.max()))


def records(frame):return json.loads(frame.to_json(orient='records'))


def main():
    out=ROOT/'results/research';jobs=[];inventory=[]
    for name in [f'research_stage{i}' for i in range(1,6)]:
        p=ROOT/f'configs/{name}.json'
        if p.exists():jobs.extend(json.loads(p.read_text()))
    for j in jobs:
        folder=ROOT/'results'/j['dataset']/j['run']
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        inventory.append(dict(dataset=j['dataset'],run=j['run'],step=status.get('step',0),
            training_complete=status.get('status')=='trained' and status.get('step',0)==100000,
            raw_evaluation_complete=(folder/'research_eval/raw_controls/complete.json').exists(),
            resumable_checkpoint=(folder/'latest.pt').exists(),weights_50k=(folder/'step_050000.pt').exists(),
            data_root=j.get('data_root','data/processed')))
    frame=pd.read_csv(out/'all_controlled_metrics.csv')
    raw=frame[(frame.step==100000)&(frame.tag=='raw_controls')&(frame.evaluation_scope=='all_genes')]
    source=raw[raw.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')]
    source_summary=[]
    for (ds,mode,target),g in source.groupby(['dataset','mode','target']):
        item=dict(dataset=ds,mode=mode,target=target,independent_training_seeds=sorted(g.training_seed.unique().astype(int).tolist()))
        for metric in ['pearson','r2','exact_linear_r2','relative_mse_gain','interaction_r2',
                       'interaction_pearson','predicted_T_mean','true_T_mean','T_amplitude_ratio','mean_gene_wasserstein']:
            if metric in g:item[metric]=describe(g[metric])
        source_summary.append(item)
    null_rows=[]
    for seed in [42,43,44]:
        p=ROOT/f'results/ipsc/source_fp32_s{seed}/sampling_nulls_source_repeats/metrics.csv'
        if not (p.parent/'complete.json').exists():continue
        m=pd.read_csv(p)
        for rep,g in m.groupby('repeat'):
            matched=g[g.schema=='matched_centroid_and_noise_all_days'].set_index('day')
            ends=g[g.schema=='source_endpoint_reconstruction'].set_index('day')
            t=matched.predicted_T
            null_rows.append(dict(training_seed=seed,noise_repeat=int(rep),
                matched_day0=float(t[0]),matched_day1=float(t[1]),matched_day2=float(t[2]),matched_day3=float(t[3]),
                day1_change_without_sampling_switch=float(t[1]-t[0]),
                released_mixed_middle_excess=float(max(t[1],t[2])-ends.predicted_T.max()),
                matched_middle_excess=float(max(t[1],t[2])-max(t[0],t[3])),
                endpoint_day3_reconstruction_r2=float(ends.loc[3,'r2']) if 'r2' in ends else None,
                day3_from_day0_r2=float(matched.loc[3,'r2']) if 'r2' in matched else None))
    pd.DataFrame(null_rows).to_csv(out/'sampling_null_summary.csv',index=False)
    null_summary={k:describe([r[k] for r in null_rows]) for k in null_rows[0] if k not in ['training_seed','noise_repeat']} if null_rows else {}
    population=[]
    for seed in [42,43,44]:
        p=ROOT/f'results/ipsc/source_fp32_s{seed}/population_audit/raw_controls_without_T.csv'
        if p.exists():
            m=pd.read_csv(p);m['training_seed']=seed;population.extend(records(m))
    population_all=[]
    for p in sorted((ROOT/'results/ipsc').glob('*/population_audit/*without_T.csv')):
        m=pd.read_csv(p);m['run']=p.parents[1].name;m['evaluation_tag']=p.stem.removesuffix('_without_T')
        population_all.extend(records(m))
    pd.DataFrame(population_all).to_csv(out/'population_diagnostics.csv',index=False)
    validation=[]
    for p in sorted((ROOT/'results/ipsc').glob('*/noise_validation/selection.json')):
        j=json.loads(p.read_text());validation.append(dict(run=p.parents[1].name,
            selected_sigma=j['sigma'],training_latent_rms=j['training_within_group_latent_rms'],
            endpoint_validation=j['selected']))
    files={name:json.loads((ROOT/path).read_text()) for name,path in [
        ('implementation','results/research/implementation_audit.json'),
        ('native_resume','results/research/native_loader_resume_audit.json'),
        ('protocol','results/research/protocol_audit.json'),
        ('raw_T_dynamics','results/research/raw_dynamics/summary.json'),
        ('source_hashes','docs/source_verification.json'),
        ('author_artifacts','docs/author_artifacts_audit.json')] if (ROOT/path).exists()}
    snapshot=dict(updated=time.strftime('%Y-%m-%d %H:%M:%S Asia/Shanghai'),
        required_research_jobs=len(jobs),completed_research_jobs=sum(r['training_complete'] for r in inventory),
        all_required_models_and_evaluations_complete=all(r['training_complete'] and r['raw_evaluation_complete'] and r['resumable_checkpoint'] for r in inventory),
        followup_complete=(ROOT/'logs/research_followup_complete.json').exists(),
        runs=inventory,source_three_seed_summary=source_summary,
        norman_all_controls_results=records(frame[(frame.dataset=='norman')&(frame.tag=='all_control_cells')]),
        sampling_null_summary=null_summary,sampling_null_runs=null_rows,
        endpoint_validated_noise_choices=validation,
        endpoint_validated_noise_forecasts=records(frame[frame.tag=='endpoint_validated_noise']),
        extension_final_results=records(raw[raw.run.isin(['fresh_endpoints_fp32_s42','full_endpoints_fp32_s42','source1000_fp32_s42','source1000_epsres_s42','balanced_fp32_s42'])]),
        thousand_input_shared203_results=records(frame[(frame.run=='source1000_fp32_s42')&(frame.evaluation_scope=='shared_source203')&frame.tag.isin(['raw_controls','all_control_cells'])]),
        population_without_T=population,population_all_inference_variants=population_all,
        checks=files,workspace_free_gib=shutil.disk_usage(ROOT).free/1024**3,
        limits=['Cell resampling intervals are conditional, not biological replication.',
                'Independent training seeds use one fixed source-like cell split; fresh cell draws are not donor-held-out.',
                'The exact original author checkpoints, feature lists and preprocessing seed remain unavailable.',
                'Oracle, all-days training, no-shift controls and generation repetitions are labelled separately.'])
    write_json(snapshot,out/'assessment.json');print('Evidence snapshot:',snapshot['completed_research_jobs'],'/',len(jobs))


if __name__=='__main__':main()
