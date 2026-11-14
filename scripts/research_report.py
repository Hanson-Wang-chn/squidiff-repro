"""Aggregate all controlled runs without selecting the best test seed or mode."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write_json
from evaluate import metrics

def main():
    dest=ROOT/'results/research';dest.mkdir(exist_ok=True)
    frames=[]
    for ds in ['ipsc','norman']:
        for path in sorted((ROOT/'results'/ds).glob('*/research_eval/*/metrics.csv')):
            if not (path.parent/'complete.json').exists():continue
            f=pd.read_csv(path)
            cfg=json.loads((path.parents[2]/'config.json').read_text())
            f['training_seed']=cfg.get('seed');f['training_precision']=cfg.get('precision')
            f['training_math']=cfg.get('math_mode','tf32');f['data_root']=cfg.get('data_root','data/processed')
            f['model_variant']=cfg.get('model_variant','source')
            f['sampling_stability_failure']=False
            stability=path.parents[2]/'sampling_stability/metrics.csv'
            if stability.exists():
                sm=pd.read_csv(stability)
                sm=sm[(sm.checkpoint=='latest.pt')&(sm.weights=='model')&
                      (sm.input=='training_control')&(sm.sampling_steps==1000)]
                if len(sm):
                    f['sampling_stability_failure']=bool(((sm.mean_r2 < -1)&(sm.generated_max>5*sm.observed_max)).any())
            f['training_cells']=cfg.get('train_cells');f['input_genes']=cfg.get('genes')
            f['scored_genes']=cfg.get('genes')
            f['eval_directory']=str(path.parent.relative_to(ROOT));f['evaluation_scope']='all_genes'
            f['diagnostic_only']=f['mode'].isin(['oracle','target_centroid'])
            f['intermediate_or_combo_training_seen']=f.data_root.str.contains('all_days|combo_seen',regex=True)
            if 'predicted_T_mean' in f:f['T_amplitude_ratio']=f.predicted_T_mean/f.true_T_mean
            frames.append(f)
            if ds=='norman' and cfg.get('genes',0)>203:
                data=ROOT/cfg.get('data_root','data/processed')/'norman'
                tr=ad.read_h5ad(data/'train.h5ad');te=ad.read_h5ad(data/'test.h5ad')
                ref=ad.read_h5ad(ROOT/'data/research/source_full/norman/test.h5ad')
                if not set(ref.var_names)<=set(tr.var_names):continue
                idx=tr.var_names.get_indexer(ref.var_names);truth_cells=te.X[:,idx].astype(float);truth=truth_cells.mean(0)
                g=tr.obs.Group.to_numpy();mm={k:tr.X[g==k][:,idx].astype(float).mean(0) for k in [0,1,2]}
                baseline=mm[1]+mm[2]-mm[0];real_i=truth-baseline
                extra=[]
                for _,row in f.iterrows():
                    pp=path.parent/f'{row["mode"]}_AB_rep{int(row["repeat"])}.npz'
                    if not pp.exists():continue
                    pred_cells=np.load(pp)['prediction'][:,idx].astype(float);pred=pred_cells.mean(0)
                    rr=row.to_dict();rr.update(**metrics(truth_cells,pred_cells,tr.X[g==0][:,idx]))
                    rr.update(evaluation_scope='shared_source203',scored_genes=len(idx),
                        exact_linear_r2=float(r2_score(truth,baseline)),exact_linear_mse=float(np.mean((truth-baseline)**2)),
                        relative_mse_gain=1-float(np.mean((truth-pred)**2))/float(np.mean((truth-baseline)**2)),
                        interaction_r2=float(r2_score(real_i,pred-baseline)),
                        interaction_pearson=float(pearsonr(real_i,pred-baseline).statistic))
                    extra.append(rr)
                if extra:frames.append(pd.DataFrame(extra))
    if not frames:return
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(dest/'all_controlled_metrics.csv',index=False)
    frame[frame.sampling_stability_failure].to_csv(dest/'sampling_failures_excluded_from_biological_comparison.csv',index=False)
    primary=frame[(frame.step==100000)&(frame.tag=='raw_controls')&(~frame.diagnostic_only)&
                  (~frame.intermediate_or_combo_training_seen)&(~frame.sampling_stability_failure)&(frame.evaluation_scope=='all_genes')]
    primary.to_csv(dest/'fixed_final_forecasts.csv',index=False)
    source=primary[primary.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')]
    if not source.empty:
        cols=[c for c in ['r2','relative_mse_gain','interaction_r2','predicted_T_mean','T_amplitude_ratio','mean_gene_wasserstein'] if c in source]
        grouped=source.groupby(['dataset','mode','target'])[cols].agg(['mean','std','min','max'])
        grouped.columns=[f'{a}_{b}' for a,b in grouped.columns];grouped.to_csv(dest/'source_independent_seed_summary.csv')
        counts=source.groupby(['dataset','mode','target']).training_seed.nunique();counts.to_csv(dest/'source_independent_seed_counts.csv')
    all_controls=frame[(frame.dataset=='norman')&(frame.tag=='all_control_cells')&
                       (frame.evaluation_scope=='all_genes')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')]
    if not all_controls.empty:
        cols=['r2','relative_mse_gain','interaction_r2','interaction_pearson','mean_gene_wasserstein']
        summary=all_controls.groupby(['mode','target'])[cols].agg(['mean','std','min','max'])
        summary.columns=[f'{a}_{b}' for a,b in summary.columns]
        summary.to_csv(dest/'source_all_control_seed_summary.csv')
    # Plot all fixed final forecasts and diagnostics with the source sigma distinguished.
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    selected=primary[((primary.dataset=='norman')&primary['mode'].eq('centroid'))|
                     ((primary.dataset=='ipsc')&primary['mode'].eq('centroid_sigma0.7'))]
    labels=[];positions=[]
    for i,(_,r) in enumerate(selected.iterrows()):
        label=f'{r.dataset}/{r.run}/{r.target}';labels.append(label);positions.append(i)
        axs[0].scatter(r.relative_mse_gain*100,i,color='C0' if r.dataset=='norman' else 'C1')
    axs[0].set_yticks(positions,labels,fontsize=7);axs[0].axvline(0,color='k',ls='--',lw=.8)
    axs[0].set_xscale('symlog',linthresh=20)
    axs[0].set(xlabel='MSE reduction versus exact linear mean baseline (%)',title='Source inference rule; fixed final models')
    ip=frame[(frame.dataset=='ipsc')&(frame.step==100000)&(frame.tag=='raw_controls')&
             (frame.target=='day1')&frame['mode'].isin(['centroid_sigma0.7','oracle','target_centroid'])]
    for i,(_,r) in enumerate(ip.iterrows()):
        axs[1].scatter(r.T_amplitude_ratio,i,color='C2' if r.diagnostic_only else 'C1')
    axs[1].set_yticks(range(len(ip)),[f'{r.run}/{r["mode"]}' for _,r in ip.iterrows()],fontsize=7)
    axs[1].axvline(1,color='k',ls='--',lw=.8);axs[1].set(xlabel='Predicted / observed day1 T mean',title='Forecasts and explicitly labelled oracle controls')
    fig.tight_layout();fig.savefig(dest/'controlled_comparison.png',dpi=180);plt.close(fig)
    write_json({'controlled_rows':len(frame),'fixed_final_forecast_rows':len(primary),
                'completed_training_runs':sorted(set(frame.dataset+'/'+frame.run)),
                'notes':['All modes and seeds retained; no test-best selection',
                         'Oracle and all-days training results are not endpoint-only forecasting',
                         '1000-gene experiments also reported on the shared 203-gene evaluation set']},dest/'report_manifest.json')
    print(primary[['dataset','run','mode','target','r2','relative_mse_gain']].to_string(index=False))

if __name__=='__main__':main()
