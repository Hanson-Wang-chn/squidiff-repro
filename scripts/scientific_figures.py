"""Standalone figures for the final scientific assessment; all seeds are retained."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, write_json


def main():
    out=ROOT/'results/research';forecasts=pd.read_csv(out/'all_controlled_metrics.csv')
    mixed,matched=[],[];plot_rows=[]
    for seed in [42,43,44]:
        run=f'source_fp32_s{seed}'
        m=pd.read_csv(ROOT/'results/ipsc'/run/'sampling_nulls_source_repeats/metrics.csv')
        for rep in sorted(m['repeat'].unique()):
            mm=m[m['repeat']==rep]
            ends=mm[mm.schema=='source_endpoint_reconstruction'].set_index('day')
            all_days=mm[mm.schema=='matched_centroid_and_noise_all_days'].set_index('day')
            original=[float((ends if d in [0,3] else all_days).loc[d,'predicted_T']) for d in range(4)]
            uniform=[float(all_days.loc[d,'predicted_T']) for d in range(4)]
            mixed.append(original);matched.append(uniform)
            for d in range(4):
                plot_rows.append(dict(seed=seed,noise_repeat=int(rep),day=d,
                    mixed_sampler_T=original[d],matched_sampler_T=uniform[d],observed_T=float(all_days.loc[d,'observed_T'])))
    dynamics=pd.DataFrame(plot_rows);dynamics.to_csv(out/'sampling_curve_comparison.csv',index=False)
    mixed,matched=np.asarray(mixed),np.asarray(matched)
    observed=dynamics.groupby('day').observed_T.first().to_numpy()
    fig,axes=plt.subplots(2,2,figsize=(13,9.5))
    ax=axes[0,0];ax.plot(range(4),observed,'o-',color='black',lw=2,label='Observed held-out cells')
    ax.plot(range(4),mixed.mean(0),'o-',color='C1',label='Released mixed sampler')
    ax.fill_between(range(4),mixed.min(0),mixed.max(0),color='C1',alpha=.15)
    calibrated=forecasts[(forecasts.dataset=='ipsc')&forecasts.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&
                         (forecasts.tag=='endpoint_validated_noise')].groupby('target').predicted_T_mean.mean()
    ax.plot([1,2],[calibrated['day1'],calibrated['day2']],'s--',color='C4',label='Endpoint-validated noise')
    positive=pd.read_csv(ROOT/'results/ipsc/all_days_fp32_s42/research_eval/raw_controls/metrics.csv')
    positive=positive[positive['mode']=='oracle'].sort_values('target')
    ax.scatter([1,2],positive.predicted_T_mean,marker='*',s=120,color='C2',label='All-days oracle: diagnostic')
    ax.set(title='A  T amplitude on the actual expression scale',xlabel='Day',ylabel='Mean log-normalized expression',xticks=range(4),ylim=(0,None));ax.legend(fontsize=8)
    ax=axes[0,1]
    for values,color,label in [(mixed,'C1','Released mixed sampler'),(matched,'C0','Same noisy-centroid sampler on all days')]:
        ax.plot(range(4),values.mean(0),'o-',color=color,label=label)
        ax.fill_between(range(4),values.min(0),values.max(0),color=color,alpha=.15)
    ax.set(title='B  Sampling control (zoomed scale)',xlabel='Day',ylabel='Mean T expression',xticks=range(4),ylim=(0,None));ax.legend(fontsize=8)
    # No checkpoint/seed selection: average all three pre-specified source models.
    tables=[pd.read_csv(ROOT/f'results/norman/source_fp32_s{s}/research_eval/raw_controls/gene_means.csv') for s in [42,43,44]]
    points=[]
    for mode in ['centroid','oracle']:
        sub=[t[t['mode']==mode].set_index('gene_id').sort_index() for t in tables]
        truth=sub[0].observed_mean-sub[0].exact_linear_mean
        pred=np.mean([(t.predicted_mean-t.exact_linear_mean).to_numpy() for t in sub],axis=0)
        for gid,real,predicted in zip(sub[0].index,truth,pred):
            points.append(dict(mode=mode,gene_id=gid,gene_name=sub[0].loc[gid,'gene_name'],true_interaction=float(real),predicted_interaction=float(predicted)))
    sub=[pd.read_csv(ROOT/f'results/norman/source_fp32_s{s}/research_eval/all_control_cells/gene_means.csv').set_index('gene_id').sort_index() for s in [42,43,44]]
    truth=sub[0].observed_mean-sub[0].exact_linear_mean
    pred=np.mean([(t.predicted_mean-t.exact_linear_mean).to_numpy() for t in sub],axis=0)
    for gid,real,predicted in zip(sub[0].index,truth,pred):
        points.append(dict(mode='all_controls',gene_id=gid,gene_name=sub[0].loc[gid,'gene_name'],true_interaction=float(real),predicted_interaction=float(predicted)))
    points=pd.DataFrame(points);points.to_csv(out/'interaction_plot_data.csv',index=False)
    ax=axes[1,0]
    for mode,color,label in [('centroid','C0','Centroid forecast'),('all_controls','C3','All-control cellwise forecast'),('oracle','C2','Encode real AB: diagnostic')]:
        p=points[points['mode']==mode]
        ax.scatter(p.true_interaction,p.predicted_interaction,s=15,alpha=.65,color=color,label=label)
        if mode=='centroid':
            for _,r in p[p.gene_name.isin(['ALAS2','HBA1','HBA2'])].iterrows():
                offset={'ALAS2':(-15,-35),'HBA1':(4,12),'HBA2':(4,-12)}[r.gene_name]
                ax.annotate(r.gene_name,(r.true_interaction,r.predicted_interaction),fontsize=8,
                            xytext=offset,textcoords='offset points',arrowprops={'arrowstyle':'-','lw':.5,'color':'gray'})
    lo=min(points.true_interaction.min(),points.predicted_interaction.min());hi=max(points.true_interaction.max(),points.predicted_interaction.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=.8);ax.axhline(0,color='gray',lw=.5)
    ax.set(title='C  Norman: non-additive interaction residual',xlabel='Observed AB - A - B + control',ylabel='Predicted AB - A - B + control');ax.legend(fontsize=8)
    ax=axes[1,1]
    frame=forecasts[(forecasts.step==100000)&forecasts.tag.isin(['raw_controls','all_control_cells','endpoint_validated_noise'])]
    if 'sampling_stability_failure' in frame:frame=frame[~frame.sampling_stability_failure]
    selections=[('Norman: intended centroid', (frame.dataset=='norman')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&(frame['mode']=='centroid')&(frame.evaluation_scope=='all_genes')),
                ('Norman: literal notebook', (frame.dataset=='norman')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&(frame['mode']=='literal')&(frame.evaluation_scope=='all_genes')),
                ('Norman: all controls', (frame.dataset=='norman')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&(frame.tag=='all_control_cells')&(frame.evaluation_scope=='all_genes')),
                ('iPSC: fixed noise / day1', (frame.dataset=='ipsc')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&(frame['mode']=='centroid_sigma0.7')&(frame.tag=='raw_controls')&(frame.target=='day1')),
                ('iPSC: fixed noise / day2', (frame.dataset=='ipsc')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&(frame['mode']=='centroid_sigma0.7')&(frame.tag=='raw_controls')&(frame.target=='day2')),
                ('iPSC: validated noise / day1', (frame.dataset=='ipsc')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&(frame.tag=='endpoint_validated_noise')&(frame.target=='day1')),
                ('iPSC: validated noise / day2', (frame.dataset=='ipsc')&frame.run.str.fullmatch(r'source_fp32_s(?:42|43|44)')&(frame.tag=='endpoint_validated_noise')&(frame.target=='day2')),
                ('iPSC: full endpoints / validated day1',(frame.dataset=='ipsc')&(frame.run=='full_endpoints_fp32_s42')&(frame.tag=='endpoint_validated_noise')&(frame.target=='day1')),
                ('iPSC: fresh cells / validated day1',(frame.dataset=='ipsc')&(frame.run=='fresh_endpoints_fp32_s42')&(frame.tag=='endpoint_validated_noise')&(frame.target=='day1')),
                ('Norman: original 1000 (unstable)',(frame.dataset=='norman')&(frame.run=='source1000_fp32_s42')&(frame['mode']=='centroid')&(frame.evaluation_scope=='shared_source203')),
                ('Norman: residual 1000 (underfit)',(frame.dataset=='norman')&(frame.run=='source1000_epsres_s42')&(frame['mode']=='centroid')&(frame.evaluation_scope=='shared_source203')),
                ('Norman: residual 1000 / all controls',(frame.dataset=='norman')&(frame.run=='source1000_epsres_s42')&(frame.tag=='all_control_cells')&(frame.evaluation_scope=='shared_source203')),
                ('Norman: balanced / all controls',(frame.dataset=='norman')&(frame.run=='balanced_fp32_s42')&(frame.tag=='all_control_cells')&(frame.evaluation_scope=='all_genes'))]
    for i,(label,mask) in enumerate(selections):
        vals=frame[mask].relative_mse_gain.to_numpy()*100
        ax.scatter(vals,np.full(len(vals),i),s=28,color=f'C{i%10}',alpha=.8)
        if 'unstable' in label:ax.text(.02,i,'excluded: training-cell generation failed',transform=ax.get_yaxis_transform(),fontsize=7,color='gray',va='center')
    ax.axvline(0,color='black',ls='--',lw=.8);ax.set_yticks(range(len(selections)),[v[0] for v in selections],fontsize=8)
    ax.set_xscale('symlog',linthresh=20)
    ax.set(title='D  Comparison with exact linear-mean baseline',xlabel='MSE reduction (%) [symmetric log beyond +/-20%]');ax.invert_yaxis()
    fig.tight_layout();fig.savefig(out/'scientific_summary.png',dpi=200);fig.savefig(out/'scientific_summary.svg');plt.close(fig)
    write_json(dict(training_seeds=[42,43,44],noise_repeats_per_seed=3,
        shaded_ranges='Min/max across the 9 generated curves, not biological confidence intervals',
        endpoints='The released mixed sampler reconstructs real individual endpoints; only its middle days are forecasts',
        positive_control='All-days oracle deliberately uses real intermediate inputs and is diagnostic only',
        interaction='All 203 source features, averaged predictions from all 3 pre-specified models'),out/'scientific_figure_method.json')


if __name__=='__main__':main()
