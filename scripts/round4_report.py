"""Build the numeric round4 report from completed saved artifacts only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write_json
from round4 import OUT,VARIANTS,load_data

NAMES=['native']+list(VARIANTS)
LABEL={'native':'Native','layernorm':'LayerNorm','resblock':'Block skip',
       'fullsource':'Full endpoints','supportbalance':'Support balance'}


def table(rows,cols):
    lines=['| '+' | '.join(label for key,label in cols)+' |',
           '| '+' | '.join('---' for _ in cols)+' |']
    for row in rows:
        fields=[]
        for key,label in cols:
            v=row.get(key,'')
            if isinstance(v,(float,np.floating)):v=f'{v:.4f}'
            fields.append(str(v))
        lines.append('| '+' | '.join(fields)+' |')
    return '\n'.join(lines)


def replace(text,label,body):
    a=f'<!-- {label}_START -->';b=f'<!-- {label}_END -->'
    start=text.index(a);stop=text.index(b,start)+len(b)
    return text[:start]+a+'\n\n'+body+'\n\n'+b+text[stop:]


def main(args):
    frames=[];gene_frames=[];selections=[];baselines=[];runs=[]
    for path in sorted(OUT.rglob('eval_final_1000/complete.json')):
        c=json.loads(path.read_text());name=c['args']['model']
        f=pd.read_csv(path.parent/'metrics.csv');frames.append(f)
        g=pd.read_csv(path.parent/'gene_means.csv');g['model']=name;gene_frames.append(g)
        s=json.loads((path.parent/'selection.json').read_text())
        selections.append(dict(model=name,criterion=s['criterion'],**s['selected']))
        b=pd.read_csv(path.parent/'distribution_baselines.csv');baselines.append(b)
    if not frames:raise RuntimeError('No completed full-chain evaluations')
    all_metrics=pd.concat(frames,ignore_index=True)
    genes=pd.concat(gene_frames,ignore_index=True)
    baseline=pd.concat(baselines,ignore_index=True)
    all_metrics.to_csv(OUT/'all_metrics.csv',index=False)
    genes.to_csv(OUT/'all_gene_means.csv',index=False)
    baseline.to_csv(OUT/'distribution_baselines.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'selections.csv',index=False)
    for name in VARIANTS:
        d=OUT/f'{name}_s42';status=json.loads((d/'status.json').read_text())
        f=pd.read_json(d/'train.jsonl',lines=True)
        vv=f.dropna(subset=['val_loss']) if 'val_loss' in f else pd.DataFrame()
        row=dict(model=name,updates=status['step'],status=status['status'])
        if len(vv):row.update(best_val_step=int(vv.loc[vv.val_loss.idxmin(),'step']),best_val=float(vv.val_loss.min()),last_val=float(vv.val_loss.iloc[-1]))
        if 'seconds' in status:row['minutes']=status['seconds']/60
        runs.append(row)
    pd.DataFrame(runs).to_csv(OUT/'training_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    colors=dict(zip(NAMES,plt.get_cmap('tab10').colors))
    for name in VARIANTS:
        f=pd.read_json(OUT/f'{name}_s42/train.jsonl',lines=True)
        axes[0].plot(f.step,f.loss,label=LABEL[name],color=colors[name])
        if 'val_loss' in f:
            vv=f.dropna(subset=['val_loss']);axes[1].plot(vv.step,vv.val_loss,'o-',label=LABEL[name],color=colors[name],ms=3)
    axes[0].set(xlabel='Optimizer updates',ylabel='Training epsilon MSE',title='Same 100k budget')
    axes[1].set(xlabel='Optimizer updates',ylabel='Held-out endpoint loss',title='Monitoring; final100k is primary')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=180);plt.close(fig)
    if args.partial:
        print('PARTIAL',len(frames),'completed evaluations',flush=True)
        return
    assert len(frames)==10
    integrity=json.loads((OUT/'final_integrity.json').read_text())
    assert integrity['total_updates']==400000 and integrity['total_predictions']==100 and integrity['supplementary_predictions']==10
    selected=all_metrics[(all_metrics.method=='selected')&all_metrics.model.isin(NAMES)&all_metrics.day.isin([1,2])].copy()
    selected['order']=selected.model.map({v:i for i,v in enumerate(NAMES)})
    selected=selected.sort_values(['order','day'])
    selected['gain_percent']=selected['mse_gain']*100
    selected['common_gain_percent']=selected['reference_mse_gain']*100
    selected['tp_fp']=selected.transient_tp.astype(int).astype(str)+'/'+selected.transient_fp.astype(int).astype(str)
    selected.to_csv(OUT/'selected_metrics.csv',index=False)
    body='## 6．已完成的训练和完整链结果\n\n'
    body+='新增四版均从头完成100000次更新，共400000次；原生基线与五个推理对照复用历史权重，不算新增训练。以下均为600生成细胞、1000步DDIM、固定最终权重，没有择优种子或中间日选参。\n\n'
    body+=table(runs,[('model','新版本'),('updates','更新数'),('minutes','训练分钟'),('best_val_step','最低端点loss步'),('best_val','最低端点loss'),('last_val','最终端点loss')])
    body+='\n\n![训练与独立端点验证](results/round4/training_curves.png)\n\n'
    body+='### 主比较：仅端点验证选择条件分布\n\n'
    body+=table(selected.to_dict('records'),[('model','版本'),('day','日'),('pearson','Pearson'),('r2','R² truth'),('notebook_reverse_r2','R² notebook'),
        ('baseline_r2','自身线性R²'),('gain_percent','自身MSE改善%'),('common_gain_percent','共同基线改善%'),('T_pred','T均值'),('T_above3','T>3比例'),('tp_fp','瞬态TP/FP'),('energy_distance','联合energy')])
    body+='\n\n共同基线始终采用历史source完整训练池的精确均值；fullsource的“自身基线”改用其16580细胞的精确均值。其余模型的两种基线相同。TP/FP使用同一个历史source端点包络，真实day1瞬态数13，day2为5。单次采样的微小分数变化不视为可靠增益。\n\n'
    body+=table(selections,[('model','模型/推理版本'),('mode','条件方式'),('sigma','σ/系数'),('wasserstein','端点W'),('mse','端点均值MSE')])
    markers=[]
    for gene in ['T','DKK1','DKK4','FST','NANOG','GATA6']:
        q=genes[(genes.method=='selected')&(genes.day==1)&(genes.gene==gene)]
        row=dict(gene=gene,truth=float(q.observed.iloc[0]))
        row.update({r['model']:float(r['predicted']) for _,r in q.iterrows() if r['model'] in NAMES})
        markers.append(row)
    pd.DataFrame(markers).to_csv(OUT/'day1_markers.csv',index=False)
    body+='\n\n### day1标记的绝对幅度\n\n'
    body+=table(markers,[('gene','基因'),('truth','真实')]+[(v,v) for v in NAMES])
    tr,va,te=load_data({'panel':'source'})
    envelope=np.maximum(*[tr.X[tr.obs.Group==g].astype(float).mean(0) for g in [0,3]])
    y=te.X[te.obs.Group==1].astype(float).mean(0);actual=y-envelope>=.5
    idx=[tr.var_names.get_loc(g) for g in ['T','DKK1','DKK4']]
    errors=[]
    for name in NAMES:
        q=genes[(genes.model==name)&(genes.method=='selected')&(genes.day==1)].set_index('gene').loc[tr.var_names]
        squared=(q.predicted.to_numpy()-y)**2
        errors.append(dict(model=name,T_DKK_error_percent=float(squared[idx].sum()/squared.sum()*100),
            transient_error_percent=float(squared[actual].sum()/squared.sum()*100)))
    pd.DataFrame(errors).to_csv(OUT/'day1_error_decomposition.csv',index=False)
    body+='\n\n'+table(errors,[('model','版本'),('T_DKK_error_percent','T/DKK占均值平方误差%'),('transient_error_percent','13瞬态基因占误差%')])
    fixed=all_metrics[(all_metrics.method=='fixed')&all_metrics.model.isin(NAMES)&all_metrics.day.isin([1,2])].copy()
    fixed['tp_fp']=fixed.transient_tp.astype(int).astype(str)+'/'+fixed.transient_fp.astype(int).astype(str)
    body+='\n\n### 固定公开σ=0.7：包括所有失败\n\n'
    body+=table(fixed.to_dict('records'),[('model','版本'),('day','日'),('r2','R²'),('pearson','Pearson'),('T_pred','T均值'),('T_above3','T>3比例'),('tp_fp','TP/FP'),('predicted_max','单值最大')])
    controls=all_metrics[(~all_metrics.model.isin(VARIANTS))&all_metrics.day.isin([1,2])&all_metrics.method.isin(['fixed','selected'])].copy()
    controls['tp_fp']=controls.transient_tp.astype(int).astype(str)+'/'+controls.transient_fp.astype(int).astype(str)
    body+='\n\n### 历史原生权重的推理修正与公式对照\n\n'
    body+='bncal仅重校准训练端点BN；ema使用历史EMA参数与原缓冲区；emacal为EMA重新估计训练端点BN；finalclip移除逐步x₀下界、仅最后裁剪非负；paper_cellnoise实现逐细胞起点加η。前四项固定原生已选条件方案，最后一项按新补读公式单独作端点验证，不按day1挑选。\n\n'
    body+=table(controls.to_dict('records'),[('model','推理版本'),('method','方案'),('day','日'),('r2','R²'),('T_pred','T均值'),('T_above3','T>3比例'),('tp_fp','TP/FP'),('energy_distance','联合energy')])
    diagnostic=all_metrics[all_metrics.method.isin(['mean_direction_removed','target_rna_oracle'])].copy()
    diagnostic['tp_fp']=diagnostic.transient_tp.astype(int).astype(str)+'/'+diagnostic.transient_fp.astype(int).astype(str)
    body+='\n\n### 匹配负对照与真实RNA编码诊断\n\n'
    body+='移除方向时保留完全相同的残差和xT；oracle把真实day1 RNA提供给编码器，仅表示条件生成诊断，不能算预测成功。\n\n'
    body+=table(diagnostic.to_dict('records'),[('model','版本'),('method','诊断'),('r2','R²'),('T_pred','T均值'),('tp_fp','TP/FP'),('energy_distance','联合energy')])
    tails=pd.read_csv(OUT/'endpoint_tail/metrics.csv')
    body+='\n\n### 训练端点的高T尾部容量检查\n\n'
    body+='选取各自训练池全部T>3的行，分别提供它们自身RNA编码或同日平均编码；两种条件共享同一xT，取自主评价已冻结的输入bank。该阈值与比较在新模型的中间日评价之前冻结。这只检验已知端点尾部的条件生成，不能当作预测day1或恢复其群体比例。fullsource与source尾部细胞集合不同，不横比其原始R²。\n\n'
    body+=table(tails.to_dict('records'),[('model','模型'),('mode','条件'),('training_rows','行数'),('unique_training_ids','唯一细胞'),('T_observed','T真实'),('T_predicted','T生成'),('DKK1_observed','DKK1真实'),('DKK1_predicted','DKK1生成'),('r2','均值R²')])
    body+='\n\n### 不学习中间态的分布基线\n\n'
    base=baseline[baseline.model.isin(['native','fullsource'])]
    body+=table(base.to_dict('records'),[('model','训练池'),('day','日'),('method','基线'),('r2','R²'),('mean_gene_wasserstein','逐基因W'),('energy_distance','联合energy'),('T_pred','T均值')])
    body+='\n\n![同面板主结果](results/round4/controlled_results.png)\n\n![统一采样的绝对幅度轨迹](results/round4/marker_trajectories.png)\n\n'
    body+='全部候选、主结果、四天曲线、oracle与负对照见 [all_metrics.csv](results/round4/all_metrics.csv)、[all_gene_means.csv](results/round4/all_gene_means.csv) 和每个 `eval_final_1000/` 目录；失败结果没有删除。\n'
    # Scientific plots retain real expression units and a visible truth reference.
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    d1=selected[selected.day==1].set_index('model').loc[NAMES]
    xx=np.arange(len(NAMES));labels=[LABEL[v] for v in NAMES]
    axes[0,0].bar(xx,d1.r2,color=[colors[v] for v in NAMES]);axes[0,0].axhline(float(d1.reference_baseline_r2.iloc[0]),color='black',ls='--',lw=1,label='Common linear baseline')
    axes[0,0].set(title='Day1 mean-expression R²',ylabel='R²');axes[0,0].legend(fontsize=7)
    axes[0,1].bar(xx,d1.common_gain_percent,color=[colors[v] for v in NAMES]);axes[0,1].axhline(0,color='black',lw=.8)
    axes[0,1].set(title='Day1 mean MSE gain',ylabel='% versus common linear baseline')
    axes[0,2].bar(xx,d1.energy_distance,color=[colors[v] for v in NAMES])
    mixture=float(base[(base.model=='native')&(base.day==1)&(base.method=='endpoint_mixture')].energy_distance.iloc[0])
    axes[0,2].axhline(mixture,color='black',ls='--',lw=1,label='Endpoint mixture');axes[0,2].set(title='Day1 joint distribution',ylabel='Energy (lower is better)');axes[0,2].legend(fontsize=7)
    for ax,gene in zip(axes[1],['T','DKK1','DKK4']):
        row=next(r for r in markers if r['gene']==gene)
        vals=[row[v] for v in NAMES]
        ax.bar(xx,vals,color=[colors[v] for v in NAMES]);ax.axhline(row['truth'],color='black',ls='--',label='Real day1')
        ax.set(title=gene,ylabel='Mean expression (CSV units)',ylim=(0,max(row['truth'],max(vals))*1.13));ax.legend(fontsize=7)
    for ax in axes.flat:ax.set_xticks(xx,labels,rotation=25,ha='right',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'controlled_results.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,gene in zip(axes,['T','DKK1','DKK4']):
        truth=[float(te.X[te.obs.Group==day,te.var_names.get_loc(gene)].astype(float).mean()) for day in range(4)]
        ax.plot(range(4),truth,'ko-',lw=2,label='Observed')
        for name in NAMES:
            qq=genes[(genes.model==name)&(genes.method=='selected')&(genes.gene==gene)].sort_values('day')
            ax.plot(qq.day,qq.predicted,'o-',color=colors[name],lw=1,ms=3,label=LABEL[name])
        ax.set(title=gene,xlabel='Biological day',ylabel='Mean expression (CSV units)',xticks=range(4))
        ax.set_ylim(bottom=0)
    axes[2].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'marker_trajectories.png',dpi=180);plt.close(fig)
    report=ROOT/'report-4.md';text=report.read_text();text=replace(text,'ROUND4_RESULTS',body);report.write_text(text)
    print('REPORT_NUMBERS_UPDATED',len(all_metrics),'rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');main(p.parse_args())
