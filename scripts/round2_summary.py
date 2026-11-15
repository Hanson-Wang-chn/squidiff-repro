"""Collect new runs, audit predictions, and render report-2 result tables/figure."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import round2 as r


def table(rows, columns):
    out=['| '+' | '.join(label for key,label in columns)+' |',
         '| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:
        cells=[]
        for key,label in columns:
            x=row.get(key,'')
            cells.append(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x))
        out.append('| '+' | '.join(cells)+' |')
    return '\n'.join(out)


def main():
    frames=[];integrity=[];statuses=[]
    for variant in r.VARIANTS:
        dest=r.OUT/f'{variant}_s42'
        if not (dest/'config.json').exists():continue
        config=json.loads((dest/'config.json').read_text())
        status=json.loads((dest/'status.json').read_text())
        statuses.append(dict(variant=variant,**status))
        tr,va,te=r.load_data(config)
        for path in sorted(dest.glob('eval_*/metrics.csv')):
            if not (path.parent/'complete.json').exists():continue
            f=pd.read_csv(path);f['evaluation']=path.parent.name
            for _,row in f.iterrows():
                pth=path.parent/f'{row.method}_day{row.day}_rep{row["repeat"]}.npz'
                blob=np.load(pth,allow_pickle=False)
                assert np.array_equal(blob['genes'],tr.var_names.to_numpy())
                pred=blob['prediction'];assert np.isfinite(pred).all()
                truth=te.X[te.obs.Group==row.day]
                err=abs(float(np.mean((truth.astype(float).mean(0)-pred.astype(float).mean(0))**2))-row.mean_mse)
                assert err<1e-8,(pth,err)
                integrity.append(dict(file=str(pth.relative_to(r.ROOT)),max_mse_error=err,
                    sha256=hashlib.sha256(pth.read_bytes()).hexdigest()))
            frames.append(f)
    if not frames:return
    f=pd.concat(frames,ignore_index=True)
    f.to_csv(r.OUT/'all_metrics.csv',index=False)
    r.write_json(dict(status=statuses,predictions=integrity,count=len(integrity),
        all_source_training_complete=all(s['status']=='trained' and s['step']==100000 for s in statuses) and len(statuses)==len(r.VARIANTS)),r.OUT/'integrity.json')
    official=f[(f.sample_steps==1000)&(f.checkpoint=='final.pt')]
    if official.empty:return
    native=official[official.evaluation=='eval_final_1000']
    selected=native[native.method=='selected']
    rows=[]
    for (v,day),group in native[native.method.isin(['fixed','selected'])&native.day.isin([1,2])].groupby(['variant','day']):
        for method,gg in group.groupby('method'):
            row=gg.iloc[0].to_dict()
            for key in ['r2','pearson','notebook_reverse_r2','mse_gain','T_pred','T_above3','common_r2','common_baseline_r2']:
                row[key]=float(gg[key].mean())
            row['tp_fp']=f"{int(gg.transient_tp.iloc[0])}/{int(gg.transient_called.iloc[0]-gg.transient_tp.iloc[0])} ({int(gg.transient_actual.iloc[0])})"
            row['gain_pct']=f"{100*row['mse_gain']:.1f}%"
            row['T_high_pct']=f"{100*row['T_above3']:.1f}%"
            rows.append(row)
    cols=[('variant','版本'),('day','day'),('method','采样'),('pearson','Pearson'),('r2','R² truth'),
          ('notebook_reverse_r2','R² notebook'),('baseline_r2','线性 R²'),('gain_pct','MSE 改善'),
          ('T_pred','T 预测'),('T_high_pct','T>3 比例'),('tp_fp','TP/FP（真实数）')]
    text='\n\n<!-- ROUND2_GENERATED_START -->\n\n## 新增正式训练与完整采样结果\n\n'
    text+='下表均为新增模型的固定 100000 步权重、1000 DDIM 步采样。MSE 改善为 `1−模型 MSE/匹配线性基线 MSE`，负值表示退步。TP/FP 针对超过端点最大均值 0.5 的上调基因。每个版本只有一个独立训练种子，采样重复不等于独立训练。\n\n'
    text+=table(rows,cols)+'\n\n'
    selections=[]
    for variant in r.VARIANTS:
        p=r.OUT/f'{variant}_s42/eval_final_1000/selection.json'
        if p.exists():
            s=json.loads(p.read_text());selections.append(dict(variant=variant,**s['selected']))
    text+='端点验证实际选中的方案（本表不使用 day1/day2 排序）：\n\n'
    text+=table(selections,[('variant','版本'),('mode','潜变量方式'),('sigma','σ/系数'),('wasserstein','端点 Wasserstein')])+'\n\n'
    marker_rows=[];decomposition=[]
    for variant,suffix in [(v,'') for v in r.VARIANTS]+[('zscore','_signed')]:
        gp=r.OUT/f'{variant}_s42/eval_final_1000{suffix}/gene_means.csv'
        if not gp.exists():continue
        gg=pd.read_csv(gp);gg=gg[(gg.day==1)&(gg.method=='selected')]
        errors=(gg.observed-gg.predicted)**2
        transient=gg.observed-gg.endpoint_max>=.5
        three=gg.gene.isin(['T','DKK1','DKK4'])
        decomposition.append(dict(variant=variant+suffix,transient_genes=int(transient.sum()),
            transient_error_share=float(errors[transient].sum()/errors.sum()),
            T_DKK1_DKK4_error_share=float(errors[three].sum()/errors.sum())))
    pd.DataFrame(decomposition).to_csv(r.OUT/'day1_error_decomposition.csv',index=False)
    for name in ['T','DKK1','DKK4','FST','NANOG','GATA6']:
        row=dict(gene=name)
        for variant in r.VARIANTS:
            gp=r.OUT/f'{variant}_s42/eval_final_1000/gene_means.csv'
            if not gp.exists():continue
            gg=pd.read_csv(gp);gg=gg[(gg.gene==name)&(gg.day==1)&(gg.method=='selected')]
            if len(gg):row['observed']=float(gg.observed.iloc[0]);row[variant]=float(gg.predicted.iloc[0])
            else:row[variant]='未入面板'
        gp=r.OUT/'zscore_s42/eval_final_1000_signed/gene_means.csv'
        if gp.exists():
            gg=pd.read_csv(gp);gg=gg[(gg.gene==name)&(gg.day==1)&(gg.method=='selected')]
            if len(gg):row['zscore_signed']=float(gg.predicted.iloc[0])
        marker_rows.append(row)
    text+='### day1 标记与瞬态程序的绝对幅度\n\n'
    text+=table(marker_rows,[('gene','基因'),('observed','真实'),('native','native'),('ln','ln'),
                            ('log2hvg','log2hvg'),('compact','compact'),('zscore','zscore 原采样'),('zscore_signed','zscore 修正下界')])+'\n\n'
    text+='### 共同 135 基因的公平比较\n\n'
    text+=table([dict(variant=v,day=int(day),r2=g.common_r2.mean(),baseline=g.common_baseline_r2.iloc[0])
                 for (v,day),g in selected[selected.day.isin([1,2])].groupby(['variant','day'])],
                [('variant','版本'),('day','day'),('r2','共同基因 R²'),('baseline','共同基因线性 R²')])+'\n\n'
    dist_path=r.OUT/'distribution_baselines.csv'
    if dist_path.exists():
        dist=pd.read_csv(dist_path);dr=[]
        for (panel,method,day),g in dist.groupby(['panel','method','day']):
            dr.append(dict(panel=panel,method=method,day=int(day),W=g.mean_gene_wasserstein.mean(),E=g.energy_distance.mean()))
        for (variant,day),g in selected[selected.day.isin([1,2])].groupby(['variant','day']):
            dr.append(dict(panel=r.VARIANTS[variant]['panel'],method=variant+'_selected',day=int(day),
                           W=g.mean_gene_wasserstein.mean(),E=g.energy_distance.mean()))
        text+='### 单细胞分布对照\n\n'
        text+='cell_linear 为独立抽取训练端点细胞的逐细胞表达插值；endpoint_mixture 只混合端点细胞。两者都不使用中间态，固定 3 次抽样取平均。W 是逐基因 Wasserstein 均值，E 是联合表达 energy distance，均越低越好；不同面板仅在各自内部比较。\n\n'
        text+=table(dr,[('panel','面板'),('method','方式'),('day','day'),('W','W'),('E','E')])+'\n\n'
    text+='### 原生模型推理对照\n\n'
    ablation=native[(native.variant=='native')&native.day.isin([1,2])]
    text+=table([dict(method=method,day=int(day),r2=g.r2.mean(),T=g.T_pred.mean(),
                     tp=g.transient_tp.iloc[0],fp=g.transient_called.iloc[0]-g.transient_tp.iloc[0])
                 for (method,day),g in ablation.groupby(['method','day'])],
                [('method','方式'),('day','day'),('r2','R²'),('T','T 均值'),('tp','TP'),('fp','FP')])+'\n\n'
    text+='oracle 编码了真实目标表达，只用于诊断重建能力。inversion 只保留 day0 随机编码；ot 的配对只用训练端点。no_direction 移除分化方向，其余采样条件匹配 selected。\n\n'
    best=f[(f.sample_steps==1000)&(f.checkpoint=='best.pt')&f.day.isin([1,2])&f.method.isin(['fixed','selected'])]
    if len(best):
        text+='### 仅端点验证选出的早停权重\n\n'
        text+=table(best.to_dict('records'),[('variant','版本'),('step','更新步数'),('day','day'),('method','方式'),('r2','R²'),('T_pred','T 均值')])+'\n\n'
    signed=official[official.evaluation.str.endswith('_signed')&official.day.isin([1,2])]
    if len(signed):
        text+='### z-score 采样下界修正\n\n'
        text+=table(signed.to_dict('records'),[('day','day'),('method','方式'),('r2','R²'),('T_pred','T 均值'),('mse_gain','MSE 改善比例')])+'\n\n'
    sp=r.OUT/'secondary_endpoint_mse.csv'
    if sp.exists():
        secondary=pd.read_csv(sp)
        text+='### 补充：改用端点均值 MSE 选择采样参数\n\n'
        text+='这是在主实验之后增加的诊断对照。仅改变端点候选表的排序指标，不重新搜索候选或按 day1/day2 选参；已有相同生成结果直接复用。主表保持原定 Wasserstein 规则。\n\n'
        text+=table(secondary.to_dict('records'),[('variant','版本'),('signed','修正下界'),('day','day'),
            ('mse_selected_mode','MSE 选定方式'),('sigma','σ/系数'),('primary_r2','W 选择 R²'),
            ('mse_selected_r2','MSE 选择 R²'),('mse_selected_T','T 均值'),('true_positives','TP'),('false_positives','FP')])+'\n\n'
    text+='![新增实验摘要](results/round2/summary.png)\n\n'
    text+='![相同权重、统一采样与原混合采样的标记曲线](results/round2/sampling_curves.png)\n\n'
    text+='![day1 各基因均值与真实表达对照](results/round2/day1_scatter.png)\n\n'
    text+='### 全部已完成推理运行的关键数值\n\n'
    text+='本表包含早期失败、端点控制、oracle 和方法扩展；它们不能混合统计成独立复现成功次数。全部统计字段见 [all_metrics.csv](results/round2/all_metrics.csv)，逐基因均值和细胞级预测见各 evaluation 目录。\n\n'
    complete_rows=f.sort_values(['variant','step','sample_steps','evaluation','method','day']).to_dict('records')
    for row in complete_rows:row['tp_fp']=f"{row['transient_tp']}/{row['transient_called']-row['transient_tp']}"
    text+=table(complete_rows,[('variant','版本'),('step','更新步数'),('sample_steps','DDIM 步'),
        ('evaluation','评估'),('method','方式'),('day','day'),('r2','R²'),('pearson','Pearson'),
        ('T_pred','T 预测'),('tp_fp','TP/FP')])+'\n\n'
    text+='<!-- ROUND2_GENERATED_END -->\n'
    report=r.ROOT/'report-2.md'
    old=report.read_text().split('<!-- ROUND2_GENERATED_START -->')[0].rstrip()
    report.write_text(old+text)

    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for variant in r.VARIANTS:
        p=r.OUT/f'{variant}_s42/train.jsonl'
        if not p.exists():continue
        log=pd.DataFrame([json.loads(l) for l in p.read_text().splitlines()])
        log=log.drop_duplicates('step',keep='last')
        if 'val_loss' in log:
            vv=log.dropna(subset=['val_loss'])
            axs[0,0].plot(vv.step,vv.val_loss,'o-',label=variant)
    axs[0,0].set(title='Endpoint validation denoising loss',xlabel='Optimizer updates',ylabel='MSE',yscale='log')
    axs[0,0].legend()
    plot_selected=selected.copy()
    sc=official[(official.method=='selected')&(official.evaluation=='eval_final_1000_signed')].copy()
    if len(sc):
        sc['variant']='zscore_signed';plot_selected=pd.concat([plot_selected,sc],ignore_index=True)
    day1=plot_selected[plot_selected.day==1].groupby('variant').mean(numeric_only=True)
    if len(day1):
        xx=np.arange(len(day1));axs[0,1].bar(xx-.18,day1.r2,.36,label='Model')
        axs[0,1].bar(xx+.18,day1.baseline_r2,.36,label='Matched linear baseline')
        labels=[name.replace('zscore_signed','zscore\ncorrected') for name in day1.index]
        axs[0,1].set(xticks=xx,xticklabels=labels,ylim=(min(-.1,day1.r2.min()-.1),1),title='Day1: endpoint-selected sampling',ylabel='R2 (truth reference)')
        axs[0,1].legend()
        axs[1,0].bar(xx,day1.T_pred,label='Generated')
        axs[1,0].axhline(day1.T_true.iloc[0],color='black',ls='--',label='Observed')
        axs[1,0].set(xticks=xx,xticklabels=labels,title='Day1 T: absolute amplitude',ylabel='Original CSV expression units')
        axs[1,0].legend()
    aa=ablation[ablation.day==1].groupby('method').mean(numeric_only=True)
    if len(aa):
        xx=np.arange(len(aa));axs[1,1].bar(xx-.2,aa.transient_tp,.4,label='True positives')
        axs[1,1].bar(xx+.2,aa.transient_called-aa.transient_tp,.4,label='False positives')
        axs[1,1].set(xticks=xx,xticklabels=aa.index,title='Native day1: transient upregulation',ylabel='Genes')
        axs[1,1].tick_params(axis='x',labelrotation=45);axs[1,1].legend()
    fig.tight_layout();fig.savefig(r.OUT/'summary.png',dpi=180);plt.close(fig)
    gp=r.OUT/'native_s42/eval_final_1000/gene_means.csv'
    if gp.exists():
        gene=pd.read_csv(gp);fig,axs=plt.subplots(1,3,figsize=(14,4))
        for ax,name in zip(axs,['T','NANOG','GATA6']):
            gg=gene[gene.gene==name]
            obs=gg.groupby('day')['observed'].first().sort_index()
            ax.plot(obs.index,obs.values,'ko-',label='Observed')
            if set(['oracle','fixed']).issubset(set(gg.method)):
                mixed=[float(gg[(gg.day==day)&(gg.method==('oracle' if day in [0,3] else 'fixed'))].predicted.iloc[0]) for day in range(4)]
                ax.plot(range(4),mixed,'o--',label='Released mixed sampling')
            for method,label in [('fixed','Fixed sigma, all days'),('selected','Endpoint selected, all days'),('no_direction','No direction')]:
                curve=gg[gg.method==method].sort_values('day')
                if len(curve):ax.plot(curve.day,curve.predicted,'o-',label=label)
            ax.set(title=name,xlabel='Day',ylabel='Original expression units',xticks=range(4))
        axs[-1].legend(fontsize=7);fig.tight_layout()
        fig.savefig(r.OUT/'sampling_curves.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,8))
    cases=[(v,'') for v in r.VARIANTS]+[('zscore','_signed')]
    for ax,(variant,suffix) in zip(axs.flat,cases):
        evaluation='eval_final_1000'+suffix
        gp=r.OUT/f'{variant}_s42/{evaluation}/gene_means.csv'
        if not gp.exists():ax.axis('off');continue
        gg=pd.read_csv(gp);gg=gg[(gg.method=='selected')&(gg.day==1)]
        ax.scatter(gg.predicted,gg.observed,s=10,alpha=.6)
        limit=max(gg.observed.max(),gg.predicted.max())*1.05
        ax.plot([0,limit],[0,limit],'k--',lw=.8)
        for name in ['T','DKK1','DKK4']:
            q=gg[gg.gene==name]
            if len(q):ax.annotate(name,(q.predicted.iloc[0],q.observed.iloc[0]),fontsize=8)
        score=official[(official.variant==variant)&(official.evaluation==evaluation)&(official.method=='selected')&(official.day==1)]
        title=variant+(' corrected' if suffix else '')
        if len(score):title+=f"; R2={score.r2.mean():.3f}; linear={score.baseline_r2.iloc[0]:.3f}"
        ax.set(title=title,xlabel='Predicted gene mean',ylabel='Observed gene mean',xlim=(0,limit),ylim=(0,limit))
    fig.tight_layout()
    fig.savefig(r.OUT/'day1_scatter.png',dpi=180);plt.close(fig)
    print('SUMMARY',len(f),'metrics;',len(integrity),'verified predictions')


if __name__=='__main__':main()
