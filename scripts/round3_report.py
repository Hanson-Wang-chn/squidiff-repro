"""Generate numeric evidence tables in report-3 without replacing interpretation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT


def table(rows,cols):
    lines=['| '+' | '.join(label for _,label in cols)+' |',
           '| '+' | '.join('---' for _ in cols)+' |']
    for row in rows:
        fields=[]
        for key,label in cols:
            value=row.get(key,'')
            fields.append(f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value))
        lines.append('| '+' | '.join(fields)+' |')
    return '\n'.join(lines)


def replace(report,label,body):
    start=f'<!-- {label}_START -->'
    stop=f'<!-- {label}_END -->'
    block=f'{start}\n\n{body}\n\n{stop}'
    if start in report:
        a=report.index(start);b=report.index(stop,a)+len(stop)
        return report[:a]+block+report[b:]
    return report+'\n\n'+block+'\n'


def main():
    p=ROOT/'report-3.md';text=p.read_text()
    folder=ROOT/'results/round3/author/eval_1000'
    f=pd.read_csv(folder/'metrics.csv')
    baseline=pd.read_csv(folder/'baseline_metrics.csv')
    names={0:'iPSC',1:'EC',2:'Fibroblast',3:'Mural'}
    rows=[]
    for group in range(4):
        q=f[f.group==group].set_index('mode')
        b=baseline[(baseline.group==group)&(baseline['mode']=='class_mean')].iloc[0]
        rows.append(dict(group=names[group],r2=q.loc['encoded','r2'],pearson=q.loc['encoded','pearson'],
            wrong=q.loc['wrong_class','r2'],gaussian=q.loc['gaussian_0.2','r2'],
            center=q.loc['center','r2'],baseline=b.r2,
            paired=q.loc['encoded','paired_cell_mse'],paired_base=b.paired_mse,
            variance=float(baseline[(baseline.group==group)&(baseline['mode']=='encoded')].variance_ratio.iloc[0])))
    body='## 作者原始权重的直接结果\n\n'
    body+='每组固定随机抽取 300 个配套训练细胞，seed1000，完整 1000 DDIM 步。四种语义方式共享初始 xT；没有按结果选择采样。`encoded` 输入自身真实 RNA 的编码；错误条件来自另一真实细胞类型。高斯中心采用作者 BVO notebook 的 σ=0.2。\n\n'
    body+=table(rows,[('group','组'),('pearson','自身编码 Pearson'),('r2','自身编码 R²'),
        ('wrong','错误类别 R²'),('gaussian','高斯中心 R²'),('center','无噪声中心 R²'),('baseline','类别均值 R²')])
    body+='\n\n类别均值使用全部该类配套训练细胞，只作本实验容易程度的对照；它没有生成真实细胞异质性。下面的配对误差只有在输入同一真实细胞 RNA 的条件生成时才有定义，不能搬用到没有配对关系的中间日预测。\n\n'
    body+=table(rows,[('group','组'),('paired','自身编码配对 MSE'),('paired_base','类别均值配对 MSE'),
                     ('variance','生成/真实总逐基因方差')])
    body+='\n\n把生成行顺序打乱，均值 R² 完全不变，配对 MSE 却从 1.61–3.33 升到 4.26–6.08。模型确实保留了一部分细胞信息；EC 的自身编码配对误差还高于类别均值基线，生成方差普遍缩小至真实的约 60%–77%。语义中心在部分均值指标上更高，却可能丢失异质性，不能仅据均值分数挑选。\n\n'
    body+='![作者权重与条件负对照](results/round3/author/eval_1000/author_controls.png)\n\n'
    body+='提供的权重两个 BN 计数均为 500000，与 BVO notebook 的更新次数相符；6838 个训练细胞中，1000 个来自 iPSC day0，5838 个为 BVO Control，含 1468 EC、952 Fibroblast、3418 Mural。后者已经覆盖 `adata_norm.h5ad` 中全部非 HSC-like Control 细胞，逐值误差为 0，因此该文件不再提供独立的 Control 测试细胞。按 0、1/3、2/3、1、1.5 生成的三谱系曲线保存在 `descriptive_curves.csv` 及 NPZ，只有描述意义，没有真实中间日作为检验。曲线统一采用随机 xT 和 σ=0.2，α=0 也采用同一规则；α=1.5 属于外推。各系数采用一致采样规则，便于解释曲线形状。\n\n'
    body+='596 基因名单中有 NANOG、GATA6、CLDN5、SOX17、LUM 等，但没有 T 或 TBXT，也没有 PECAM1、PDGFRB；指标和曲线仅涵盖模型实际输出的基因，标记覆盖清单见 `author/marker_coverage.json`。因此这份权重不具备 Fig.2 T 的输出坐标。\n\n'
    body+='作者权重可以生成部分与 Fig.4 描述方向相近的定性变化：EC 路径 α=0→1 的 NANOG 从 3.416 降至 0.462，CLDN5 从 0.864 升至 4.393；LUM 为 0.951→1.710→2.180→2.134，存在较弱的中间隆起。它支持某些定性图形可以由作者权重产生，但缺少对应真实时间序列，不能据此确认隆起幅度或发生时间准确。\n\n'
    body+='![作者权重描述性曲线](results/round3/author/eval_1000/author_curves.png)\n\n'
    unseen=ROOT/'results/round3/author/unseen_ipsc/complete.json'
    if unseen.exists():
        u=json.loads(unseen.read_text());s=u['scores']['encoded_unseen_ipsc'];b=u['scores']['training_ipsc_mean']
        body+=f'另外从原始 iPSC 缓存抽取了与配套训练矩阵无 ID 重合的 300 个 day0 细胞，596 基因全部可对齐，作者 1000 个 iPSC 训练细胞与缓存逐值误差也为 0。未见细胞自身编码条件生成 Pearson={s["pearson"]:.4f}、R²={s["r2"]:.4f}、配对 MSE={s["paired_mse"]:.4f}；训练 iPSC 均值基线配对 MSE={b["paired_mse"]:.4f}。这是细胞分离的端点生成诊断，目标 RNA 仍被提供给编码器，没有供体隔离，也不是未见日龄预测。\n\n'
    body+='这组证据支持“公开作者权重具有条件生成能力”，但不验证 Fig.2 的纯端点瞬态预测或 BVO 插值时间的真实性。当前制品无法把后两项结论逐点重放。\n'
    text=replace(text,'AUTHOR_RESULTS',body)
    all_rows=[]
    for variant in ['xstart','minsnr','epsres']:
        pp=ROOT/f'results/round3/{variant}_s42/eval_final_1000/metrics.csv'
        if pp.exists() and (pp.parent/'complete.json').exists():
            z=pd.read_csv(pp);z['evaluation']='round3';all_rows.append(z)
    if len(all_rows)==3:
        f=pd.concat(all_rows,ignore_index=True)
        f.to_csv(ROOT/'results/round3/all_metrics.csv',index=False)
        old=pd.read_csv(ROOT/'results/round2/native_s42/eval_final_1000/metrics.csv')
        combined=pd.concat([old[old.method.isin(['fixed','selected'])&old.day.isin([1,2])],f],ignore_index=True)
        rows=[]
        for _,q in combined.iterrows():
            row=q.to_dict();row['gain']=100*q.mse_gain
            row['tp_fp']=f'{int(q.transient_tp)}/{int(q.transient_called-q.transient_tp)}'
            rows.append(row)
        body='## 同面板的新增训练与正式预测\n\n'
        body+='均为 203 基因、相同训练/验证/测试细胞、seed42、固定最终 100000 更新、1000 DDIM 步、600 生成细胞。native 来自第二轮，是历史基线，不计为本轮新增训练。每个新版本只有一次独立训练，采样不是训练重复。`fixed` 固定 σ=0.7；`selected` 仅按独立端点 Wasserstein 选择潜变量方案。\n\n'
        body+=table(rows,[('variant','版本'),('day','日'),('method','采样'),('pearson','Pearson'),('r2','R²'),
            ('notebook_reverse_r2','反向 R²'),('baseline_r2','线性 R²'),('gain','MSE 改善%'),('T_pred','T 均值'),
            ('T_above3','T>3 比例'),('tp_fp','瞬态 TP/FP'),('energy_distance','联合 energy')])
        body+='\n\n所有结果包含失败和退步版本。真实 day1 T 均值为 6.2501，13 个基因的真实均值超过训练端点最大均值 0.5；day2 依相同规则单独评价。MSE 改善为相对同面板线性表达均值插值的误差下降，负数表示退步。energy 是含对角线的有限样本 V 统计量，越低越好。\n\n'
        selections=[]
        for v in ['xstart','minsnr','epsres']:
            s=json.loads((ROOT/f'results/round3/{v}_s42/eval_final_1000/selection.json').read_text())
            selections.append(dict(variant=v,**s['selected']))
        body+='实际端点选择：\n\n'+table(selections,[('variant','版本'),('mode','方式'),('sigma','σ/系数'),('wasserstein','端点 W')])+'\n'
        body+='\n![同面板主结果与瞬态幅度](results/round3/controlled_comparison.png)\n\n'
        marker_path=ROOT/'results/round3/marker_means.csv'
        if marker_path.exists():
            g=pd.read_csv(marker_path);g=g[g.day==1]
            marker_rows=[]
            for name in ['T','DKK1','DKK4','FST','NANOG','GATA6']:
                q=g[g.gene==name];row=dict(gene=name,observed=float(q.observed.iloc[0]))
                row.update({str(v):float(gg.predicted.iloc[0]) for v,gg in q.groupby('version')})
                marker_rows.append(row)
            body+='day1 标记绝对均值（端点选择方案）：\n\n'
            body+=table(marker_rows,[('gene','基因'),('observed','真实'),('native','native'),
                ('xstart','xstart'),('minsnr','minsnr'),('epsres','epsres')])+'\n\n'
        base=pd.read_csv(ROOT/'results/round2/distribution_baselines.csv')
        base=base[base.panel=='source']
        base_rows=[dict(method=method,day=int(day),W=gg.mean_gene_wasserstein.mean(),
                       E=gg.energy_distance.mean()) for (method,day),gg in base.groupby(['method','day'])]
        body+='同一 source 面板的简单分布基线沿用第二轮已保存的 3 次抽样均值，没有新增训练：`cell_linear` 对随机端点细胞逐细胞表达插值，`endpoint_mixture` 只混合端点细胞。\n\n'
        body+=table(base_rows,[('method','基线'),('day','日'),('W','逐基因 W'),('E','联合 energy')])+'\n\n'
        candidates=[]
        for v in ['xstart','minsnr','epsres']:
            cc=pd.read_csv(ROOT/f'results/round3/{v}_s42/eval_final_1000/candidates.csv')
            candidates.extend(dict(variant=v,**row) for row in cc.to_dict('records'))
        body+='完整端点候选结果（仅端点验证；主选择指标为 W，MSE 同时记录）：\n\n'
        body+=table(candidates,[('variant','版本'),('mode','方式'),('sigma','σ/系数'),
            ('wasserstein','端点 W'),('mse','端点均值 MSE')])+'\n'
        text=replace(text,'TRAINING_RESULTS',body)
    diag=ROOT/'results/round3/diagnostics'
    if (diag/'complete.json').exists():
        f=pd.read_csv(diag/'metrics.csv')
        rows=[]
        for _,q in f.iterrows():
            row=q.to_dict();row['tp_fp']=f'{int(q.transient_tp)}/{int(q.transient_fp)}';rows.append(row)
        body='## 固定主结果后的 day1 条件诊断\n\n'
        body+='以下诊断不改变主方案或采样选择。`selected_replay` 重放已保存的主预测；`mean_direction_removed` 保留相同潜变量残差、相同初始 xT，仅减去 α·Δz 的均值方向；`target_rna_oracle` 将真实 day1 RNA 提供给编码器，仅检查未见状态的条件生成能力。它不是预测成绩。\n\n'
        body+=table(rows,[('variant','版本'),('method','诊断'),('r2','R²'),('T_pred','T 均值'),
            ('tp_fp','瞬态 TP/FP'),('energy_distance','联合 energy')])+'\n\n'
        done=json.loads((diag/'complete.json').read_text())
        body+='与保存的主预测重放最大差：'+', '.join(f'{v["variant"]}={v["selected_replay_max_error"]:.3g}' for v in done['replays'])+'。\n'
        text=replace(text,'DIAGNOSTICS',body)
    p.write_text(text)


if __name__=='__main__':main()
