# Squidiff 本地复现

已在本机 **NVIDIA A100 40GB**、独立 conda 环境 **`squidiff`** 中完成 Norman 与 iPSC 两组训练，均为 **100000 次优化迭代**。使用作者公开模型，从头训练，未使用作者预训练权重。完成时间：2026-11-13 13:33（北京时间）。

**科学复现状态：未通过核心结论验证。** 已完成的是第一版训练与评估流程；总体表达均值指标接近公开 notebook，不能视为已经复现非加性交互、未见瞬态状态或稳定优于简单基线的结论。以下保留原始结果，并补充使用全部训练细胞精确均值的审计。

## 客观复核与下一步

原线性基线只抽取部分端点/控制细胞，其均值存在抽样误差。改用全部训练细胞的精确均值后：

| 任务 | 模型R²（采样平均） | 精确线性均值基线R² | 模型相对基线MSE |
|---|---:|---:|---:|
| Norman 双扰动 | 0.98626 | 0.98590 | 降低2.55% |
| iPSC day1 | 0.82755 | 0.82547 | 降低1.19% |
| iPSC day2 | 0.86261 | 0.87323 | 升高8.38% |

进一步计算真正的交互残差 `I = mean(AB) - mean(A) - mean(B) + mean(ctrl)`：Norman的交互残差R²仅 **0.01094**（10次采样均值对应的MSE），首轮交互相关性约0.238。强效应基因变化方向正确不能替代非加性交互的恢复。

iPSC首轮T的day1真实均值为 **6.238**，预测为 **0.309**。额外使用真实day1表达作为编码器输入，生成的T均值也只有 **0.401**。这是已读取测试表达的事后oracle诊断，不能计作预测成绩；它提示当前所学表示/生成器对中间状态泛化不足，不能把问题仅归于插值系数或采样噪声。端点重建好不等于中间状态预测成功。现有逐基因Wasserstein在day1/day2也均劣于细胞级线性基线。单训练种子及多次噪声采样不足以建立稳健性或显著性。

审计结果：`results/audit_v1/exact_baselines.csv`、`results/audit_v1/oracle_ipsc.json`。原始v1结果未覆盖。

下一步优先级：

1. 对齐作者原始基因ID、预处理/归一化、细胞划分及对应权重；分开记录原代码字面行为、论文意图公式和严格无泄漏协议，避免将混合改动的v1直接视作论文原实验。
2. 做受控数值/实现检查：FP32对照、潜变量形状与代数、BatchNorm推理状态、噪声尺度、DDIM与非负截断；检查实际中间状态的表示与重建，定位信息丢失环节。
3. 预先固定验收：非加性交互残差、瞬态基因峰值及分布质量相对强基线的增益；运行独立训练种子并报告全部结果。当前已查看的测试结果作为开发诊断，改进后的最终检验使用新的未查看分区/任务。
4. 若严格对齐后仍失败，保留复现失败结论。增加时间点、外部发育轨迹或基因交互先验可作为新方法研究，但属于改变信息条件的扩展，不能用来反证原始端点预测设定已成功复现。

## 结果

下表为固定训练种子42、1000步DDIM推理的采样均值。Norman重复10次，iPSC重复6次；这些是**采样重复，不是独立训练重复**。R²采用真实表达作为参考。

| 实验 | Pearson | R² | 扰动增量R² |
|---|---:|---:|---:|
| Norman：PTPN12+ZBTB25留出 | 0.99317 | 0.98626 | 0.94377 |
| iPSC：day1预测 | 0.91315 | 0.82755 | 0.18069 |
| iPSC：day2预测 | 0.93889 | 0.86261 | 0.66620 |
| iPSC：day0编码重建 | 0.99898 | 0.99713 | — |
| iPSC：day3编码重建 | 0.99805 | 0.99482 | — |

表达均值指标达到公开 notebook 的量级（其 Norman Pearson≈0.9914、R²≈0.9826；iPSC day1/day2 Pearson≈0.8645/0.9125）。原 iPSC notebook 的R²参数顺序颠倒，输出中同时保留该口径，便于核对。基因表和划分存在下述差异，不能把数值差异解释为显著优于论文。

**边界：**Norman与表达线性相加基线接近（基线R²=0.98583）；iPSC day1略优于线性基线（0.82378），day2略低于线性基线（0.87164）。NANOG下降、GATA6上升的总体方向得到恢复，但 **T的day1瞬态峰值未恢复**。训练端点中T均值仅0.073/0.282。Norman的弱效应基因方向也不稳定：首轮全部基因反向比例约72%，但21个真实绝对增量≥0.1的基因方向全部正确。完整分布指标及逐基因结果均保留，不能将总体相关性达标视为全部生物学结论已复现。

## 资料与协议

- [正式论文](https://www.nature.com/articles/s41592-025-02877-y)及补充材料：`docs/supplement.pdf`。
- [模型源码](https://github.com/siyuh/Squidiff)：`upstream/Squidiff`，提交 `abdfc27d84947dcccd745d1067c0840a41d32eb8`。
- [作者复现 notebook](https://github.com/siyuh/Squidiff_reproducibility)：`upstream/Squidiff_reproducibility`，提交 `7ce6d0291ee489d1fd89dda5abe8d4f993450d6e`。共49个源码/notebook文件已按Git对象哈希核验一致。
- 数据：Norman使用GEARS公开预处理的GSE133344（[Dataverse 6154020](https://dataverse.harvard.edu/api/access/datafile/6154020)）；iPSC来自[Zenodo 3625024](https://zenodo.org/records/3625024)，下载受限后改用[作者Figshare镜像](https://doi.org/10.6084/m9.figshare.27948633)。文件已校验MD5，来源见 `data/raw/manifest.json`。

保留作者2048隐层、3个MLP块、60维语义编码器与非负采样截断；1000步线性扩散，beta=0.0001→0.02，batch=64，AdamW（weight decay=0），初始lr=1e-4并线性衰减。使用PyTorch 2.5.1 CUDA 12.1、Scanpy 1.10.4及BF16自动混合精度/fused AdamW适配本机。

**数据划分与差异：**

- Norman：控制组及两种单扰动用于训练/验证（7102/788个细胞），257个双扰动细胞完全留出。当前作者预处理代码的1000 HVG与保存的203维模型不一致；本次用训练细胞的200 HVG并集其7个标记基因，实际为**206维**。采用论文双扰动潜变量公式，修正notebook中向量形状及重复加控制变量的问题。
- iPSC：每天无放回抽2000个细胞，按1200/200/600分为发现/验证/测试分区，避免原有放回抽样造成重复细胞泄漏。200 HVG与NANOG/T/GATA6并集为**203维**；HVG沿用作者四天发现分区，day1/2的2400个发现细胞只用于基因筛选，不进入模型训练或测试。因此这不是完全不接触中间条件的特征发现设置。模型仅训练day0/3共2400个细胞，端点验证400个，四天测试各600个。
- 检查点和噪声只用训练条件的验证集选择。iPSC改用端点生成均值MSE选择检查点，再用逐基因Wasserstein校准噪声；最终为100000步、sigma=0.5。选择过程在读取day1/2测试数据之前完成。

## 运行与恢复

```bash
bash scripts/setup_env.sh
conda activate squidiff
python scripts/download_data.py
python scripts/prepare.py norman
python scripts/prepare.py ipsc
python scripts/pipeline.py
python scripts/status.py
python scripts/report.py
```

单组续训：`python scripts/train.py norman --steps 100000`（iPSC替换数据集名）。每2000步原子保存模型、EMA、优化器及Python/NumPy/CPU/CUDA随机状态，保留 `latest.pt`、`previous.pt`，并保存阶段模型；SIGTERM/SIGINT会在当前优化步结束后保存。训练使用独立进程，已有断点自动恢复。Norman已实际验证从20000步恢复到100000步。

## 文件

- `results/summary.csv`：两组指标与采样标准差；`results/all_metrics.csv`：所有采样与基线明细。
- `results/{norman,ipsc}/v1/`：完整断点、阶段模型、训练日志、预测NPZ、逐基因CSV及PNG；Norman用 `best.pt`，iPSC用 `best_generation.pt`。
- iPSC的 `generation_selection.json`、`generation_validation.csv/png`：验证选择依据；`marker_trends.png`：标记基因趋势。
- `results/training_curves.png`：训练曲线；`logs/`：下载、训练、评估和后台流程日志。`logs/pipeline_complete.json`确认流程结束。
- `environment.yml`、`requirements-lock.txt`：环境记录；`docs/source_verification.json`：源码核验；数据划分/基因表在 `data/processed/*/manifest.json`。

## 关键时间记录（2026-11-13，北京时间）

- **12:54**：开始，创建Python 3.10环境 `squidiff`，核对空闲A100及磁盘。
- **13:02–13:04**：环境、公开数据下载与MD5校验完成；源码固定。修复原入口恢复被注释、覆盖模型、保留loss计算图等问题，另写本地入口。服务重启后检查状态，改用独立进程持续运行。
- **13:03**：启动Norman第一版训练。**13:06**：完成20000步并从完整断点恢复。
- **13:18**：Norman完成100000步、10次采样及基线分析；启动iPSC训练。
- **13:24**：iPSC训练损失下降而验证去噪损失上升，检查端点实际生成：4000步模型的生成均值MSE约35.8，后期明显更好。决定增加端点生成验证，避免按去噪MSE误选早期模型。诊断保存在 `results/ipsc/validation_diagnostic.json`。
- **13:31**：iPSC完成100000步。候选模型先以100步DDIM筛选，再用1000步确认；100000步端点均值MSE约0.0188。
- **13:32**：仅用端点验证校准潜变量噪声至0.5，平均Wasserstein=0.5052（sigma=0.7时0.6951）。随后首次运行中间时间点测试。
- **13:33–13:37**：完成iPSC六次采样、基线/标记基因分析及汇总。核验两组均训练100000步、预测有限、采样数量完整；GPU已释放。记录T峰值和弱效应预测的未复现部分。

- **14:22（复核）**：将v1科学状态修正为“未通过核心结论验证”；用全部训练细胞精确均值重算线性基线，补充非加性交互残差与真实中间状态编码的oracle诊断。未启动新训练，保留原始v1输出。

- 2026-11-13 18:42:37：开始训练 ipsc/fp32_tf32_s42：2400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 18:49:06：研究数据 source_features/norman 准备完成：train=(7102, 203), test=(257, 203)，训练/测试重合细胞ID=0。

- 2026-11-13 18:49:08：研究数据 source_1000/norman 准备完成：train=(7102, 1000), test=(257, 1000)，训练/测试重合细胞ID=0。

- 2026-11-13 18:49:08：研究数据 combo_seen/norman 准备完成：train=(7307, 206), test=(52, 206)，训练/测试重合细胞ID=0。

- 2026-11-13 18:52:35：研究评估 ipsc/v1/raw_controls 完成：{"('cell_shift', 'day1')": {"pearson": 0.908218104271844, "r2": 0.8199201077394852, "relative_mse_gain": -0.03180604454127134}, "('cell_shift', 'day2')": {"pearson": 0.9231454976781976, "r2": 0.8447319240215063, "relative_mse_gain": -0.22478683259090437}, "('centroid_sigma0', 'day1')": {"pearson": 0.8888768317479006, "r2": 0.7691305295130721, "relative_mse_gain": -0.3228157356060708}, "('centroid_sigma0', 'day2')": {"pearson": 0.8870121868747612, "r2": 0.7634010923521474, "relative_mse_gain": -0.8663413252613457}, "('centroid_sigma0.5', 'day1')": {"pearson": 0.9134645898143854, "r2": 0.8276762375002952, "relative_mse_gain": 0.012634350602195199}, "('centroid_sigma0.5', 'day2')": {"pearson": 0.9410927956752723, "r2": 0.8658753855276661, "relative_mse_gain": -0.058002816720674755}, "('centroid_sigma0.7', 'day1')": {"pearson": 0.8942708100117241, "r2": 0.7008506266931469, "relative_mse_gain": -0.7140399615089255}, "('centroid_sigma0.7', 'day2')": {"pearson": 0.938619358752918, "r2": 0.7258178094788823, "relative_mse_gain": -1.1628060666360645}, "('oracle', 'day1')": {"pearson": 0.944147425306245, "r2": 0.8903187479965811, "relative_mse_gain": 0.3715579381496762}, "('oracle', 'day2')": {"pearson": 0.9777833950873885, "r2": 0.9527387396112336, "relative_mse_gain": 0.6271933618612698}}

- 2026-11-13 18:53:23：研究评估 ipsc/v1/ema_controls 完成：{"('cell_shift', 'day1')": {"pearson": 0.9086172737803505, "r2": 0.821538829189812, "relative_mse_gain": -0.022531235699984276}, "('cell_shift', 'day2')": {"pearson": 0.9238734086874518, "r2": 0.8470905656749146, "relative_mse_gain": -0.2061813773375325}, "('centroid_sigma0', 'day1')": {"pearson": 0.8901282112772668, "r2": 0.7742150849110654, "relative_mse_gain": -0.2936826940010435}, "('centroid_sigma0', 'day2')": {"pearson": 0.891749216242519, "r2": 0.7756495773886847, "relative_mse_gain": -0.7697227312754482}, "('centroid_sigma0.5', 'day1')": {"pearson": 0.9134458492538516, "r2": 0.8276674651380435, "relative_mse_gain": 0.012584087486848916}, "('centroid_sigma0.5', 'day2')": {"pearson": 0.9413946996471025, "r2": 0.8666131988108272, "relative_mse_gain": -0.052182792298838265}, "('centroid_sigma0.7', 'day1')": {"pearson": 0.8938567197869113, "r2": 0.6948285460233984, "relative_mse_gain": -0.748544753564069}, "('centroid_sigma0.7', 'day2')": {"pearson": 0.9382102960366228, "r2": 0.7175459126286945, "relative_mse_gain": -1.2280565070682847}, "('oracle', 'day1')": {"pearson": 0.9443114585349934, "r2": 0.890938686374646, "relative_mse_gain": 0.375110006943704}, "('oracle', 'day2')": {"pearson": 0.9778963290737905, "r2": 0.9529723457386738, "relative_mse_gain": 0.6290360955146514}}

- 2026-11-13 18:53:43：研究数据 all_days/ipsc 准备完成：train=(6400, 203), test=(2400, 203)，训练/测试重合细胞ID=1200。

- 2026-11-13 18:53:45：研究数据 source/ipsc 准备完成：train=(2400, 203), test=(2400, 203)，训练/测试重合细胞ID=163。

- 2026-11-13 18:53:46：研究数据 fresh_endpoints/ipsc 准备完成：train=(2400, 203), test=(2400, 203)，训练/测试重合细胞ID=0。

- 2026-11-13 18:54:17：研究数据 source_features/norman 准备完成：train=(7102, 203), test=(257, 203)，训练/测试重合细胞ID=0。

- 2026-11-13 18:54:19：研究数据 source_1000/norman 准备完成：train=(7102, 1000), test=(257, 1000)，训练/测试重合细胞ID=0。

- 2026-11-13 18:54:19：研究数据 combo_seen/norman 准备完成：train=(7307, 206), test=(52, 206)，训练/测试重合细胞ID=0。

- 2026-11-13 18:54:49：ipsc/fp32_tf32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.502379。

- 2026-11-13 18:57:04：研究评估 ipsc/fp32_tf32_s42/raw_controls 完成：{"('cell_shift', 'day1')": {"pearson": 0.9072785981775335, "r2": 0.8175815476467472, "relative_mse_gain": -0.04520532198930183}, "('cell_shift', 'day2')": {"pearson": 0.9218078632638269, "r2": 0.8412139994692753, "relative_mse_gain": -0.2525369521339422}, "('centroid_sigma0', 'day1')": {"pearson": 0.8855657160516633, "r2": 0.7586202484156481, "relative_mse_gain": -0.3830366266229399}, "('centroid_sigma0', 'day2')": {"pearson": 0.8788905750360967, "r2": 0.7413500690657233, "relative_mse_gain": -1.040284376955427}, "('centroid_sigma0.5', 'day1')": {"pearson": 0.9137047060870814, "r2": 0.8283776548037255, "relative_mse_gain": 0.01665327022914742}, "('centroid_sigma0.5', 'day2')": {"pearson": 0.9406394154762756, "r2": 0.8655402052146565, "relative_mse_gain": -0.06064678864684647}, "('centroid_sigma0.7', 'day1')": {"pearson": 0.8955749821239933, "r2": 0.7073159867410477, "relative_mse_gain": -0.676995305973987}, "('centroid_sigma0.7', 'day2')": {"pearson": 0.9387736423064673, "r2": 0.7324079571598048, "relative_mse_gain": -1.1108216129513209}, "('oracle', 'day1')": {"pearson": 0.9438868583120881, "r2": 0.8893570274496828, "relative_mse_gain": 0.3660475557244489}, "('oracle', 'day2')": {"pearson": 0.9771698282561672, "r2": 0.951548592494512, "relative_mse_gain": 0.6178052342102995}}

- 2026-11-13 18:57:12：开始训练 ipsc/all_days_fp32_s42：6400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 20000 步。

- 2026-11-13 18:58:08：研究评估 norman/v1/inference_controls 完成：{"('cell_shift', 'AB')": {"pearson": 0.9931716701744393, "r2": 0.9862878898753267, "relative_mse_gain": 0.02741367483202295}, "('centroid', 'AB')": {"pearson": 0.9933318032271953, "r2": 0.9864329280210348, "relative_mse_gain": 0.037701086168491105}, "('literal', 'AB')": {"pearson": 0.99344412839073, "r2": 0.9866816252746843, "relative_mse_gain": 0.05534093487209213}, "('oracle', 'AB')": {"pearson": 0.9997736958081184, "r2": 0.9993392306028877, "relative_mse_gain": 0.9531322842452512}}

- 2026-11-13 18:59:39：ipsc/all_days_fp32_s42 trained，已训练 20000 步，验证集去噪 MSE=0.399305。

- 2026-11-13 19:00:45：研究数据 all_days/ipsc 准备完成：train=(4800, 203), test=(2400, 203)，训练/测试重合细胞ID=0。

- 2026-11-13 19:00:46：研究数据 source/ipsc 准备完成：train=(2400, 203), test=(2400, 203)，训练/测试重合细胞ID=163。

- 2026-11-13 19:00:47：研究数据 fresh_endpoints/ipsc 准备完成：train=(2400, 203), test=(2400, 203)，训练/测试重合细胞ID=0。

- 2026-11-13 19:00:57：开始训练 ipsc/all_days_fp32_s42：4800 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 20000 步。

- 2026-11-13 19:03:10：ipsc/all_days_fp32_s42 trained，已训练 20000 步，验证集去噪 MSE=0.424689。

- 2026-11-13 19:05:00：研究评估 ipsc/all_days_fp32_s42/raw_controls 完成：{"('cell_shift', 'day1')": {"pearson": 0.8914004187209421, "r2": 0.44652241756780464, "relative_mse_gain": -2.1712675296665926}, "('cell_shift', 'day2')": {"pearson": 0.9268728272346609, "r2": 0.385640060356886, "relative_mse_gain": -3.8461988068329793}, "('centroid_sigma0', 'day1')": {"pearson": 0.8994836755068037, "r2": 0.6515821429325723, "relative_mse_gain": -0.9963342183046957}, "('centroid_sigma0', 'day2')": {"pearson": 0.9346149844621312, "r2": 0.6348337706689547, "relative_mse_gain": -1.8805070622082267}, "('centroid_sigma0.5', 'day1')": {"pearson": 0.8750093964777429, "r2": 0.2663190821806968, "relative_mse_gain": -3.2037808678572386}, "('centroid_sigma0.5', 'day2')": {"pearson": 0.9196659599390073, "r2": 0.1756881806334849, "relative_mse_gain": -5.502342841222563}, "('centroid_sigma0.7', 'day1')": {"pearson": 0.8274592554237332, "r2": -0.5624932700249208, "relative_mse_gain": -7.952637522874991}, "('centroid_sigma0.7', 'day2')": {"pearson": 0.8820009297516141, "r2": -0.777456809063453, "relative_mse_gain": -13.020948488738282}, "('oracle', 'day1')": {"pearson": 0.9637411956956243, "r2": 0.6130263185484399, "relative_mse_gain": -1.217248014689404}, "('oracle', 'day2')": {"pearson": 0.9715069099473241, "r2": 0.4455923691779641, "relative_mse_gain": -3.3732825427218005}, "('target_centroid', 'day1')": {"pearson": 0.9749608107223933, "r2": 0.8176536359347166, "relative_mse_gain": -0.044792276810665665}, "('target_centroid', 'day2')": {"pearson": 0.9792591746839125, "r2": 0.7733247333282568, "relative_mse_gain": -0.7880615840956093}}

- 2026-11-13 19:05:05：ipsc/all_days_fp32_s42 从第 20000 步完整恢复。

- 2026-11-13 19:07:12：采样负对照 ipsc/v1 完成：统一四天的潜变量抽样方式，分离方向移动与额外潜变量噪声。

- 2026-11-13 19:07:37：Norman bootstrap参考审计 v1 完成：比较原notebook只使用控制组前257个细胞与全部6618控制细胞。

- 2026-11-13 19:11:21：Norman bootstrap参考审计 v1 完成：比较原notebook只使用控制组前257个细胞与全部6618控制细胞。


## 第二轮受控排查（进行中）

- 18:37：开始逐项核验。系统盘约81GB可用；保留v1，另建研究数据和实验目录。
- 实现审计：与原训练循环5步损失相同、参数最大差约3.3e-6；原/本地采样相同噪声输出完全一致；恢复后的3步损失与权重逐位一致。扩散代数及1000步时间映射通过，见 `results/research/implementation_audit.json`。
- 数据审计：v1的全部训练/验证/测试数值与公开CSV重取后完全一致，细胞顺序和203个HVG可重新生成。source近似协议重抽样后HVG与v1仅重合91/203；说明仅对齐维数不足以保证同一实验。Norman按全四条件筛200HVG并集标记基因确为203维，已单列对照。
- 作者公开VO权重为596维，配套大AnnData为BVO；未在三个Figshare版本及GitHub releases中找到本任务对应权重。已记录元数据并删除无关的2.77GB下载，不能将VO权重硬套本任务。
- 第一组FP32配对训练保持v1数据、种子与抽样不变，完成100000步；T day1仍约0.34（真实6.24），暂不支持BF16导致主要失败。
- 新建阳性对照首版误把全部2000个中间日样本纳入训练，检查发现与测试重合1200个ID；该运行已作废并删除权重，留存 `results/rejected/all_days_overlap_s42/EXCLUDED.json`。已修成仅用1200个发现分区细胞，训练4800、与测试重合0，从头重训；v1不含这个新建对照错误。
- 原作者固定sigma=0.7可产生T的弱隆起；v1校准到0.5后弱隆起消失，故不能将两者混为同一复现协议。正在检查统一四天抽样方式后的曲线和无方向移动负对照。
- 原notebook将目标抽样下标重用于控制组，实际只访问前257个控制细胞。当前模型按该bootstrap口径可得到约8%的反向比例；全控制均值口径则大幅更高。正确从全控制池抽小样本也会降低该比例，须区分索引问题与共享参考噪声，不能把差异全部归为索引bug。原“反向”定义也不惩罚零变化。见 `results/research/bootstrap_audit/`。
- 后续固定问题/队列见 `configs/research_protocol.json`、`configs/research_stage1.json`；所有oracle和提供中间状态的阳性对照都只作诊断，不计为零样本预测成功。

- 2026-11-13 19:14:25：采样负对照 ipsc/v1 完成：统一四天的潜变量抽样方式，分离方向移动与额外潜变量噪声。

- 2026-11-13 19:14:35：非加性交互信号审计 v1 完成：四组分别独立拆半，评估残差信号的细胞抽样重复性；不替代生物学重复。

- 2026-11-13 19:16:32：ipsc/all_days_fp32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.424689。

- 2026-11-13 19:16:56：群体状态诊断 ipsc/v1 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 19:17:49：研究评估 ipsc/all_days_fp32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 19:17:54：开始训练 norman/fp32_tf32_s42：7102 个细胞、206 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 19:18:14：群体状态诊断 ipsc/v1 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 19:24:49：群体状态诊断 ipsc/all_days_fp32_s42 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 19:28:39：norman/fp32_tf32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.010815。

- 2026-11-13 19:29:00：研究评估 norman/fp32_tf32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 19:29:05：开始训练 ipsc/fp32_ieee_s42：2400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 19:30:24：非加性交互信号审计 fp32_tf32_s42 完成：四组分别独立拆半，评估残差信号的细胞抽样重复性；不替代生物学重复。

- 2026-11-13 19:35:46：开始训练 ipsc/source_fp32_s43：2400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 19:36:34：ipsc/source_fp32_s43 interrupted，已训练 2435 步，验证集去噪 MSE=inf。

- 2026-11-13 19:40:36：ipsc/fp32_ieee_s42 trained，已训练 100000 步，验证集去噪 MSE=0.502104。

- 2026-11-13 19:41:42：研究评估 ipsc/fp32_ieee_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 19:41:47：开始训练 ipsc/source_fp32_s42：2400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 19:46:04：ipsc/source_fp32_s42 interrupted，已训练 22282 步，验证集去噪 MSE=0.522793。

- 2026-11-13 19:46:19：开始训练 norman/source_fp32_s42：7890 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 19:46:20：ipsc/source_fp32_s43 从第 2435 步完整恢复。

- 2026-11-13 19:46:20：ipsc/source_fp32_s42 从第 22282 步完整恢复。

- 2026-11-13 19:47:47：Norman bootstrap参考审计 v1 完成：比较原notebook只使用控制组前257个细胞与全部6618控制细胞。

- 2026-11-13 19:50:31：研究数据 full_endpoints/ipsc 准备完成：train=(16546, 203), test=(2400, 203)，训练/测试重合细胞ID=0。

- 2026-11-13 19:56:56：研究评估 ipsc/source_fp32_s42/step050000_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:02:43：研究评估 norman/source_fp32_s42/step050000_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:09:31：ipsc/source_fp32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.522793。

- 2026-11-13 20:11:18：研究评估 ipsc/source_fp32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:11:22：开始训练 norman/source_fp32_s43：7890 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 20:13:36：ipsc/source_fp32_s43 trained，已训练 100000 步，验证集去噪 MSE=0.505654。

- 2026-11-13 20:15:48：研究评估 ipsc/source_fp32_s43/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:15:53：开始训练 norman/source_fp32_s44：7890 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 20:16:46：采样负对照 ipsc/source_fp32_s42 完成：统一四天的潜变量抽样方式，分离方向移动与额外潜变量噪声。

- 2026-11-13 20:17:32：norman/source_fp32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.008465。

- 2026-11-13 20:18:07：研究评估 norman/source_fp32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:18:12：开始训练 ipsc/source_fp32_s44：2400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 20:19:46：非加性交互信号审计 source_fp32_s42 完成：四组分别独立拆半，评估残差信号的细胞抽样重复性；不替代生物学重复。

- 2026-11-13 20:19:47：Norman bootstrap参考审计 source_fp32_s42 完成：比较原notebook只使用控制组前257个细胞与全部7353控制细胞。

- 2026-11-13 20:19:54：群体状态诊断 ipsc/source_fp32_s42 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 20:28:29：开始训练 norman/balanced_fp32_s42：7890 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 20:33:38：研究评估 norman/source_fp32_s42/all_control_cells 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:44:58：norman/source_fp32_s43 trained，已训练 100000 步，验证集去噪 MSE=0.008865。

- 2026-11-13 20:45:36：研究评估 norman/source_fp32_s43/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:45:40：开始训练 ipsc/fresh_endpoints_fp32_s42：2400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 20:47:48：norman/source_fp32_s44 trained，已训练 100000 步，验证集去噪 MSE=0.009406。

- 2026-11-13 20:48:25：研究评估 norman/source_fp32_s44/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:48:30：开始训练 ipsc/full_endpoints_fp32_s42：16546 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 20:51:28：ipsc/source_fp32_s44 trained，已训练 100000 步，验证集去噪 MSE=0.520851。

- 2026-11-13 20:53:29：研究评估 ipsc/source_fp32_s44/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 20:53:33：开始训练 norman/source1000_fp32_s42：7890 个细胞、1000 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 21:04:41：norman/balanced_fp32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.008192。

- 2026-11-13 21:05:20：研究评估 norman/balanced_fp32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:08:12：ipsc/full_endpoints_fp32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.289657。

- 2026-11-13 21:08:26：研究评估 norman/balanced_fp32_s42/all_control_cells 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:09:54：研究评估 ipsc/full_endpoints_fp32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:15:00：source_fp32_s42 完成仅端点验证的噪声校准：sigma=0.4227；未读取中间日测试数据选参数。

- 2026-11-13 21:15:23：研究评估 ipsc/source_fp32_s42/endpoint_validated_noise 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:17:41：source_fp32_s43 完成仅端点验证的噪声校准：sigma=0.4289；未读取中间日测试数据选参数。

- 2026-11-13 21:18:03：研究评估 ipsc/source_fp32_s43/endpoint_validated_noise 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:18:34：ipsc/fresh_endpoints_fp32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.499348。

- 2026-11-13 21:19:22：群体状态诊断 ipsc/source_fp32_s43 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 21:19:22：群体状态诊断 ipsc/source_fp32_s42 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 21:20:29：研究评估 ipsc/fresh_endpoints_fp32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:20:33：source_fp32_s44 完成仅端点验证的噪声校准：sigma=0.4246；未读取中间日测试数据选参数。

- 2026-11-13 21:20:52：研究评估 ipsc/source_fp32_s44/endpoint_validated_noise 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:22:46：full_endpoints_fp32_s42 完成仅端点验证的噪声校准：sigma=0.4617；未读取中间日测试数据选参数。

- 2026-11-13 21:23:05：研究评估 ipsc/full_endpoints_fp32_s42/endpoint_validated_noise 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:23:58：norman/source1000_fp32_s42 trained，已训练 100000 步，验证集去噪 MSE=0.099020。

- 2026-11-13 21:24:31：研究评估 norman/source1000_fp32_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:25:34：fresh_endpoints_fp32_s42 完成仅端点验证的噪声校准：sigma=0.5437；未读取中间日测试数据选参数。

- 2026-11-13 21:25:56：研究评估 ipsc/fresh_endpoints_fp32_s42/endpoint_validated_noise 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:27:06：采样负对照 ipsc/source_fp32_s43 完成：统一四天的潜变量抽样方式，分离方向移动与额外潜变量噪声。

- 2026-11-13 21:27:18：群体状态诊断 ipsc/source_fp32_s43 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 21:27:23：Norman bootstrap参考审计 source_fp32_s43 完成：比较原notebook只使用控制组前257个细胞与全部7353控制细胞。

- 2026-11-13 21:27:26：非加性交互信号审计 source_fp32_s43 完成：四组分别独立拆半，评估残差信号的细胞抽样重复性；不替代生物学重复。

- 2026-11-13 21:28:45：研究评估 norman/source_fp32_s43/all_control_cells 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:30:44：采样负对照 ipsc/source_fp32_s44 完成：统一四天的潜变量抽样方式，分离方向移动与额外潜变量噪声。

- 2026-11-13 21:30:56：群体状态诊断 ipsc/source_fp32_s44 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 21:31:01：Norman bootstrap参考审计 source_fp32_s44 完成：比较原notebook只使用控制组前257个细胞与全部7353控制细胞。

- 2026-11-13 21:31:04：非加性交互信号审计 source_fp32_s44 完成：四组分别独立拆半，评估残差信号的细胞抽样重复性；不替代生物学重复。

- 2026-11-13 21:32:19：研究评估 norman/source_fp32_s44/all_control_cells 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:34:13：研究评估 norman/source1000_fp32_s42/all_control_cells 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:35:21：研究评估 ipsc/source_fp32_s42/ema_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:36:00：研究评估 norman/source_fp32_s42/ema_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:36:00：开始训练 norman/source1000_epsres_s42：7890 个细胞、1000 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，fp32，计划 100000 步。

- 2026-11-13 21:36:19：群体状态诊断 ipsc/source_fp32_s44 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 21:36:30：群体状态诊断 ipsc/fresh_endpoints_fp32_s42 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 21:36:42：群体状态诊断 ipsc/full_endpoints_fp32_s42 完成；分类器仅用于事后评价，参考细胞与真实评估细胞不重合。

- 2026-11-13 21:52:45：norman/source1000_epsres_s42 trained，已训练 100000 步，验证集去噪 MSE=0.230477。

- 2026-11-13 21:53:10：研究评估 norman/source1000_epsres_s42/raw_controls 完成；详细指标见对应 research_eval 目录。

- 2026-11-13 21:55:23：研究评估 norman/source1000_epsres_s42/all_control_cells 完成；详细指标见对应 research_eval 目录。
