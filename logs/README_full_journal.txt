# Squidiff 本地复现

使用作者公开实现，在本机 NVIDIA A100 40GB 上完成 Norman 双基因扰动与 iPSC 分化两组实验。独立 conda 环境为 `squidiff`，两组各100000步训练已完成；Norman已完成评估，iPSC正在进行最终端点校准与测试。

## 资料与目录

- 论文：[Nature Methods](https://www.nature.com/articles/s41592-025-02877-y)，同时核对[公开预印本](https://pmc.ncbi.nlm.nih.gov/articles/PMC12407682/)。
- 作者代码：`upstream/Squidiff`（https://github.com/siyuh/Squidiff）。
- 作者复现：`upstream/Squidiff_reproducibility`（https://github.com/siyuh/Squidiff_reproducibility）。
- 原始数据：iPSC 为 Zenodo 3625024；Norman 为 GSE133344。优先使用作者提供的预处理数据以匹配公开实验，保留来源和校验值。

## 实验原则

固定数据划分和随机种子；测试集不参与模型训练或参数选择。保存可恢复的模型、优化器、随机数状态及训练日志。分别报告表达均值、扰动增量和分布指标，避免只靠全基因相关性判断效果。训练故障、协议差异和性能不足均如实记录。

## 使用

```bash
bash scripts/setup_env.sh
conda activate squidiff
python scripts/download_data.py          # 可续传并校验；已完成的下载会跳过
python scripts/prepare.py norman
python scripts/prepare.py ipsc
python scripts/pipeline.py               # 顺序完成两组 10 万步训练及评估
python scripts/status.py                 # 查看持久化进度
python scripts/report.py                 # 汇总 CSV 与训练曲线
```

单组继续训练：`python scripts/train.py norman --steps 100000`（iPSC 替换数据集名）。已有 `latest.pt` 时自动恢复；最新文件缺失则尝试 `previous.pt`。训练中发送 SIGTERM/SIGINT 会在当前优化步结束后保存。`logs/pipeline.pid` 记录本次独立运行进程，查看日志无需依赖当前会话。

输出在 `results/{norman,ipsc}/v1/`：`latest.pt`/`previous.pt` 为完整断点，`best.pt` 按验证去噪损失选取，iPSC 的 `best_generation.pt` 则按端点验证生成质量选取，`step_*.pt` 为阶段模型；`train.jsonl` 为训练记录，`metrics_1000steps.csv` 和 `evaluation_1000steps.json` 为最终指标，`predictions_1000steps.npz`、逐基因 CSV、PNG 为预测与图。环境锁定在 `environment.yml`、`requirements-lock.txt`，原始数据来源与校验在 `data/raw/manifest.json`，源码核验在 `docs/source_verification.json`。

## 关键记录（北京时间）

- 2026-11-13 12:54：开始复现。当前目录为空；本机 A100-PCIE-40GB 空闲，磁盘约 98GB 可用，已有 conda，无 git。启动创建 `squidiff`（Python 3.10）。通过 GitHub 源码归档下载作者仓库，稍后固定版本与校验值。

- 2026-11-13 13:02：环境安装完成，使用 PyTorch 2.5.1 CUDA 12.1、Scanpy 1.10.4。源码固定为 Squidiff `abdfc27d84947dcccd745d1067c0840a41d32eb8` 和复现仓库 `7ce6d0291ee489d1fd89dda5abe8d4f993450d6e`。Zenodo 文件接口返回 403，iPSC 改从作者 Figshare 27948633 镜像下载；Norman 使用作者 notebook 所读的 GEARS 预处理公开数据（Dataverse 6154020）。
- 2026-11-13 13:02：核对后保留作者 MLP（2048 隐层、3 个块、60 维语义变量、1000 步线性扩散），另写本地训练/评估入口。修复原入口无法真正恢复、模型覆盖、loss 保留计算图、iPSC R² 参数颠倒等问题；每 2000 步保存模型/EMA/优化器/随机状态，并保留上一次完整检查点。初版检查点按独立验证集去噪损失选择，iPSC 后续增加实际生成验证（见下）。
- 2026-11-13 13:02：协议差异：Norman 当前 prep notebook 的 1000 HVG 与已存 notebook 203 维模型不一致，采用 200 HVG 并集其 7 个标记基因（本次并集为 **206** 个，不能声称与原 203 个基因完全相同），HVG 仅拟合训练细胞。双扰动使用论文公式 `z_ctrl + (mean(z_PTPN12)-mean(z_ctrl)) + (mean(z_ZBTB25)-mean(z_ctrl))`，修正 notebook 的向量形状与重复加控制变量问题。iPSC 每天无放回抽 2000 个细胞（1200/200/600），基因发现遵循作者独立的四天 discovery 分区；day1/2 的 discovery 细胞仅用于 HVG，完全不进入模型训练和评估。day0/3 用于训练，day1/2 为预测；day0/3 测试集编码重建会明确标注。采用作者代码 beta=0.0001→0.02（预印本文字与代码不一致）。
- 2026-11-13 13:04：已下载正式版补充材料，Table S1 确认 batch=64、隐藏层2048、3层、1000步扩散、lr=1e-4、100000次迭代。采用 BF16 自动混合精度及 fused AdamW 适配 A100；Norman 实测约120步/秒，先运行20000步并实际恢复到100000步，iPSC 随后训练100000步。最终使用1000步DDIM，Norman重复采样10次，iPSC重复6次；这些是固定模型的采样重复，不能冒充多次独立训练。作者原 notebook 参考值：Norman Pearson≈0.9914、R²≈0.9826；iPSC day1/2 Pearson≈0.8645/0.9125，原 notebook 反向R²≈0.6545/0.7419。由于原始划分和基因表未公开，比较为接近协议的复现。
- 2026-11-13 13:18：Norman 完成100000步与10次采样，Pearson=0.99317、R²=0.98626、增量R²=0.94377；达到公开 notebook 的表达预测量级。线性表达相加基线 R²=0.98583，与模型接近，不能据此宣称显著的非加性优势。逐基因检查发现多数反向变化出现在绝对真实增量小于0.01的弱效应基因；首轮采样中21个绝对增量≥0.1的基因方向全部正确，但弱效应方向预测仍是局限。保留未筛选的全部指标，不隐藏这一结果。
- 2026-11-13 13:24：iPSC 验证去噪损失上升，诊断显示其不能可靠选择生成模型：4000步模型虽然该损失最小，端点生成R²却为负；约30000步模型的端点重建R²已约0.93。新增仅使用 day0/day3 验证的选择：阶段模型先用100步DDIM比较表达均值MSE，再以1000步确认前两名；随后以端点逐基因Wasserstein选择潜变量噪声（0、0.05、0.1、0.2、0.3、0.5、0.7）。固定0.7在早期模型上会严重放大方差。模型训练仍完成100000步，day1/day2尚未读取，不据测试集调参。诊断保存在 `results/ipsc/validation_diagnostic.json`，最终选择记录在 `generation_selection.json`。

- 2026-11-13 13:02:57：norman 数据准备完成：{"train": {"shape": [7102, 206], "groups": {"0": 6618, "2": 309, "1": 175}}, "val": {"shape": [788, 206], "groups": {"0": 735, "2": 34, "1": 19}}, "test": {"shape": [257, 206], "groups": {"3": 257}}}

- 2026-11-13 13:03:10：开始训练 norman/v1：7102 个细胞、206 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，bf16，计划 20000 步。

- 2026-11-13 13:04:02：ipsc 数据准备完成：{"train": {"shape": [2400, 203], "groups": {"0": 1200, "3": 1200}}, "val": {"shape": [400, 203], "groups": {"0": 200, "3": 200}}, "test": {"shape": [2400, 203], "groups": {"0": 600, "1": 600, "2": 600, "3": 600}}}

- 2026-11-13 13:06:15：norman/v1 trained，已训练 20000 步，验证集去噪 MSE=0.040726。

- 2026-11-13 13:06:21：norman/v1 从第 20000 步完整恢复。

- 2026-11-13 13:17:33：norman/v1 trained，已训练 100000 步，验证集去噪 MSE=0.010802。

- 2026-11-13 13:18:03：norman/v1 评估完成（DDIM 1000 步，10 次采样）：{"PTPN12+ZBTB25": {"pearson": 0.9931650671584006, "r2": 0.986260795777144, "notebook_reverse_r2": 0.9862746464969312, "delta_pearson": 0.9724452990150342, "delta_r2": 0.9437721851688157}}

- 2026-11-13 13:18:07：开始训练 ipsc/v1：2400 个细胞、203 个基因；作者原始 2048 隐层网络，batch=64，lr=0.0001，bf16，计划 100000 步。

- 2026-11-13 13:31:45：ipsc/v1 trained，已训练 100000 步，验证集去噪 MSE=0.479640。

- 2026-11-13 13:32:31：iPSC 仅用 day0/day3 验证生成结果选择第 100000 步检查点，潜变量噪声 sigma=0.5（原值0.7），端点平均Wasserstein=0.5052。原去噪损失选择与所有候选记录保留。

- 2026-11-13 13:33:34：ipsc/v1 评估完成（DDIM 1000 步，6 次采样）：{"day0": {"pearson": 0.9989788410545618, "r2": 0.9971314168141362, "notebook_reverse_r2": 0.9970874488235446, "delta_pearson": 0.3836710273372661, "delta_r2": -0.9030673759910831}, "day1": {"pearson": 0.9131541020104175, "r2": 0.8275476949311913, "notebook_reverse_r2": 0.7953671544278643, "delta_pearson": 0.4696385173137989, "delta_r2": 0.180690056081931}, "day2": {"pearson": 0.9388926622506238, "r2": 0.8626110752291433, "notebook_reverse_r2": 0.8475495548349876, "delta_pearson": 0.8525394541611048, "delta_r2": 0.6661980773424321}, "day3": {"pearson": 0.9980534049562082, "r2": 0.9948209349136485, "notebook_reverse_r2": 0.9944288069996561, "delta_pearson": 0.9968872038220017, "delta_r2": 0.9905230520201935}}
