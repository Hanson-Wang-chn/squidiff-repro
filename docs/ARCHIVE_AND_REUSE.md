# GitHub 归档、恢复与复现

本仓库保存 2026-11-13 至 2026-11-15 的三轮 Squidiff 本地复现与科学审计。2026-11-15 将 day1 专项诊断、第二轮和第三轮实验纳入 Git 归档，提交身份为 `whs <hs.wang.chn@outlook.com>`，默认分支为 `main`。科学结论及报告索引见 [README](../README.md)。

## 归档保留什么

| 路径 | 保留的内容 | 用途 |
|---|---|---|
| `scripts/`、`configs/` | 训练、预处理、推理、校准、对照、绘图和审计代码与实验计划 | 重建流程或扩展实验 |
| `upstream/` | 本轮使用的作者源码、原始 notebook、PyPI 代码快照与许可证 | 保留实现来源；普通文件随仓库下载，无须子模块初始化 |
| `results/` | 全部现有 CSV、JSON、JSONL、图和预测 NPZ，包括 v1、失败与作废实验的说明 | 阅读结论、重画图、继续分析预测 |
| `logs/` | 历史训练/评估/下载日志、事件记录、完整工作纪要 | 追溯参数、错误及修改过程 |
| `docs/` | 来源核查、论文资料、上游哈希与完成记录 | 核对证据和公开协议边界 |
| `data/**/manifest.json` | 下载 URL/MD5、基因名单、分区大小、种子与数据变体说明 | 下载原数据并重建分区 |
| 环境文件 | 核心依赖、完整快照、便于迁移的环境入口 | 在新机器安装 |

**不归档数据矩阵、缓存及任何 checkpoint。** 包括原始 CSV/TSV/H5AD/压缩数据、处理后 H5AD、作者权重、本地模型/EMA/优化器/RNG 状态。预测 NPZ 属于有价值的实验输出，全部保留；其中可能含用于比较的测试表达和细胞/基因标识，不等同于完整原始数据集。

完整工作区文件约 **310 MiB**。为完整保留新增两轮的预测、图和审计证据，归档预算从 300 MiB 调整为 **350 MiB**，单文件仍限制 **25 MiB**；实际字节数、逐文件 SHA256 和有意排除的数据/权重清单见 [archive_manifest.json](archive_manifest.json)。Git 会共享内容完全相同的文件对象，实际传输大小由 Git 打包决定。清单本身不对自身计算 SHA256，由 Git 提交保护。`.gitignore` 不删除服务器上的原文件；本次也没有清理现存数据和模型。

`status.json`、`complete.json`、`assessment.json`、`final_integrity.json` 和日志中的 PID、可用空间、checkpoint 存在性等，都是实验完成时的历史记录，不能当作新机器的实时状态。**克隆后可以直接查看结果，但不能从旧训练步数恢复或直接调用模型推理。** 原权重删除后，需要重新训练才能做新的模型推理。

## 首次上传 GitHub

当前 `origin` 已配置为 `https://github.com/Hanson-Wang-chn/squidiff-repro.git`。后续更新在完成校验和提交后运行 `git push origin main` 即可；以下创建空仓库的步骤仅供迁移到其他仓库时使用。

1. 在 GitHub 网页创建一个空仓库。名称可用 `squidiff-reproduction`，公开或私有由你选择。创建时不要添加 README、`.gitignore` 或 License，本地已有首个提交。
2. 在当前服务器执行下列命令，把 URL 中的账号和仓库名替换成网页上的实际值。Git 提交名 `whs` 不一定是你的 GitHub 账号名。

```bash
cd /home/whs/squidiff
source /home/whs/miniforge3/etc/profile.d/conda.sh
conda activate squidiff
python scripts/verify_archive.py
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

如果已添加过 `origin`，用 `git remote set-url origin URL` 修改。HTTPS 提示 Username 时填 GitHub 账号，Password 时填 **Personal Access Token（PAT）**；GitHub 登录密码不能用于 Git 推送。可在 GitHub Settings → Developer settings → Personal access tokens 创建只授权目标仓库、具有 Contents 读写权限的 fine-grained token。也可使用你已配置好的 SSH URL。Token 只在认证提示中输入，不要写到 remote URL 或仓库文件里。

建议使用以上 Git 推送，一次上传整个已提交归档。网页拖放需要自行维持目录结构和筛选文件；不要把服务器上的整个数十 GB 目录拖进去，也不要上传 `.git` 目录或 `.bundle` 文件来代替仓库内容。

官方说明：[上传本地仓库](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[PAT 认证](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)、[文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 删除服务器目录前验证远端

推送成功后，在另一个目录重新克隆并校验。这一步验证 GitHub 上的文件足以独立恢复归档：

```bash
cd /home/whs
git clone https://github.com/Hanson-Wang-chn/squidiff-repro.git squidiff-restore-check
cd squidiff-restore-check
python scripts/verify_archive.py
git log -1 --format='%h %an <%ae> %s'
```

看到 `PASS`，并在 GitHub 网页确认 README、`results/research/`、预测 NPZ 和文档均存在后，再自行决定是否删除原目录。原目录中被忽略的数据和 checkpoint 不会被这次上传备份，之后的数据可按清单重下，模型需重训。

如需额外的离线备份，可自行在原目录外生成包含全部提交的 bundle。它不含数据和 checkpoint；每次新增提交后需重新生成，旧 bundle 不会自动更新：

```bash
git bundle create /home/whs/squidiff-github.bundle --all
git bundle verify /home/whs/squidiff-github.bundle
```

将该文件下载到自己的电脑后，可以这样恢复：

```bash
git clone /path/to/squidiff-github.bundle squidiff
cd squidiff
python scripts/verify_archive.py
# 从 bundle 克隆时，origin 指向 bundle 文件；首次推送前修改它。
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## 只查看和分析已有结果

阅读 Markdown、CSV、JSON 和图不需要 GPU，也不需要重新下载数据。校验工具只用 Python 标准库：

```bash
python scripts/verify_archive.py
```

建议先看：

- `results/research/scientific_summary.png`：主要科学对照。
- `results/research/assessment.json`：完整实验清单与限制。
- `results/research/source_all_control_seed_summary.csv`：Norman 全控制细胞、三个训练种子的摘要。
- `results/research/all_controlled_metrics.csv`：全部受控指标，含诊断与被排除模型，解读时保留标签。
- `results/research/sampling_null_summary.csv`、`transient_programs/`、`conditional_uncertainty/`：负对照、瞬态程序、条件抽样区间。
- `results/{norman,ipsc}/{run}/research_eval/{tag}/`：逐基因表、指标与预测 NPZ；`results/rejected/` 记录作废实验，不能作有效预测成绩。
- `logs/README_research_full_20261113.txt`、`logs/events.jsonl`：完整过程。
- [day1 诊断](../report.md)、[第二轮](../report-2.md)、[第三轮](../report-3.md)：后续报告；对应的 `results/diagnosis_20261115/`、`results/round2/`、`results/round3/` 保留指标、图、逐基因表和预测 NPZ。
- `docs/round2/`、`docs/round3/`、`logs/round2/`、`logs/round3/`：后续实验的来源、环境与历史训练/评估过程。

使用 NumPy 检查预测文件时，应保留归档里的基因顺序和评估标签：

```python
from pathlib import Path
import numpy as np

path = next(Path('results/norman/source_fp32_s42/research_eval').rglob('*.npz'))
with np.load(path, allow_pickle=False) as result:
    print(path, result.files)
    for name in result.files:
        print(name, result[name].shape, result[name].dtype)
```

## 新机器安装

训练入口面向 Linux 和 NVIDIA CUDA GPU，原实验使用 A100 40GB。仅查看归档不需安装下面的完整环境。训练/数据预处理需要重新准备数十 GB 磁盘和足够内存；调度器要求剩余空间至少 20 GiB。

先安装 conda/Miniforge，然后在仓库根目录执行：

```bash
bash scripts/setup_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate squidiff
```

脚本通过 `environment-portable.yml` 创建 Python 3.10 + pip + Git 环境，安装 PyTorch 2.5.1/CUDA 12.1、`requirements.txt` 的核心依赖和仓库内的作者包，并检查 CUDA。它使用现有 `squidiff` 环境时只补 Git 和所需 Python 包。

`environment.yml` 是原 Linux 环境的完整构建快照，保留用于追溯，包含具体 conda build 和平台依赖；迁移优先用上述入口。若还需尽量对齐全部 Python 间接依赖，可在安装后运行：

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-lock.txt
python -m pip check
```

锁文件中的旧机器 `file:///.../packaging` 已替换成实际版本 `packaging==26.3`，作者包仍从 `./upstream/Squidiff` 安装。不同 GPU、驱动或软件构建不保证训练逐位一致；原始作者权重、203 基因名单和细胞划分种子缺失的限制也不会因重装环境而消失。

## 重新下载、训练与评估

**在单独的新克隆中重跑。** 先执行归档校验，再把旧结果和日志移到被忽略的 `archive/`，避免调度器把历史完成标记当作新训练已经结束。以下移动命令仅用于尚未重跑过的新克隆：

```bash
python scripts/verify_archive.py
mkdir -p archive
mv results archive/results
mv logs archive/logs
mkdir -p results/research logs

python scripts/download_data.py
python scripts/prepare.py norman
python scripts/prepare.py ipsc
python scripts/prepare_research.py
```

下载脚本支持断点下载及 MD5 校验，自动解压 Norman；iPSC CSV 会被转换成缓存和各实验分区。下载链接与摘要也保留在 `data/raw/manifest.json`。预处理会重写分区 manifest，并通过 `common.record()` 向 README 和事件日志追加记录，所以请保留独立的原始归档。

最小的源协议 Norman 训练与全控制细胞评估示例：

```bash
python scripts/train.py norman --run source_fp32_s42 \
  --data-root data/research/source_full --seed 42 --steps 100000 \
  --precision fp32 --math-mode ieee --sampler native --optimizer source \
  --lr-timing source --validation-source training --save-every 5000 --val-every 10000

python scripts/research_evaluate.py norman --run source_fp32_s42 \
  --data-root data/research/source_full --tag all_control_cells \
  --modes cell_shift --cells 7353 --inference-math ieee
```

需要重跑本轮全部 15 个研究模型时，可把原五个阶段合并为串行计划，并显式将最终步数统一到 100000。原阶段一阳性对照保留了早期 20000 步配置，后续诊断要求 100000 步；下面生成的计划解决这个历史差异，不改动原实验计划。

```bash
python - <<'PY'
import json
from pathlib import Path
jobs = []
for stage in range(1, 6):
    jobs.extend(json.loads(Path(f'configs/research_stage{stage}.json').read_text()))
for job in jobs:
    job['steps'] = 100000
Path('archive/replay_plan.json').write_text(json.dumps(jobs, indent=2))
PY
python scripts/research_pipeline.py --plan archive/replay_plan.json
python scripts/research_followup.py
python scripts/final_integrity.py
```

这重跑的是第一轮 15 个研究模型及后续流程；历史 v1、硬件基准、额外探索诊断不会因此全部重建。需重建 v1 时另外运行 `python scripts/pipeline.py`。`research_parallel.py` 是原机器的 CUDA MPS 三队列入口，要求系统有 `nvidia-cuda-mps-control`；通用复现先用上述串行流程。

第二、三轮的参数和命令分别见 [report-2.md](../report-2.md) 和 [report-3.md](../report-3.md)。必须先重建第一轮 iPSC 分区，再运行 `python scripts/round2.py prepare`。在新克隆中，应先按前述步骤隔离全部旧结果和日志；报告中的绝对环境路径应替换为新机器的 `conda activate squidiff`。第三轮作者权重诊断还依赖重新下载的作者数据和 checkpoint，不能仅凭归档直接推理。

重训中断后，同一 run 有 `latest.pt` 时可恢复；**只存在旧 `status.json` 而没有权重时不能续训**，所以必须先隔离旧结果。`final_integrity.py` 检查本地数据、权重与预测的一致性，应在重训完成后使用；归档下载后的完整性应使用 `verify_archive.py`。历史 `results/research/final_integrity.json` 保留本轮原机器检查结果。

## 后续维护归档

新增代码、结果或文档后，在仓库根目录更新清单、校验并提交：

```bash
git add .
python scripts/verify_archive.py --write
git add docs/archive_manifest.json
python scripts/verify_archive.py
git diff --cached --stat
git commit -m "Update code and archived results"
git push
```

`--write` 从暂存区取文件名、从工作区读取内容；请先完成修改，再按上述顺序执行，避免生成清单后又改文件。请勿用 `git add -f` 把数据或模型加回；清单校验会拒绝数据、权重和超过归档预算的文件。

当前目录已配置提交身份。Git 的仓库配置不会随克隆传递；换机器后首次提交前可重新设置：

```bash
git config --local user.name whs
git config --local user.email hs.wang.chn@outlook.com
```

作者代码的 BSD 3-Clause 许可证原样保留在 `upstream/Squidiff/LICENSE` 和 PyPI 快照中。原源码的 LICENSE 含上游遗留的合并冲突标记，未在归档时改写。上游快照原有的 `.ipynb_checkpoints/` 是 notebook 的源码备份，连同其发行包保留，以匹配 `docs/upstream_source_hashes.json`；它们不是模型 checkpoint。其他第三方文档/notebook 保留其来源与现有声明；本次未替整个混合归档新增统一开源许可证。
