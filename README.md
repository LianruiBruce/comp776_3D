# Identity Layers Lab

这是一个研究“多张参考自拍如何保留人物身份”的可复现计算机视觉项目。第一阶段先分析人脸识别 ViT：不同 Transformer 层在表情等干扰发生变化时，是否仍然保留可区分的身份信息；后续再把经过验证的多层、多参考表示接入图像生成模型。

当前可运行基线使用 FEI 手工对齐正脸数据和 LVFace-T Glint360K。代码已经在 NVIDIA RTX 4080、Python 3.11、PyTorch 2.8.0+cu128 上验证。

## 当前实验状态

| 项目 | 当前设置 |
|---|---|
| 数据 | FEI aligned，200 个身份，每人 neutral/smile 两张 |
| 划分 | 按身份划分：120 dev / 40 val / 40 test，seed 776 |
| 模型 | LVFace-T Glint360K，12 blocks，约 19.1M 参数 |
| 分析表示 | normalized token mean；official final head applied per block |
| 选层规则 | dev 上最大化 d-prime |
| 阈值规则 | val 异人对校准，test 只评估冻结层和冻结阈值 |
| 运行性能 | RTX 4080 上 400 张图像提取 12 层约 0.77 秒，峰值显存约 353 MiB |

首轮实验的两种表示均选择第 12 层，因此 FEI neutral/smile 这组简单扰动暂不支持“中间层优于最终层”的假设。完整结论和限制见 [reports/SMOKE_FEI_LVFACE_T.md](reports/SMOKE_FEI_LVFACE_T.md)。

## 仓库结构与 Git 规则

| 路径 | 是否提交 GitHub | 内容 |
|---|---|---|
| `src/id_layers/` | 是 | 可复用实验代码 |
| `scripts/` | 是 | 资产下载和校验脚本 |
| `configs/` | 是 | 实验配置、URL、revision 和 SHA256 资产锁 |
| `requirements/`、`pyproject.toml` | 是 | Python/CUDA 依赖和包配置 |
| `tests/` | 是 | 单元测试 |
| `data/DATASET_CARD.md` | 是 | 数据来源、许可和限制 |
| `data/manifests/` | 是 | 不含图像的样本 provenance 与身份划分 |
| `models/MODEL_CARD.md` | 是 | 模型结构、权重来源和使用限制 |
| `docs/`、`reports/` | 是 | 实验协议和聚合结果 |
| `AGENTS.md` | 是 | 组内代理/研究状态的唯一同步文件 |
| `artifacts/README.md`、`third_party/README.md` | 是 | 本地目录说明 |
| `data/raw/`、`data/processed/` | **否** | 受限原始人脸和处理后图像 |
| `artifacts/cache/` | **否** | 模型权重与下载缓存 |
| `artifacts/runs/`、`artifacts/tmp/` | **否** | 可重新生成的运行产物和临时预览 |
| `third_party/LVFace/` | **否** | bootstrap 创建的 pinned 第三方 checkout |
| `.venv/`、Python cache、IDE 文件 | **否** | 机器相关文件 |

请勿上传 FEI 图片、LVFace 权重、自拍、密钥或任何可识别个人身份的人脸样本。`.gitignore` 已覆盖这些路径，但提交前仍要人工检查 staged files。

## 1. 克隆项目

Windows PowerShell：

```powershell
git clone https://github.com/LianruiBruce/comp776_3D.git
Set-Location comp776_3D
```

Linux shell：

```bash
git clone https://github.com/LianruiBruce/comp776_3D.git
cd comp776_3D
```

## 2. 准备运行环境

当前 smoke 配置要求：

- 64-bit Python 3.11；
- Git；
- NVIDIA GPU 和能够运行 CUDA 12.8 PyTorch wheel 的驱动；
- 可访问 FEI、GitHub 和 Hugging Face 的网络；
- 足够空间保存本地虚拟环境、约 400 张 FEI 图片和 LVFace 权重。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Linux shell：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements/torch-cu128.txt
.venv/bin/python -m pip install -r requirements/base.txt
.venv/bin/python -m pip install -e . --no-deps
```

不需要激活虚拟环境；下面所有命令都直接调用仓库内的 Python，从而减少使用错误环境的概率。

## 3. 检查 CUDA

Windows：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Linux：

```bash
.venv/bin/python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

继续运行前应看到 `cuda: True`。当前配置明确使用 `device: cuda`，尚不保证 CPU 路径可用。

## 4. 下载并校验研究资产

先阅读 [数据卡](data/DATASET_CARD.md) 和 [模型卡](models/MODEL_CARD.md)。FEI 数据与 LVFace 预训练权重只能按发布者条款用于本地研究，不得重新分发。

Windows：

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap_assets.py --accept-research-terms
```

Linux：

```bash
.venv/bin/python scripts/bootstrap_assets.py --accept-research-terms
```

该命令会：

1. 从官方地址下载并校验 FEI archives；
2. checkout 固定 revision 的 LVFace，且要求工作树 clean；
3. 下载固定 revision 的 LVFace-T 权重；
4. 对所有文件验证 `configs/assets.lock.yaml` 中的 SHA256。

再次运行同一命令不会重复下载，主要用于完整性检查。

## 5. 运行 smoke experiment

Windows：

```powershell
.\.venv\Scripts\python.exe -m id_layers.cli --config configs\smoke_fei_lvface_t.yaml
```

Linux：

```bash
.venv/bin/python -m id_layers.cli --config configs/smoke_fei_lvface_t.yaml
```

成功后终端会输出新的目录：

```text
artifacts/runs/<UTC时间>_smoke_fei_aligned_lvface_t/
```

重点检查：

- `status.json`：必须为 `complete`；
- `summary.json`：数据量、模型、性能和最终指标；
- `metrics_by_layer.csv`：dev/val 逐层指标以及冻结后的 test 层；
- `selection.json`：dev 选出的层；
- `frozen_operating_points.json`：val 校准阈值后的 test FAR/TAR；
- `plots/layer_curves_dev.png`：dev 层曲线；
- `environment.json`、`environment.lock.txt`：硬件和完整 Python 环境；
- `source_snapshot/`：该次运行使用的精确可执行源码。

`artifacts/runs/` 不提交 GitHub。需要共享结论时，把不含人脸和权重的聚合结果整理进 `reports/`。

## 6. 测试与代码检查

Windows：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe -m pip check
```

Linux：

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pip check
```

当前基线应通过 4 个单元测试、Ruff 检查和依赖一致性检查。第三方 LVFace 的 `timm`/AMP deprecation warnings 来自 pinned upstream 代码，不代表实验失败；不要直接修改 `third_party/LVFace/` 来消除它们。

## 7. 提交到 GitHub 前检查

确认 `.gitignore` 生效：

```powershell
git status --short
git status --short --ignored
git check-ignore -v data/raw/fei/frontalimages_manuallyaligned_part1.zip
git check-ignore -v artifacts/cache/models/LVFace/LVFace-T_Glint360K/LVFace-T_Glint360K.pt
git check-ignore -v artifacts/runs
git check-ignore -v third_party/LVFace
```

第一次整理提交时，可以在确认状态后执行：

```powershell
git add .
git diff --cached --stat
git diff --cached --check
git status --short
```

在 `git commit` 或 `git push` 前，staged files 中应当只出现前面表格标为“是”的内容。如果出现人脸图片、ZIP、模型权重、`.venv`、`third_party/LVFace` 或 `artifacts/runs`，立即停止提交并检查路径和 `.gitignore`。

## 常见问题

### `torch.cuda.is_available()` 为 `False`

确认机器有 NVIDIA GPU、驱动可见，并且先安装了 `requirements/torch-cu128.txt`。不要只执行 `pip install torch`，否则可能安装不符合项目配置的版本。

### bootstrap 报 checksum mismatch

脚本会拒绝覆盖校验失败的文件。检查下载是否被代理或网络中断；确认目标确实是损坏的本地缓存后再删除该单个文件并重新运行 bootstrap。

### bootstrap 报 LVFace checkout has local changes

`third_party/LVFace/` 被设计成只读缓存。项目适配应写在 `src/id_layers/`；如需恢复 checkout，请先保存自己的修改，再重新创建该本地目录。

### 运行结果与报告略有速度差异

吞吐会随 GPU、驱动、后台负载和温度变化。首先比较配置、资产 SHA256、逐层指标和 embedding fingerprints，而不是要求耗时完全相同。

## 研究边界与下一步

当前 FEI aligned 只有近正脸 neutral/smile 条件，适合验证工程链路，不足以证明姿态、光照或真实生成场景中的身份保持。下一步计划是加入独立 ArcFace evaluator、五点对齐、FEI full/Extended Yale B、CVLFace ViT-B 和多参考聚合，之后再进入生成模型实验。

实验假设、数据泄漏规则和生成阶段评估要求见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。协作状态以 [AGENTS.md](AGENTS.md) 为准。
