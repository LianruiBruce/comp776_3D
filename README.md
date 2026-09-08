# Identity Layers Lab

这是一个研究“怎样让参考自拍驱动的生成照片在人类看来更像本人”的可复现计算机视觉项目。人脸识别 ViT 的逐层分析用于诊断身份信息是否可访问；生成阶段进一步区分参考采样、身份编码、条件注入和扩散渲染四类瓶颈，并以盲法人评作为感知 likeness 的主要终点。

当前已经跑通四条链路：FEI 手工对齐子集 smoke、FEI 原始 2,800 张图像的 SCRFD 五点对齐与 LVFace-T 全层 E1、五随机种子的 E2 等容量监督度量读出，以及 PhotoMaker V2 多参考身份生成诊断。代码已在 NVIDIA RTX 4080、Python 3.11、PyTorch 2.8.0+cu128 上验证。

## 当前实验状态

| 项目 | 当前设置 |
|---|---|
| 数据 | FEI original，200 个身份 × 14 条件；2,782/2,800 对齐成功 |
| 划分 | 按身份划分：100 train / 50 dev / 50 internal eval，seed 20260902 |
| 模型 | LVFace-T Glint360K，12 blocks，约 19.1M 参数 |
| 分析表示 | E1 diagnostics；E2 fresh-LN + token mean + 256→512 angular readout |
| E2 训练 | 冻结 backbone，5 seeds × 12 layers × 120 epochs，CosFace |
| 选层规则 | dev identity-wise cross-fitted template TAR@FAR=.01；固定 tie-break |
| 阈值规则 | dev 异人模板分数校准，internal eval 只使用冻结层和阈值 |
| E2 结果 | dev 选择 block 11；相对 matched block 12 的内部 end-to-end TAR 差 +0.00067，95% CI [0, .002]，H2 gate 未通过 |
| E2 性能 | 探针与对照训练约 540 秒；探针阶段峰值 CUDA allocation 约 69.7 MiB |
| 生成诊断 | PhotoMaker V2 + RealVisXL V4；8 个 FEI internal-eval 身份 × 6 条件 × 2 prompts × 2 seeds，共 192 图 |
| 生成状态 | 192/192 成功且均检测到单张人脸；独立 LVFace 自动评估与 v2 人评均完成：10 位评分者、96 个配对、480 次判断；P0/P1 无明确优势，P3 未决 |

E1 中 raw token mean 的 block 11 优于 block 12；但 E2 给两层相同训练读出后，内部差异几乎消失且不满足预注册判据。因此当前证据不支持“最终层丢失身份”或存在一个确定的“身份层”。完整结果见 [E2 报告](reports/E2_FEI_FULL_LVFACE_PROBES.md)、[E1 报告](reports/E1_FEI_FULL_LVFACE_T.md)；早期 smoke 见 [reports/SMOKE_FEI_LVFACE_T.md](reports/SMOKE_FEI_LVFACE_T.md)。

正式生成运行位于 `artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot`。自动指标显示原生身份条件明显不同于 text-only，且冲突实验中输出主要跟随全局 InsightFace 身份向量；四张多样参考相对单张或重复参考只有小幅自动相似度增益。v2 人评已完成，P0/P1 不支持明确的多参考感知收益，P3 区间很宽且无法判断比例高。该实验只有 8 个已用于开发的 FEI 身份，不能作为外部确认，也没有直接测量生成图相对真实照片的人类 likeness 缺口。详见 [生成诊断报告](E:/comp776_3D/reports/GENERATION_IDENTITY_PILOT.md)。

第一次接触 CV 可先读 [中文研究计划：从真实自拍到生成肖像](E:/comp776_3D/docs/RESEARCH_PLAN_ZH.md)，其中说明已有证据、基础术语、最小实验、统计、预算和论文完成标准。当前优先级是建立新身份的真实照片参照与本人/陌生人分开的评分流程，先测量 likeness 缺口，再选择有证据的干预。旧 [身份保真实验路线](E:/comp776_3D/docs/IDENTITY_FIDELITY_EXPERIMENT_ROADMAP.md) 保留为历史文献审计；已完成的人评见 [v2 协议与结果](E:/comp776_3D/docs/HUMAN_IDENTITY_LIKENESS_V2.md)。

## 仓库结构与 Git 规则

| 路径 | 是否提交 GitHub | 内容 |
|---|---|---|
| `src/id_layers/` | 是 | 可复用实验代码 |
| `scripts/` | 是 | 资产 bootstrap、实验运行、评估与盲评入口 |
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
| `third_party/LVFace/`、`third_party/InsightFace/`、`third_party/PhotoMaker/` | **否** | bootstrap 创建的 pinned 第三方 checkout |
| `.venv/`、Python cache、IDE 文件 | **否** | 机器相关文件 |

请勿上传 FEI 图片、生成图片、盲评材料/响应/答案、模型权重、自拍、密钥或任何可识别个人身份的人脸样本。`.gitignore` 已覆盖这些路径，但提交前仍要人工检查 staged files。

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

当前环境要求：

- 64-bit Python 3.11；
- Git；
- NVIDIA GPU 和能够运行 CUDA 12.8 PyTorch wheel 的驱动；
- 可访问 FEI、GitHub 和 Hugging Face 的网络；
- 若运行 E1，至少预留约 2 GB 保存 FEI 原始/对齐图像、检测模型和运行产物。
- 若运行生成实验，另需约 9 GB 保存 PhotoMaker V2 adapter、RealVisXL V4 和本地 checkout，并为生成图片与评估产物预留空间。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\face-preprocess.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\generation.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Linux shell：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements/torch-cu128.txt
.venv/bin/python -m pip install -r requirements/base.txt
.venv/bin/python -m pip install -r requirements/face-preprocess.txt
.venv/bin/python -m pip install -r requirements/generation.txt
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

先阅读 [数据卡](data/DATASET_CARD.md) 和 [模型卡](models/MODEL_CARD.md)。FEI 数据、LVFace/InsightFace 预训练权重以及 PhotoMaker V2 生成栈必须分别遵守各自上游条款；本项目按本地、非商业研究资产管理，不得从仓库重新分发。

Windows：

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap_assets.py --accept-research-terms
.\.venv\Scripts\python.exe scripts\bootstrap_assets.py --accept-research-terms --profile e1
.\.venv\Scripts\python.exe scripts\bootstrap_assets.py --accept-research-terms --profile generation
```

Linux：

```bash
.venv/bin/python scripts/bootstrap_assets.py --accept-research-terms
.venv/bin/python scripts/bootstrap_assets.py --accept-research-terms --profile e1
.venv/bin/python scripts/bootstrap_assets.py --accept-research-terms --profile generation
```

第一条命令只准备 aligned smoke；`--profile e1` 还会准备 FEI 全集、固定 commit 的 InsightFace checkout 和 SCRFD-10G；`--profile generation` 在这些资产之外还会准备 InsightFace `w600k_r50.onnx`、固定 revision 的 PhotoMaker V2 checkout/adapter 与 RealVisXL V4 fp16 snapshot。脚本会：

1. 从官方地址下载并校验对应的 FEI archives；
2. checkout 固定 revision 的 LVFace/InsightFace，且要求工作树 clean；
3. 下载并校验 LVFace-T 权重、检测/识别 ONNX 模型及所选 profile 的生成模型；
4. 对所有文件验证 `configs/assets.lock.yaml` 中的 SHA256。

再次运行同一命令不会重复下载，主要用于完整性检查。

## 5. 运行实验

### 5.1 最小 aligned smoke

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

### 5.2 FEI-full E1

先跑 12 个身份的端到端 gate：

```powershell
.\.venv\Scripts\python.exe -m id_layers.cli --config configs\smoke_fei_full_lvface_t.yaml
```

gate 完成后再跑正式 E1：

```powershell
.\.venv\Scripts\python.exe -m id_layers.cli --config configs\fei_full_lvface_layers.yaml
```

Linux 将 Python 路径替换为 `.venv/bin/python`，配置路径使用 `/`。E1 除通用运行文件外还会生成：

- `preprocess_coverage.csv`：所有成功和失败样本，失败不会被静默删除；
- `metrics_by_condition.csv`：按固定 query 条件拆分的指标与冻结阈值结果；
- `bootstrap_intervals.csv`：1,000 次身份级配对 bootstrap；
- `embeddings.npz`：本地逐层表示，仅留在 Git 忽略的 run 目录。

### 5.3 FEI-full E2 等容量读出

先运行快速工程 gate：

```powershell
.\.venv\Scripts\python.exe -m id_layers.cli --config configs\smoke_fei_full_lvface_probes.yaml
```

再运行正式五随机种子实验：

```powershell
.\.venv\Scripts\python.exe -m id_layers.cli --config configs\fei_full_lvface_probes.yaml
```

Linux 将 Python 路径替换为 `.venv/bin/python`，配置路径使用 `/`。在已准备好资产的 RTX 4080 上，正式 probe 与 controls 约运行九分钟。重点检查：

- `selection.json`：五 seed 汇总后的唯一 dev-selected layer；
- `metrics_by_layer_seed.csv`、`metrics_by_layer.csv`：逐 seed 与聚合 dev 证据；
- `crossfit_operating_points.csv`：每个身份 fold 的独立阈值校准；
- `internal_frozen_metrics.csv`：只引用 `frozen_*` 字段作为正式 internal 指标；
- `selected_vs_final_bootstrap.json`：失败计拒识的 matched-readout 配对比较；
- `control_metrics.csv`、`training_history.csv`：Gaussian、标签打乱、未训练投影和收敛检查；
- `access_audit.json`：确认 internal features 在选层落盘后才被提取；
- `probe_repeat.json`：重复训练的参数与 embedding 是否完全一致。

E2 的公平 H2 比较是 selected matched readout 对 matched block-12 readout。官方 LVFace 最终 embedding 是结构和训练过程不同的 operational baseline，不能混作“等容量”对照。

### 5.4 PhotoMaker V2 生成身份诊断

实验配置为 [configs/generation_identity_pilot.yaml](configs/generation_identity_pilot.yaml)。它固定 8 个 FEI `internal_eval` 身份、6 个身份条件、2 个 prompts 和 2 个 seeds；完整运行应生成 `8 × 6 × 2 × 2 = 192` 张 1024×1024 图片。这里研究的是“哪条身份信息通道影响生成结果”，不是继续选择 ViT 层，也不是外部测试。

先用一个任务检查 4080、模型路径和端到端生成：

```powershell
.\.venv\Scripts\python.exe scripts\run_generation_pilot.py --condition k1_full --max-tasks 1
```

smoke 通过后再运行冻结的完整矩阵：

```powershell
.\.venv\Scripts\python.exe scripts\run_generation_pilot.py
```

单图 smoke 只验证工程链路，不满足完整矩阵校验器的输入要求，不要对它运行正式 `evaluate_generation_pilot.py`。

在本项目 RTX 4080 上，正式运行 192/192 完成，总生成时间约 29.8 分钟，平均约 9.32 秒/图；峰值 CUDA allocated/reserved 分别约 6.40/13.53 GiB。运行目录由脚本打印并写入 `artifacts/runs/`。不要把 smoke 与正式矩阵写到同一个 run ID；只有显式指定同一 `--run-id` 后才能用 `--resume` 校验并续跑。

把下面的 `RUN_ID` 替换为刚完成的完整运行目录名，然后执行独立自动评估和身份级 bootstrap 汇总：

```powershell
$run = "artifacts\runs\RUN_ID"
.\.venv\Scripts\python.exe scripts\evaluate_generation_pilot.py --run-dir $run --evaluation-name evaluation
.\.venv\Scripts\python.exe scripts\summarize_generation_pilot.py --run-dir $run --evaluation-name evaluation --output-name analysis
```

评估必须保留检测失败并计入分母。`lvface_independent` 是未被 PhotoMaker 使用的主要自动身份评价器；PhotoMaker 自己依赖的 InsightFace 仅作为 `insightface_conditioner` 二级复用诊断。正式运行 `20260903T050000Z_photomaker_v2_generation_pilot` 的第一次评估因验证器错误保留在 `evaluation/`；清理后复评的当前入口为 `evaluation_v3/` 和 `analysis_v3/`。首次成功的 `evaluation_v2/` 与 v3 的六个核心产物 SHA256 全部一致。不要把失败目录当作结果，也不要手工覆盖已完成目录。

自动相似度不等同于“人看起来像本人”。v1 master 包保留在 `human_eval/` 作为历史工程产物；当前实验使用更精简的 v2，只比较与研究问题直接相关的 P0、P1、P3，共 96 个唯一 pair。它生成 10 份各 48 题的匿名表单，使每个 pair 恰好由 5 名不同参与者评分。详细设计见 [H1 人类感知身份相似度实验](docs/HUMAN_IDENTITY_LIKENESS_V2.md)。

```powershell
.\.venv\Scripts\python.exe scripts\build_grouped_human_eval.py `
  --run-dir $run `
  --manifest data\manifests\fei_full_seed20260902.csv `
  --output "$run\human_eval_v2" `
  --seed 20260904
```

将 `participant/` 下的 `A01–A05`、`B01–B05` 分别分配给 10 名参与者。HTML 依赖同目录下的 `media/`，移动或分享时要保持两者的相对位置；每人只完成指定的一份表单。下载的 response JSON 放入 Git 忽略的 `human_eval_v2/responses/`，然后统一分析：

当前正式包的第一份页面是 `artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot/human_eval_v2/participant/A01.html`。

```powershell
$responseFiles = (Get-ChildItem "$run\human_eval_v2\responses\*.json").FullName
.\.venv\Scripts\python.exe scripts\analyze_grouped_human_eval.py `
  --answer-key "$run\human_eval_v2\private\answer_key.json" `
  --responses $responseFiles `
  --output "$run\human_eval_v2\analysis\analysis.json"
```

页面只问身份相似度，允许 A、B、相同和无法判断；不混入 4-AFC、画质、prompt 或 text-only 题。公开页面与 `private/answer_key.json` 必须分开保存。在 Linux 上将 Python 路径换为 `.venv/bin/python`、路径分隔符换为 `/`；response 文件可用 shell glob。没有收齐真实评分前，不能报告人类偏好或“哪种条件更像本人”。

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

当前应通过全部单元测试、Ruff 检查和 `pip check`。第三方代码的 `timm`/AMP deprecation warnings 不代表实验失败；不要直接修改 pinned checkout 来消除它们。

## 7. 提交到 GitHub 前检查

确认 `.gitignore` 生效：

```powershell
git status --short
git status --short --ignored
git check-ignore -v data/raw/fei/frontalimages_manuallyaligned_part1.zip
git check-ignore -v artifacts/cache/models/LVFace/LVFace-T_Glint360K/LVFace-T_Glint360K.pt
git check-ignore -v artifacts/runs
git check-ignore -v third_party/LVFace
git check-ignore -v third_party/InsightFace
git check-ignore -v third_party/PhotoMaker
```

第一次整理提交时，可以在确认状态后执行：

```powershell
git add .
git diff --cached --stat
git diff --cached --check
git status --short
```

在 `git commit` 或 `git push` 前，staged files 中应当只出现前面表格标为“是”的内容。如果出现人脸图片、生成图片、盲评 media/response/answer key、ZIP、模型权重、`.venv`、任何 `third_party` checkout 或 `artifacts/runs`，立即停止提交并检查路径和 `.gitignore`。

## 常见问题

### `torch.cuda.is_available()` 为 `False`

确认机器有 NVIDIA GPU、驱动可见，并且先安装了 `requirements/torch-cu128.txt`。不要只执行 `pip install torch`，否则可能安装不符合项目配置的版本。

### bootstrap 报 checksum mismatch

脚本会拒绝覆盖校验失败的文件。检查下载是否被代理或网络中断；确认目标确实是损坏的本地缓存后再删除该单个文件并重新运行 bootstrap。

### bootstrap 报第三方 checkout has local changes

`third_party/LVFace/`、`third_party/InsightFace/` 和 `third_party/PhotoMaker/` 被设计成只读、固定 revision 的缓存。项目适配应写在 `src/id_layers/`；如需恢复 checkout，请先保存自己的修改，再重新创建对应本地目录。

### 运行结果与报告略有速度差异

吞吐会随 GPU、驱动、后台负载和温度变化。首先比较配置、资产 SHA256、逐层指标和 embedding fingerprints，而不是要求耗时完全相同。

### SCRFD 静默回退到 CPU 或缺少 CUDA DLL

本项目固定 `onnxruntime-gpu==1.26.0`，因为 PyTorch 使用 CUDA 12.8，而 ORT 1.27 及以上的 PyPI wheel 已切换到 CUDA 13。预处理配置要求 `CUDAExecutionProvider`；如果只启用 CPU，实验会立即失败而不是静默变慢。不要同时安装 `onnxruntime` 和 `onnxruntime-gpu`。

## 研究边界与下一步

FEI-full E2 没有支持可靠的中间层优势。PhotoMaker V2 的八身份生成与 v2 人评均已完成：10 名不同评者完成 480 条展示试次，多参考的人类 likeness 收益未得到支持。该 pilot 已冻结，不继续追加评分或调参寻找显著结果。

下一轮主路线以面向初学者的 [中文研究计划](docs/RESEARCH_PLAN_ZH.md) 为准。新增自动工作、独立识别器、VAE 检查、第二模型验证与功效模拟见 [中文自动研究报告](docs/AUTO_RESEARCH_REPORT_ZH.md)；数据格式见 [输入说明](docs/AUTO_RESEARCH_INPUTS_ZH.md)。旧路线图与早期逐层设计保留作历史记录，协作状态以 [AGENTS.md](AGENTS.md) 为准。

自动研究使用独立环境 `.venv-auto`，PuLID 使用 `.venv-pulid`，不升级原 `.venv`：

```powershell
.\.venv-auto\Scripts\python.exe scripts/run_auto_research.py check
.\.venv-auto\Scripts\python.exe scripts/run_auto_research.py resume
.\.venv-auto\Scripts\python.exe scripts/run_auto_research.py summarize
```

执行配置在 `configs/auto_research.yaml`。GPU 串行，上限四小时；新增下载上限 30 GB。所有新结果存放于独立的 `artifacts/runs/20260905_auto_research/`，历史运行只读。自动分数与模拟响应均不能代替本人或陌生人的盲评。

2026-09-05 已完成 192 图补充诊断、56 图 VAE 重建、PhotoMaker 48 图、PuLID 16 图 smoke 后完整 48 图，以及数据/评分工具与功效模拟。累计下载约 3.41 GB、GPU 工作时间约 34.53 分钟；旧运行 1,337 个文件 hash 不变。219 项 CPU 测试与 Ruff 通过，实际环境的 `pip check` 声明例外已在报告中记录。PuLID 的自动身份分数较高，但尚无新本人/陌生人人评证明它改善感知 likeness。
