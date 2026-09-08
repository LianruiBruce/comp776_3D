# 人脸个性化生成中的“像本人”：现有实验审计与下一阶段路线

> **2026-09-05 更新：本文保留为历史文献审计与方案记录。v2 人评已完成，本文中的“尚无人评 / 0 responses / 正在收集”属于旧状态；结果以 [v2 协议与分析](E:/comp776_3D/docs/HUMAN_IDENTITY_LIKENESS_V2.md) 为准。后续研究优先级、核心课题范围与新手执行说明由 [中文研究计划](E:/comp776_3D/docs/RESEARCH_PLAN_ZH.md) 替代：先测量真实参照下的人类 likeness 缺口，再选择有证据的干预，不预先承诺 global/local router 或四类瓶颈的完整因果分解。**

> 版本：2026-09-04
>
> 适用硬件：NVIDIA RTX 4080 16 GB
>
> 文档性质：研究决策与实验流程；不是已经取得的新实验结果
>
> 研究主终点：盲法的人类感知身份相似度（perceived identity likeness）

> 2026-09-04 执行更新：当前只实施冻结生成结果上的 P0/P1/P3 身份相似度盲评，不同时加入 4-AFC、attention checks、prompt/quality 子实验或新模型。实际执行协议以 [H1 人类感知身份相似度实验](HUMAN_IDENTITY_LIKENESS_V2.md) 为准；本文后半部分列出的其他方法仅是以后可能采用的研究选项。

## 一页结论

这个项目值得继续，但主问题需要从“ViT 的某一层是不是丢了身份”改成：

> **人类觉得不像本人的信息，究竟在参考图片、身份编码、条件注入，还是扩散生成过程中丢失或没有被使用？怎样用局部、多尺度和多参考信息把它补回来？**

目前已有实验不是失败，而是完成了两次重要排除和一次定位：

1. **E1/E2 没有支持“最终 ViT 层擦除身份”。** E1 的 block 11 优势在等容量读出后几乎消失，E2 的预注册门槛未通过。因此不能把 block 11 当作“身份层”，也不应直接把它接进生成器。
2. **PhotoMaker V2 的身份条件确实有效。** K=1 相对 text-only 有很大的自动身份增益，所以问题不是“模型完全没看到是谁”。
3. **全局人脸 embedding 在现有生成器中占主导。** reciprocal donor 冲突里，独立 LVFace 在 60/64 张图上判断输出跟随全局 InsightFace 来源；但这不证明全局向量足以产生人类眼中的“神似”。

当前没有真实人评，因此还不能回答“哪种信息让人觉得更像本人”。当前唯一正在执行的下一步，是从冻结的 192 张图中取 P0/P1/P3 三个必要对照，完成一次只问身份相似度的分组盲评。

```text
多张自拍
   │
   ├─ ① 参考采样：这些照片是否覆盖了这个人的稳定外貌变化？
   ▼
身份编码器 ── ② 编码/读出：信息是否存在、是否可被下游读出？
   ▼
条件聚合与注入 ── ③ 注入：全局、局部、几何信息谁真正控制输出？
   ▼
扩散去噪 ── ④ 渲染：身份是否被 prompt、风格和生成先验覆盖？
   ▼
生成图 ── 独立识别器只能做代理；人类盲评回答“像不像”
```

## 1. 先把问题定义准确

### 1.1 这不是 target 图片的像素重建

生成任务不是“给一张 target，让模型一模一样地重建 target”。正确设置是：

- 输入：同一个人的一张或多张 reference 自拍；
- 生成要求：由 prompt 指定新的姿态、表情、光照或场景；
- 评估：拿**未送入生成器的同身份照片**组成 held-out gallery，判断生成图是否仍像这个人；
- 另行检查：生成器有没有直接复制 reference 的姿态、构图或纹理。

如果把输入 reference 当成唯一 target，分数会奖励 copy-paste，而不是真正的跨场景身份保持。

### 1.2 必须区分三个终点

| 终点 | 回答的问题 | 合适指标 | 不能替代什么 |
|---|---|---|---|
| 身份可识别性 | 能否从候选人中认出是谁？ | Rank-1、4-AFC、TAR@FAR | 细粒度“神似” |
| 连续身份保真 | 与真实同身份 gallery 在识别空间有多近？ | cosine、target-impostor margin | 人类判断 |
| 人类感知 likeness | 人是否觉得生成图像这个人？ | 方法盲化的 pairwise/评分实验 | 不能由 ArcFace 直接推断 |

[ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) 优化的是归一化角空间中的类内紧致和类间分离，不是人类 likeness 标注。[F-Bench/FaceQ](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_F-Bench_Rethinking_Human_Preference_Evaluation_Metrics_for_Benchmarking_Face_Generation_ICCV_2025_paper.html) 在 12,255 张图、491,130 次评分上把质量、真实性、身份保真和文本一致性分开；其中 ArcFace 与人类 ID-fidelity MOS 的 SRCC/PLCC 仅为 0.5062/0.5572，说明自动指标不能稳定替代人类终点。

### 1.3 “像我”和“陌生人能认出我”也不同

人对熟悉面孔的表征来自许多照片和长期观察；陌生评分者只能根据实验给出的 gallery 做匹配。人脸认知研究表明，同一个人的不同照片变化很大，而且 good likeness 判断受观察者熟悉度影响（[Jenkins et al., 2011](https://static1.squarespace.com/static/5e3fac24244c110e4dea7c2a/t/5e43af87685c425c5d5d7c39/1581494168773/JenkinsEtAl2011_Cognition.pdf)，[White et al., 2018](https://eprints.whiterose.ac.uk/id/eprint/122330/)）。因此未来应分开报告：

- **本人/熟人评价**：最接近产品问题“看起来像我”；
- **陌生评分者 + held-out gallery**：可规模化、可严格盲化，但只能称 gallery-based resemblance。

一篇在本报告前一天发布的产业预印本也采用了 10 名同意参与者、约 1,000 次本人/熟人盲配对（[Persistent Identity Preservation, 2026-09-03](https://arxiv.org/abs/2609.04151)）。它说明 familiarity-aware evaluation 已成为最新关注点，但尚未同行评审，而且把身份、质量和指令完成度合并成总体偏好；本项目只把它作为监测信号，不作为协议定论。

## 2. 既有实验流程审计

### 2.1 已经完成了什么

| 阶段 | 设置 | 关键结果 | 可以得出的结论 | 不能得出的结论 |
|---|---|---|---|---|
| E0 smoke | FEI neutral/smile；LVFace-T 12 blocks | 全流程、确定性、CUDA、final-head parity 均通过 | 工程链路可靠 | 任何生成或自然自拍结论 |
| E1 layer scan | FEI 200 人×14 条件；100/50/50 identity split | raw token mean 选 block 11；官方 head 仍以 block 12 最强 | 简单读出在层间有可访问性差异 | block 12 删除了身份 |
| E2 matched readout | 每层相同读出、5 seeds、dev 选层 | block 11−12 end-to-end TAR = +0.00067，95% CI [0, .002]；仅 1/5 seed 为正 | “最终层身份擦除”门槛未通过 | block 11 是生成器需要的身份层 |
| G0 generation pilot | PhotoMaker V2；8 IDs×6 conditions×2 prompts×2 seeds | 192/192；三种正常条件 Rank-1 都为 1 | 身份通路有效；工程可跑 | 人看起来更像本人 |
| G0 channel conflict | global/patch reciprocal donor | 输出随 global 来源 60/64 | 当前实现中 global 通道相对占主导 | global 足够；patch 无用 |
| H0 human package | v1：352 项 master；v2：P0/P1/P3 共 96 个 pair、10 份表单 | v2 分组、匿名媒体和解盲分析已构建；0 responses | 当前探索实验材料可直接评分 | 任何人类偏好结论 |

详细数值分别保存在 [E1 报告](../reports/E1_FEI_FULL_LVFACE_T.md)、[E2 报告](../reports/E2_FEI_FULL_LVFACE_PROBES.md) 和 [生成 pilot 报告](../reports/GENERATION_IDENTITY_PILOT.md)。

不可手改的本地证据路径是：

| 证据 | 路径 |
|---|---|
| E0 latest smoke | `artifacts/runs/20260901T051845Z_smoke_fei_aligned_lvface_t/` |
| E1 formal | `artifacts/runs/20260903T013013Z_fei_full_lvface_layers/` |
| E2 formal | `artifacts/runs/20260903T022924Z_fei_full_lvface_equal_capacity_probes/` |
| G0 formal | `artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot/` |
| G0 有效自动评价 | 上述 G0 run 的 `evaluation_v3/` 与 `analysis_v3/` |
| H0 人评原型 | v1 在上述 G0 run 的 `human_eval/`；当前 v2 在 `human_eval_v2/` |

### 2.2 对原始假设的判决

#### 假设 A：“预训练让模型朝平均脸生成”

当前证据**不支持这样表述**。正常生成 cohort 的身份间 spread 比真实 cohort 小约 9%–10%，可以称为识别 embedding 空间中的**身份间收缩**；但生成样本反而离真实 cohort 的字面 centroid 更远，不能称为“被吸向平均脸”。要检验平均化，需要预先定义 centroid、对照生成分布、跨 cohort 复现，并证明结果不只是评估器几何造成的。

#### 假设 B：“ViT 的后几层把身份特征丢了”

E2 是目前最直接的检验，结果为 null。E1 的 block 11 优势主要出现在 raw token mean；当每层使用相同容量的监督读出后，block 11 和 12 的内部差距只有一个 accepted query 的量级。更合理的解释是**读出不匹配或表示重组**，不是身份信息被删除。

#### 当前最合理的工作假设

识别 embedding 已经足以让生成器产生“这是某个人”的方向，但人类 likeness 依赖的局部比例、轮廓、五官组合、纹理和典型变化，可能：

1. 没被单张 reference 覆盖；
2. 在全局向量中不容易被生成器读出；
3. 存在于 patch/multiscale 特征中，但当前注入太弱；
4. 已经进入扩散网络，却被 prompt、风格或去噪阶段覆盖。

这四种原因必须用干预实验逐个区分。

### 2.3 现有生成与人评流程的具体缺口

- 只有 8 个已参与开发的 FEI 身份，且 FEI 是受控拍摄，不是手机自拍分布。
- P1 同时改变姿态、表情、光照和背景，无法归因到单一 nuisance。
- K=4 diverse 同时改变照片数量、姿态覆盖和照片质量，只能说明“新增信息”，不能说明是哪种信息。
- donor 是固定循环配对，没有按人口属性、姿态或独立识别难度匹配；冲突条件只能作 OOD 机制诊断。
- text-only 仍加载 PhotoMaker adapter/LoRA，不是纯 base-SDXL。
- 正式自动 CSV 尚未实现配置中列出的 prompt adherence、copying、seed diversity 和 image quality。
- 只有独立 LVFace 与复用的 InsightFace，缺少 AdaFace/CurricularFace 敏感性分析。
- 旧版 v1 要求每位 rater 完成全部 352 项，且 P0 的 win 方向容易反读；它只作为历史工程产物。
- 当前 v2 已收缩为 96 个 P0/P1/P3 pair：每位参与者 48 题，每个 pair 五票，participant ID、正方向、左右平衡和解盲 schema 已统一。
- v2 使用完整肖像，因此测量的是 whole-portrait identity likeness；不能把结果缩窄解释成只来自内部五官。
- 协议要求生成失败保留在分母，但当前正式 evaluator 只接受 manifest 全部 `status=complete`；本次 192/192 不受影响，未来含失败的 run 必须统一 schema 和统计规则。

因此，当前 v2 可以回答冻结 pilot 的三个探索问题，但不能承担自然自拍或局部五官机制的外推结论。

## 3. Related work 给出的共同方向

### 3.1 不是只用一个全局向量

| 方法 | 身份条件 | 对本项目最有价值的机制 | 证据边界 |
|---|---|---|---|
| [PhotoMaker, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.pdf) | 多张 reference 的 ID token 堆叠 | 多图不必简单平均 | 同行评审的是 V1 |
| [PhotoMaker V2 official code](https://github.com/TencentARC/PhotoMaker/blob/main/photomaker/model_v2.py) | 512-D face vector + CLIP patch features | 正好允许研究 global/patch 通道 | 截至本文日期仍缺完整 V2 technical report；性能主张不可当成同行评审消融 |
| [InstantID](https://arxiv.org/abs/2401.07519) | global face embedding + 五点几何；多图 embedding 平均 | 检验“全局身份 + 几何”与平均聚合 | 多图增益主要需自行严格复现 |
| [PuLID, NeurIPS 2024](https://arxiv.org/abs/2404.16022) | global face + EVA-CLIP global/multilevel features | 最关键的多层局部信息对照 | 论文自动身份评价仍不是人类真值 |
| [ConsistentID](https://arxiv.org/abs/2404.16771) | global context + 眼/鼻/嘴/耳局部 token | 显式五官区域因果实验的先例 | 环境较旧，需先解决可复现安装 |
| [Face2Diffusion, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shiohara_Face2Diffusion_for_Fast_and_Editable_Face_Personalization_CVPR_2024_paper.html) | 多尺度 ID + expression guidance | 将身份与表情解耦 | 不宜直接与当前 SDXL 配置混作公平主表 |
| [WithAnyone, ICLR 2026](https://openreview.net/forum?id=xFo13SaHQm) | 同身份 paired data + contrastive ID training | 把 reference copy-paste 与身份保真分开，MultiID-Bench 使用 prompt 对应 GT | 训练使用 8×H100，不是 4080 baseline；主要借鉴其评价逻辑 |
| [InfiniteYou, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_InfiniteYou_Flexible_Photo_Recrafting_While_Preserving_Your_Identity_ICCV_2025_paper.html) | 独立 identity residual path | 身份/文本/美学存在可测 trade-off | FLUX 与显存配置不同 |
| [OmniPortrait, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/64f4014a46a58a23bcfd9988e8204a12-Abstract-Conference.html) | coarse-to-fine dual stream + diffusion-feature guidance | 分阶段、分位置注入的最新先例 | 本次未找到可直接纳入的官方代码，先作设计依据 |

共同趋势是：**全局识别向量负责“是谁”，局部、多尺度或几何条件补充“具体长什么样”，再通过空间位置和去噪阶段控制二者如何被使用。** 这比“找一个神奇的 ViT 层”更接近当前方法的发展方向。

### 3.2 embedding 可能“有信息”，但下游未必会用

[ID2image](https://arxiv.org/abs/2304.07522) 和 [Arc2Face](https://arxiv.org/abs/2403.11641) 表明，人脸识别描述符可以泄露或恢复相当多的外观信息。这意味着三个命题不能混在一起：

1. **encoded**：信息是否存在于表示中；
2. **accessible**：一个受控读出是否能取出它；
3. **used**：生成器是否真的利用它改变输出。

E1/E2 主要研究前两项；donor swap 开始触及第三项。未来的 channel、region 和 timestep intervention 才能定位生成端的因果瓶颈。

## 4. 建议的正式研究假设

以下假设按顺序检验，前一阶段不通过时，不自动进入昂贵的下一阶段。

| 编号 | 可证伪假设 | 主要实验 | 继续门槛 |
|---|---|---|---|
| H1 | 多样 K=4 比 K=1 和重复 K=4 更像本人 | 冻结 G0 输出的人类 pairwise | 对 K=1 的方向明确，且对 repeated-K4 也一致；否则不声称多参考有效 |
| H2 | global 和 local/multiscale 提供互补的人类 likeness 信息 | 匹配 donor、global-only/local-only、完整通路 | 完整通路在人评上优于 global-only，且不是只靠 prompt/画质变化 |
| H3 | 某些局部区域是 H2 的主要来源 | 眼眉、鼻、嘴、轮廓、纹理/头发逐组 ablation 或 swap | 预注册区域产生可重复、方向一致的 pairwise 变化 |
| H4 | 身份条件在特定网络位置/去噪时段被覆盖 | early/mid/late timestep 与 down/mid/up injection | 相同总强度下出现稳定的人评增益而不明显损害 prompt/质量 |
| H5 | 小型 global+local router 能改善 likeness | 冻结 backbone，只训练 resampler/router/gate | 外部身份与新 prompt 上达到预设最小实际效应；独立自动指标只作支持 |

对于正式确认，可把“排除 tie 后 pairwise win probability 至少 0.60”作为**初始**最小实际效应提案；最终值必须在不看正式条件结果之前，根据探索阶段方差和研究成本冻结。

## 5. 分阶段实验流程

### Stage 1 — 当前正在执行的单一人评实验

本轮只问：“A 或 B 哪张生成照片更像三张 reference 中的同一个人？”材料全部来自已经冻结的生成结果，不重新生成或筛图。

- P0：diverse K=4 vs K=1；
- P1：diverse K=4 vs repeated K=4；
- P3：global-target/patch-donor vs global-donor/patch-target。

共有 96 个唯一 pair。它们被分成 10 份各 48 题的表单，每个 pair 由 5 名不同参与者评分；左右方向、prompt、seed、identity 和 contrast 在表单内平衡。只收集 `A / B / 相同 / 无法判断`，不混入其他任务。统计按 pair、identity、contrast 依次聚合，详细定义见 [H1 人类感知身份相似度实验](HUMAN_IDENTITY_LIKENESS_V2.md)。

### Stage 2 — 建立外部自然自拍 cohort

#### 推荐数据

- 目标：先招募 10–15 个新身份做 instrument/power pilot，再保留至少 40 个完全不参与调参的身份；资源允许时总计 60–90 人更稳妥。
- 每人 8–12 张自然自拍，覆盖但不混合记录：正/侧脸、表情、室内/室外光照、手机距离/镜头、眼镜/遮挡、发型/妆容等。
- 每人划分为：generator references、held-out likeness gallery、never-touched confirmation photos。
- 身份所有者本人和至少两位熟人只评价该身份，形成 familiar 子研究；另招募陌生评分者完成 gallery-based 盲评，两者分开建模和报告。

受控数据仍有辅助作用：[FEI](https://fei.edu.br/~cet/facedatabase.html) 可继续做开发回归；[Yale Face Database B](https://cvc.cs.yale.edu/cvc/projects/yalefacesB/yalefacesB.html) 适合冻结后的 pose/illumination 表示验证，但都不能代表自然自拍的最终结论。不要默认 VGGFace2、IJB 或任意抓取名人集允许派生身份生成和对外人评；逐项做许可与伦理审查。

#### Prompt 分层

第一轮一次只改变一种 nuisance：

1. 普通中性肖像；
2. 视角/姿态；
3. 表情；
4. 光照；
5. 遮挡或配饰；
6. 风格/背景。

正式主分析可先冻结前四类；遮挡和风格作为次要压力测试。每个 condition 使用相同 prompt、seed、sampler 和初始 latent，避免生成噪声掩盖身份机制差异。

### Stage 3 — 同 backbone 的方法基线

先在同一个 SDXL-compatible base、相同分辨率、精度、sampler、steps 和 seed 上比较：

1. **PhotoMaker V2**：现有主基线；
2. **InstantID**：global identity + landmark geometry；
3. **PuLID v1.1 SDXL**：global + multilevel local visual features；
4. **ConsistentID-SDXL**：只有在依赖可可靠 pin 住时加入显式 facial-region 对照。

另跑 **Arc2Face** 作为“单一 global ArcFace vector 是否足够”的诊断，不把它与自由文本 SDXL 方法混成公平排名。PuLID-FLUX、InfiniteYou 和 DreamO 使用不同 backbone 或量化/offload，放在独立的跨 backbone 附录，不进入核心因果表。

每个新方法先跑 16-cell smoke：2 identities × 2 prompt strata × 2 conditions × 2 seeds。记录成功率、每图时间、peak allocated/reserved、detector coverage 和输出 hash；通过后再冻结完整配置。

### Stage 4 — 定位信息在哪里失效

#### 4A. Encoder accessibility

在 identity-disjoint split 上，用相同容量的读出分别输入：

- global FR embedding；
- CLIP patch token / 多层 token；
- 显式眼眉、鼻、嘴、轮廓、纹理区域特征；
- global + local 组合。

预测目标不再只是身份分类，而是 held-out 人类 pairwise likeness、局部属性判断和 nuisance 稳定性。这样才能检验“人类需要的信息是否可读出”。

#### 4B. Channel causality

- 重新构造 global-target/local-donor 与 reciprocal 条件；donor 按独立 FR 相似度、姿态和可用人口属性匹配。
- 在官方接口允许且不会制造无效输入时，再做 global-only、local-only 和完整通路。
- 同图重复、多图重复和多图多样三个条件继续保留，以区分 token 数量与新信息。

#### 4C. Region causality

使用同一 reference、prompt、latent，逐组 mask、drop 或 swap：

- 眼睛与眉毛；
- 鼻；
- 嘴与下巴；
- 脸型/轮廓；
- 肤质与稳定纹理；
- 头发与外部头部特征。

不要一次同时更换五官、姿态和光照。每个 ablation 都同时评 identity crop、whole portrait、prompt 和 quality。

#### 4D. Injection causality

固定身份条件总强度，分别只在 early/mid/late 去噪窗口，或 down/mid/up 网络阶段注入。观察同一 latent 下输出变化。attention map 只能说明模型“看了哪里”，真正的因果证据来自干预后的人评与输出差异。

### Stage 5 — 只训练轻量方案

若 Stage 3–4 证明 local/multiscale 信息与人类 likeness 有稳定关系，建议的模型贡献是：

- 冻结 SDXL/UNet 主干和人脸编码器；
- 保留每张 reference 的局部 token，不先压成一个均值；
- 同时保留 global identity token；
- 训练小型 set resampler/router/gate，将局部 token 按脸部区域和去噪时段注入；
- 可选少量 LoRA，但不要在单卡上重新训练大 backbone。

损失可包括 diffusion/face-region reconstruction、训练专用 FR identity loss、局部特征一致性、文本一致性与 reference-copy penalty。最终评估不能复用训练损失中的识别器。

对照至少包含：现有 PhotoMaker V2、global-only、相同参数量但无 region/time routing 的 adapter，以及 strongest released local-feature baseline。

### Stage 6 — 冻结外部确认

- 锁定权重、reference 选择规则、prompt 模板、seed、sampler、identity strength 和分析代码。
- 只在未接触身份与未调参 prompt 上运行。
- detection/alignment 失败保留在分母；不得只报告成功检测的图。
- 同时报告本人/熟人和陌生评分者结果、自动评估器敏感性、prompt/quality trade-off、身份异质性和预注册 subgroup。
- 只有外部人类 primary endpoint 通过，才可写“改善了人类感知的本人相似度”。

## 6. 人评统计方案

### 6.1 最小正式设计

一个可执行的起点是：

- 至少 40 个完全未参与调参的身份，目标 60；
- 2 个 primary contrasts；
- 4 个 prompt strata；
- 2 seeds；
- 共 `40 × 2 × 4 × 2 = 640` 个 pairwise items；
- 每 item 至少 5 个独立判断，即约 3,200 个有效判断；
- 每位陌生评分者约 40–50 个主 item 加 10% attention checks，约需 70–80 人完成覆盖。

这不是最终 power 保证。应先用 10–15 个新身份估计 identity、rater 和 item 方差，再用 mixed-model simulation 确定样本量。若要检测身份级标准化配对效应约 `d=0.4–0.5`，独立近似通常落在约 34–52 个身份，因此“8 个 FEI 身份”只适合作 pilot。

### 6.2 主模型

- 二选一无 tie：mixed-effects logistic regression；condition 为 fixed effect，identity、rater、prompt 为 crossed random intercepts；数据足够时加入 identity random slope。
- 允许 tie：Bradley–Terry–Davidson 或等价的分层 pairwise 模型，显式建模 tie。
- 报告 estimated win probability、odds ratio、95% CI 和绝对差异。
- 以 identity 和 rater 聚类的 bootstrap 作为稳健性分析；不能把每张图或每一票当成完全独立样本。
- 多个 primary hypotheses 使用 Holm 校正；subgroup 若未预先有功效，明确标成 exploratory。
- 报告 unjudgeable、检测失败、排除比例和 inter-rater agreement，不只报告留下的数据。

## 7. 自动评价方案

### 7.1 身份评价

至少使用两个不同训练目标/实现来源的识别器：

- 保留当前独立 LVFace；
- 新增 [AdaFace](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html) 或 [CurricularFace](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_CurricularFace_Adaptive_Curriculum_Learning_Loss_for_Deep_Face_Recognition_CVPR_2020_paper.html)；最好二者都做敏感性分析；
- PhotoMaker/PuLID 使用的 InsightFace/ArcFace 风格 encoder 仅作 conditioner-aligned 诊断，不作唯一 primary evaluator。

每个身份用 held-out gallery 的 L2-normalized embedding 均值建 enrollment template，报告：

- generated-to-template cosine；
- target−nearest-impostor margin；
- Rank-1/Recall@K；
- validation negatives 冻结阈值后的 TAR@FAR=.01；
- face detection/alignment coverage；
- real-to-real 跨 nuisance 上界。

Rank-1 在当前 8-way 设置已饱和，正式分析应优先看 margin、阈值性能与人评，而不是继续追逐 Rank-1。

### 7.2 其余维度

分别报告，不默认合成一个总分：

- prompt adherence：自动文本图像分数 + 独立人评；
- facial/image quality：无参考质量诊断 + 独立人评；
- copy risk：生成图对 exact input 与 held-out same-ID 的相似关系，同时检查姿态/构图复制；有 prompt 对应真实 target 时，可借鉴 [WithAnyone/MultiID-Bench](https://openreview.net/forum?id=xFo13SaHQm) 的 reference–target–generation 区分，自由生成时不能伪造这个 target；
- diversity：相同身份、prompt 下跨 seed 的有效变化，防止“更像”只是输出坍缩；
- fairness/sensitivity：按预注册 nuisance 和人口属性分层，并报告不确定性。

## 8. RTX 4080 上的模型选择

| 模型 | 角色 | 16 GB 可行性 | 决策 |
|---|---|---|---|
| PhotoMaker V2 + RealVisXL V4 | 当前主基线 | 已实测；peak allocated/reserved 约 6.87/14.52 GB，9.32 s/图 | 立即复用 |
| InstantID SDXL | global + geometry 对照 | 官方支持 offload/tiling；本机尚未实测 | 先 16-cell smoke |
| PuLID v1.1 SDXL | global + multilevel local 核心对照 | 预计可行；官方无本机精确值 | 先 16-cell smoke |
| ConsistentID-SDXL | 显式五官局部对照 | 显存与旧依赖需验证 | 第二优先级 |
| Arc2Face | global-vector sufficiency 诊断 | SD1.5，较轻 | 诊断用，不进公平主排名 |
| PuLID-FLUX | 跨 backbone | 官方需 FP8/offload 才低于约 15 GB，且 FP8 会影响面部细节 | 暂缓核心实验 |
| InfiniteYou | 跨 backbone | 官方约 16 GB 需 8-bit+offload，处于显存边界 | 暂缓核心实验 |

不同 backbone、量化和 offload 可能同时改变基础画质与面部细节，不能把它们与身份注入机制的效果混为一谈。核心表先保持 SDXL、精度和采样设置一致。

## 9. 规模、算力与产物预算

- 现有 PhotoMaker V2 实测 192 图用时 1,788.8 秒。
- 若正式同-backbone benchmark 为 40 identities × 3 methods × 4 prompt strata × 2 seeds，共 960 图；仅按 PhotoMaker 当前速度线性估算约 2.5 小时。InstantID/PuLID 必须用 smoke 实测后再预算，不能直接套用。
- 640 个 pairwise items × 5 judgments = 3,200 个有效判断；attention、退出和无效 response 需另留余量。
- 生成图片、face crops、embeddings、response、private key 全部留在 Git-ignored run 目录；GitHub 只提交无身份图像的 manifest、配置、代码与聚合报告。

每个新正式 run 必须保存：resolved config、asset lock、source snapshot、完整环境锁、manifest snapshot、输出 hash、detector coverage、metrics、selection、performance、log 和 status。完成的 run 不手改；任何修正生成新 run/evaluation 目录。

## 10. 当前工程文件

本轮只增加一对构包/分析模块及其命令行入口：

```text
src/id_layers/grouped_human_eval.py
src/id_layers/grouped_human_eval_analysis.py
scripts/build_grouped_human_eval.py
scripts/analyze_grouped_human_eval.py
docs/HUMAN_IDENTITY_LIKENESS_V2.md
```

旧的 `human_eval/` 保留不动；新的 10 份表单、匿名媒体、私钥和 response 全部写入 Git 忽略的 `human_eval_v2/`。

## 11. 当前结果如何决定下一步

当前先完成 P0/P1/P3 人评，不并行增加新模型或新指标。收齐评分后按三个结果直接解释：

1. P0 和 P1 都偏向 diverse K=4：继续研究多张自拍中哪些局部或视角信息有效；
2. P0 有差异而 P1 没有：现有结果不能把收益归因于参考多样性；
3. P3 明显偏向 global=target：下一步研究全局 embedding 已保留但生成端没有还原的细粒度信息；若偏向 patch=target，则优先研究局部视觉条件；
4. 三个结果都接近 0.5：保留 null 结果，并把后续工作转向新的自然自拍身份和更直接的局部信息干预。

在本轮评分完成前，最近的执行顺序只有三步：构建 10 份表单、每份交给一名不同参与者、用冻结的分析脚本汇总结果。

## 13. 最终可以形成的论文贡献

最稳妥的论文故事不是“我们发现了 ViT 第 11 层”，而是：

1. 一个把 recognition、continuous fidelity 和 human likeness 严格分开的自拍身份生成协议；
2. 对 reference、global/local encoding、injection site/time 和 denoising 四类瓶颈的因果分解；
3. 一个在单张 16 GB GPU 上可训练/运行的轻量 global+local 路由方法；
4. 在本人/熟人与陌生评分者上分别验证，并在外部身份和 prompt 上冻结复现。

即使最后没有新 adapter，只要人类实验和因果干预严谨地证明“自动识别正确但不像本人”的具体来源，也已经比继续扫描 ViT 层更贴近原始问题，并且可以形成有价值的研究结论。

## 参考的一手论文与官方实现

- [ArcFace — CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)
- [AdaFace — CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)
- [CurricularFace — CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_CurricularFace_Adaptive_Curriculum_Learning_Loss_for_Deep_Face_Recognition_CVPR_2020_paper.html)
- [PhotoMaker — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.pdf)；[官方代码](https://github.com/TencentARC/PhotoMaker)
- [InstantID](https://arxiv.org/abs/2401.07519)；[官方代码](https://github.com/instantX-research/InstantID)
- [PuLID — NeurIPS 2024](https://arxiv.org/abs/2404.16022)；[官方代码](https://github.com/ToTheBeginning/PuLID)
- [ConsistentID](https://arxiv.org/abs/2404.16771)；[官方代码](https://github.com/JackAILab/ConsistentID)
- [Face2Diffusion — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shiohara_Face2Diffusion_for_Fast_and_Editable_Face_Personalization_CVPR_2024_paper.html)
- [WithAnyone / MultiID-Bench — ICLR 2026](https://openreview.net/forum?id=xFo13SaHQm)
- [InfiniteYou — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_InfiniteYou_Flexible_Photo_Recrafting_While_Preserving_Your_Identity_ICCV_2025_paper.html)；[官方代码与显存说明](https://github.com/bytedance/InfiniteYou)
- [OmniPortrait — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/64f4014a46a58a23bcfd9988e8204a12-Abstract-Conference.html)
- [Arc2Face — ECCV 2024](https://arxiv.org/abs/2403.11641)
- [ID2image](https://arxiv.org/abs/2304.07522)
- [F-Bench/FaceQ — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_F-Bench_Rethinking_Human_Preference_Evaluation_Metrics_for_Benchmarking_Face_Generation_ICCV_2025_paper.html)
- [DreamBench++ — ICLR 2025](https://dreambenchplus.github.io/)；[官方代码](https://github.com/yuangpeng/dreambench_plus)
- [Stellar](https://arxiv.org/abs/2312.06116)；[官方指标代码](https://github.com/stellar-gen-ai/stellar-metrics)
- [What makes a face photo a “good likeness”? — Cognition](https://eprints.whiterose.ac.uk/id/eprint/122330/)
- [Variability in photos of the same face — Cognition](https://static1.squarespace.com/static/5e3fac24244c110e4dea7c2a/t/5e43af87685c425c5d5d7c39/1581494168773/JenkinsEtAl2011_Cognition.pdf)
- [Persistent Identity Preservation — 2026-09-03 preprint](https://arxiv.org/abs/2609.04151)

## 证据限制

- 文献中的人脸相似度经常来自与 conditioner 同源或近源的 ArcFace/InsightFace，可能偏爱该方法自己的表示空间。
- PhotoMaker V2 的双通道结论来自官方代码；截至研究日期，没有把全部 V2 训练与消融细节写清的正式技术报告。
- 新近论文的人评规模、是否熟悉身份、是否把质量/文本/身份混成一个问题差异很大，不能直接横向比较百分比。
- 4080 可行性只有 PhotoMaker V2 是本仓库实测；其余模型必须先 smoke。
- 自然自拍 cohort 涉及生物特征数据、衍生生成图和人类参与者，合规方案是实验设计的一部分，不是事后补充。
