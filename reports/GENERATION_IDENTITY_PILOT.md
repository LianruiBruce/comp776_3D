# Generation Pilot — PhotoMaker V2 生成人脸身份保真诊断

## 状态与结论

- 正式运行：`artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot`
- 正式自动评估：`evaluation_v3/status.json` 为 `complete`，汇总分析 `analysis_v3/status.json` 为 `complete`；正式矩阵生成并评估 `192/192`，无生成失败、无人脸或多脸输出。
- 当前研究状态：自动评估和 v2 盲法人类评价均已完成；run 根状态文件仍保留生成阶段写入的 `automatic_evaluation_complete_human_pending`，正式人评结果位于 `human_eval_v2/analysis/analysis.json`。
- 研究范围：8 个已在项目开发中接触过的 FEI `internal_eval` 身份上的探索性输出诊断；不是外部确认、SOTA 比较或对总体人群的推断。

本实验得到四个主要自动指标结论：

1. **PhotoMaker V2 的原生身份通路确实强烈改变了输出。** 独立 LVFace 评估下，K=1 相对 text-only 的目标相似度提高 `+0.552276`，95% identity-bootstrap 区间为 `[+0.511392, +0.592464]`；8-way Rank-1 提高 `+0.9375`，区间为 `[+0.84375, +1.00000]`。
2. **“能分到正确身份”不等于“细节上像本人”。** K=1、K=4 diverse 和 K=4 repeat 的 8-way Rank-1 都是 `1.0`，但独立 LVFace 目标余弦仅为 `0.5613–0.5743`；真实 K=1 参考图对未参与生成的真实查询模板为 `0.9132`。生成结果保留了足以击败 7 个 impostor 的身份方向，但离真实自拍身份簇仍很远。这与“看得出大概是谁、却不像本人”的现象一致，但尚不能替代人类感知证据。
3. **冲突干预中，512-D global face embedding 明显主导 CLIP patch 图像通路。** LVFace 判定输出跟随 global 身份 `60/64=93.75%`，复用的 InsightFace 判定为 `63/64=98.44%`。这是相对通路影响的证据，不等于 global embedding 已包含人类判断相似度所需的全部细节。
4. **多张且多样的参考图只有小幅自动指标增益，尚不足以证明人眼更像。** K=4 diverse 相对 K=1 的 LVFace 目标相似度点估计为 `+0.013093`，但 95% 区间跨零；相对 K=4 repeat 为 `+0.013894`，目标相似度区间刚好高于零，但 target-impostor margin 的区间仍跨零，Rank-1 完全不变。

v2 人评进一步表明：P0（diverse K=4 对 K=1）为 `0.47969 [0.38698, 0.57083]`，P1（diverse K=4 对 repeated K=4）为 `0.50573 [0.43880, 0.58256]`，均未显示可检测的多参考感知收益。P3（global-target 对 patch-target）为 `0.38385 [0.13906, 0.66615]`；可判断票偏向 patch-target，但 69/160 无法判断且区间跨 0.5。因此自动 recognizer 的 global-following 与人类可能使用 patch/local cues 之间存在值得外部确认的张力，但本 pilot 不能判定哪个通道普遍主导人类 likeness。

此外，三种正常身份条件的 LVFace 身份间几何只有真实查询模板的 `90.19%–91.33%`，即约 `8.67%–9.81%` 的 between-identity contraction。该结果只说明生成身份在嵌入空间中更难区分；它**不支持**“输出被拉向这个真实 cohort 的均值”，更不能证明预训练人口均值或某个 ViT layer 导致了 mean face。

## 研究问题

该实验直接检查生成器输出，而不再用 E1/E2 的 layer 指标间接解释生成质量：

1. 单张原生身份参考是否使输出靠近目标身份？
2. 固定 K=4 时，视角/尺度多样性是否比把同一张照片重复四次提供更多身份信息？
3. PhotoMaker V2 的 global InsightFace embedding 与 CLIP patch 像素来自不同人时，输出跟随哪一路？
4. 身份控制能否承受指定姿态、表情、光照和背景的 prompt 变化？
5. 独立识别器、复用 conditioner 识别器和盲法人评是否得出一致结论？
6. 生成身份的 cohort-level 几何是否发生收缩？

该设计只能定位输出层面的症状，不能区分身份信息是在 face encoder、QFormer、fusion、U-Net 注入还是某个 denoising step 中丢失。

## 冻结协议

### 数据与身份

- 数据：官方 FEI original images；固定使用 8 个 `internal_eval` 身份：`fei_0058`、`fei_0075`、`fei_0092`、`fei_0100`、`fei_0115`、`fei_0162`、`fei_0168`、`fei_0193`。
- 每个目标的 donor 是上述列表中的下一个身份，末尾循环回第一个；该映射在观察生成结果前冻结。
- K=1：`frontal_neutral`。
- K=4 diverse：`frontal_neutral`、`left_profile`、`right_profile`、`frontal_scale`。
- K=4 repeat：精确重复同一张 `frontal_neutral` 四次，不使用四张近重复图。
- 真实查询模板：`frontal_smile`、`left_three_quarter`、`right_three_quarter_expression` 三张图逐图 L2 normalize、求均值并再次 L2 normalize。查询条件与生成器参考条件不重叠。

所有 56 个所需身份—条件输入在冻结 manifest 中存在。生成使用原始图；真实查询评估使用已固定的五点对齐结果。没有按照生成效果换参考图、换身份或删除失败样本。

### 生成系统

- 官方 PhotoMaker V2 代码 revision `060b4fcb10b76a4554edf565d6106b7e36c968f0`。
- PhotoMaker V2 adapter revision `f5a1e5155dc02166253fa7e29d13519f5ba22eac`；权重 SHA256 `0eeac57d7e09b95cee670a18e0d939ff303dd6f76d317c950bf584c6c9bdb8f5`。
- RealVisXL V4.0 FP16 base revision `26dfe44930964cd70d0a817b6d1cc945c130e38d`。
- Euler scheduler，`1024×1024`，batch size 1，FP16，30 steps，CFG `5.0`，RTX 4080 上 model CPU offload。
- 五个身份输入条件使用 `start_merge_step=10`；text-only 使用 `start_merge_step=30`，使 30 次 denoising 都不使用 fused identity embedding。它仍加载 PhotoMaker adapter 和已融合 LoRA，因此不是原始 base-SDXL 对照。
- 两个 prompt：P0 为正面、中性表情、均匀光照和灰背景；P1 要求左侧三分之四视角、微笑、暖侧光与咖啡馆背景。
- base seeds 为 `776`、`1776`；`effective_seed = base_seed + identity_ordinal × 100000`。
- 同一身份和 base seed 的六个条件及两个 prompt 共享完全相同的 initial latent；不同身份不共享。验证得到 16 个 paired-latent groups。
- 完整矩阵为 `8 identities × 6 conditions × 2 prompts × 2 seeds = 192`。每格只生成一张，无 best-of、重抽 seed、人工挑图或失败重跑。

六个条件为：

| 条件 | Global 512-D 身份来源 | CLIP patch 像素来源 | 目的 |
|---|---|---|---|
| `k1_full` | target | target K=1 | 原生单参考正条件 |
| `k4_diverse_full` | target diverse K=4 | target diverse K=4 | 多视角/尺度参考 |
| `k4_repeat_full` | target K=1 重复四次 | target K=1 重复四次 | 等 K、无新增信息对照 |
| `global_target_patch_donor` | target | donor | 通路冲突干预 |
| `global_donor_patch_target` | donor | target | 反向通路冲突干预 |
| `text_only` | 计算但不进入 denoising | 计算但不进入 denoising | 关闭 denoising-time 身份条件 |

### 评估器与指标

主要自动评估器是固定的 **LVFace-T Glint360K official final embedding**。PhotoMaker V2 不以 LVFace 作为 conditioner 或生成 loss，因此它对本次原生 PhotoMaker 诊断具有 method-specific independence。禁止把 E1 的 block 11 或 E2 probe 当作生成评估器。

次级诊断是 PhotoMaker conditioner 同系列的 InsightFace `w600k_r50.onnx`。它可能偏向生成器所优化的特征，只用于检查 evaluator alignment，不能单独支持身份保真结论。

对生成 embedding `g` 和目标真实查询模板 `q_i`，核心连续指标为：

```text
target-impostor margin = cosine(g, q_i) - max_{j != i} cosine(g, q_j)
```

同时报告 8-way gallery Rank-1、目标/捐赠者相似度、冲突条件 global-follow rate、检测覆盖和 cohort geometry。所有不确定性均先在身份内聚合，再以身份为 bootstrap 单元进行 1,000 次 percentile bootstrap；固定 seed 为 `20260903`。只有 8 个 cluster，区间应视为描述性而非总体显著性检验。

## 运行完整性与性能

| 检查项 | 结果 |
|---|---:|
| 计划 / 观察到的唯一 factorial cells | 192 / 192 |
| 完成生成 | 192 / 192 |
| 输出 SHA256 验证通过 | 192 / 192 |
| 检测到人脸 | 192 / 192 |
| 恰好单脸并可对齐 | 192 / 192 |
| Paired initial-latent groups | 16 |
| 总生成时间 | 1788.832 s（29.814 min） |
| 平均 / 中位单图时间 | 9.317 / 9.146 s |
| 单图最短 / 最长时间 | 9.016 / 13.523 s |
| 吞吐 | 0.1073 image/s |
| Peak CUDA allocated / reserved | 6.395 / 13.527 GiB |

第一次自动评估写入的旧 `evaluation/` 因 text-only route validator 预期错误而在编码前停止；它没有改变任何生成图。修正 validator 后，`evaluation_v2/` 首次成功，随后从同一冻结生成 manifest 独立复跑为正式 `evaluation_v3/`。两次成功评估的 per-image metrics、condition/prompt aggregates、paired bootstrap、cohort geometry、real baseline、query coverage、validation、embeddings 和 summary 共 9 个核心文件 SHA256 均相同。run 根 `status.json` 已清除旧错误字段，`state` 为 `automatic_evaluation_complete_human_pending` 并指向 `evaluation_v3/summary.json`。

## 主要自动结果：独立 LVFace

下表跨两个 prompt 和两个 seed 聚合，每行 `n=32`。冲突条件中的 “目标” 始终是原始 target；因此在 `global_donor_patch_target` 中，它是 patch 身份而不是 global 身份。

| 条件 | 目标 Rank-1 | 目标 cosine | Target–max-impostor margin | Global-follow rate |
|---|---:|---:|---:|---:|
| K=1 full | 1.00000 | 0.561250 | 0.449639 | — |
| K=4 diverse full | 1.00000 | **0.574343** | **0.454802** | — |
| K=4 repeated full | 1.00000 | 0.560449 | 0.447715 | — |
| Global target + patch donor | 0.90625 | 0.316294 | 0.176162 | 0.90625 |
| Global donor + patch target | 0.03125 | 0.050107 | -0.237389 | **0.96875** |
| Text-only | 0.06250 | 0.008974 | -0.095490 | — |

### 原生身份通路

K=1 相对 text-only 的配对 identity-bootstrap 结果为：

| 指标 | K=1 − text-only | 95% interval |
|---|---:|---:|
| 目标 cosine | +0.552276 | [+0.511392, +0.592464] |
| Target–impostor margin | +0.545129 | [+0.489712, +0.598356] |
| End-to-end Rank-1 | +0.93750 | [+0.84375, +1.00000] |

因此，对这 8 个身份和冻结 seeds，身份通路具有清楚的输出效应。这个因果结论只针对“完整原生身份条件开/关”，不能拆分到 encoder、QFormer、fusion 或 U-Net。

### 多参考：数量与信息多样性

| 对比 | 指标 | 点估计 | 95% interval |
|---|---|---:|---:|
| K=4 diverse − K=1 | 目标 cosine | +0.013093 | [-0.010806, +0.038127] |
| K=4 diverse − K=1 | Margin | +0.005164 | [-0.020482, +0.028528] |
| K=4 diverse − K=1 | Rank-1 | 0.00000 | [0.00000, 0.00000] |
| K=4 diverse − K=4 repeat | 目标 cosine | +0.013894 | [+0.000119, +0.029653] |
| K=4 diverse − K=4 repeat | Margin | +0.007087 | [-0.007348, +0.020811] |
| K=4 diverse − K=4 repeat | Rank-1 | 0.00000 | [0.00000, 0.00000] |

K=4 repeat 和 K=1 的跨 prompt 均值几乎相同：目标 cosine 相差 `-0.000802`，95% 区间 `[-0.015127, +0.011202]`；margin 相差 `-0.001923`，区间 `[-0.017050, +0.011636]`；Rank-1 差为零。因此，单纯把同一信号堆叠四次没有可见收益；多样 K=4 的小幅增益更可能来自新增视角/尺度信息，而非输入数量本身。不过独立 evaluator 的 margin 区间仍跨零，且 Rank-1 已饱和，不能据此声称人眼相似度提升。

### Prompt nuisance

三种正常身份条件的 P1-minus-P0 点估计为：

| 条件 | Δ目标 cosine [95% interval] | Δmargin [95% interval] | ΔRank-1 |
|---|---:|---:|---:|
| K=1 full | -0.000559 [-0.017934, +0.014463] | -0.002953 [-0.030823, +0.022026] | 0.00000 |
| K=4 diverse full | +0.001783 [-0.021834, +0.027384] | -0.006945 [-0.042094, +0.026726] | 0.00000 |
| K=4 repeat full | -0.009840 [-0.036934, +0.018701] | -0.013128 [-0.044145, +0.022754] | 0.00000 |

在这一个 nuisance prompt 下没有身份崩溃；K=4 repeat 的下降略大，但绝对量仍小。本实验只有两个 prompt，不能推广到遮挡、极端姿态、年龄变化或复杂多人场景。

### Global 与 patch 通路冲突

| 评估器 | Global=target 时跟随 global | Global=donor 时跟随 global | 合并 |
|---|---:|---:|---:|
| LVFace independent | 29/32 = 90.625% | 31/32 = 96.875% | **60/64 = 93.750%** |
| InsightFace conditioner | 31/32 = 96.875% | 32/32 = 100.000% | **63/64 = 98.438%** |

LVFace 下，global 相对 patch 身份的平均 cosine 优势分别为 `+0.233166`（95% 区间 `[+0.091769, +0.378493]`）和 `+0.237389`（`[+0.131717, +0.344579]`）；对应 global-follow 区间分别为 `[0.71875, 1.00000]` 和 `[0.90625, 1.00000]`。InsightFace 下的 cosine 优势分别为 `+0.335745` 和 `+0.334029`。两个评估器方向一致：当 global 与 patch 指向不同人时，输出通常跟随 global 512-D face embedding。

该干预是 PhotoMaker 正常使用分布之外的人为冲突。它说明 global 通路的相对控制更强，不说明 patch 通路无用，也不说明把 global embedding 单独送入任意生成器就能恢复人类在意的痣、眼形、脸部比例或局部纹理。

## 真实照片基线与“Rank-1 饱和”

真实 K=1 `frontal_neutral` 参考图与三个未参与生成的真实查询形成如下基线：

| 来源 | LVFace Rank-1 | 目标 cosine | Margin |
|---|---:|---:|---:|
| 真实 K=1 reference | 1.00000 | **0.913164** | **0.819500** |
| 生成 K=1 | 1.00000 | 0.561250 | 0.449639 |
| 生成 K=4 diverse | 1.00000 | 0.574343 | 0.454802 |
| 生成 K=4 repeat | 1.00000 | 0.560449 | 0.447715 |

对应的 generated-minus-real 目标 cosine gap 为 K=1 `-0.351914`（95% 区间 `[-0.379579, -0.321888]`）、K=4 diverse `-0.338821`（`[-0.384007, -0.300832]`）、K=4 repeat `-0.352715`（`[-0.389290, -0.322074]`）；margin gap 分别为 `-0.369861`（`[-0.397223, -0.343604]`）、`-0.364697`（`[-0.416742, -0.320572]`）、`-0.371785`（`[-0.412992, -0.336895]`）。

这解释了为什么只报告 face-recognition accuracy 会过度乐观：在仅有 8 个候选身份时，一个样本只要比 7 个其他人更接近 target 就能得到 Rank-1=1，而无需落入该人的真实照片簇。当前结果最支持的表述是：**PhotoMaker 保留了强的类别级身份方向，但生成结果与真实同一人的细粒度距离仍有较大差距。** “人类觉得不像”的最终判断必须由后续盲评给出。

## 次级 InsightFace 诊断

| 条件 | 目标 Rank-1 | 目标 cosine | Margin | Global-follow rate |
|---|---:|---:|---:|---:|
| K=1 full | 1.00000 | 0.599259 | 0.486075 | — |
| K=4 diverse full | 1.00000 | **0.618019** | **0.501490** | — |
| K=4 repeated full | 1.00000 | 0.599620 | 0.493234 | — |
| Global target + patch donor | 0.96875 | 0.419316 | 0.278923 | 0.96875 |
| Global donor + patch target | 0.00000 | 0.061994 | -0.334029 | **1.00000** |
| Text-only | 0.09375 | 0.014337 | -0.084863 | — |

InsightFace 同样认为身份通路有效且 global 通路占主导。它对 K=4 diverse 的增益略大：相对 K=1 的目标 cosine 为 `+0.018759`，95% 区间 `[+0.002049, +0.034808]`；但 margin 为 `+0.015415`，区间 `[-0.007716, +0.039935]`。由于 PhotoMaker 本身使用 InsightFace 风格身份信息，这个更乐观的相似度增益可能包含 evaluator reuse alignment，故不能覆盖独立 LVFace 或人评结果。

InsightFace 的真实 K=1 基线目标 cosine/margin 为 `0.927591/0.813628`，同样远高于三个正常生成条件的 `0.599–0.618/0.486–0.501`。

## Cohort-level identity contraction

令每个身份的真实查询模板为 `r_i`，某条件下跨 prompt/seed 的生成身份中心为 `g_i`，并定义：

```text
B(X) = mean_{i<j}(1 - cosine(x_i, x_j))
contraction ratio = B(G) / B(R)
```

LVFace 的真实身份间 spread 为 `0.979951`，centered effective rank 为 `6.91953`（8 个身份的理论最大值为 7）。正常生成条件结果如下：

| 条件 | Generated spread | Contraction ratio | Spread reduction | Effective rank | Real-cohort-centroid shift |
|---|---:|---:|---:|---:|---:|
| K=1 full | 0.888207 | 0.906379 | 9.36% | 6.73090 | -0.092920 |
| K=4 diverse full | 0.883806 | 0.901888 | 9.81% | 6.77838 | -0.073447 |
| K=4 repeat full | 0.895024 | 0.913336 | 8.67% | 6.79131 | -0.100282 |

Text-only 的 contraction ratio 为 `0.462417`，说明相同 prompt/base-model 先验在没有身份注入时产生了更强的 cohort collapse；两个冲突条件的 ratio 为 `0.848952`（global target）和 `0.887250`（global donor），只作为 OOD 诊断。InsightFace 对三个正常条件得到相同数量级的 ratio：`0.895366`、`0.905419`、`0.911592`。

必须区分两个不同命题：

- `ratio < 1` 表明这 8 个生成身份的相互距离比真实身份模板小，即 **between-identity embedding contraction**；
- 它不等于生成结果趋向“真实 cohort 平均脸”。事实上，LVFace 的 `generated_real_cohort_centroid_similarity - real_real_cohort_centroid_similarity` 在 K=1、K=4 diverse、K=4 repeat 下分别为 `-0.092920`、`-0.073447`、`-0.100282`，方向不是向这个真实 cohort centroid 靠近。

因此，本实验不能声称 literal mean-face attraction，更不能从 8 个身份的几何收缩反推训练集人口均值、预训练数据偏差或最终 ViT layer 的因果机制。

## 人类评价：identity-likeness v2 已完成

旧版匿名 master 包位于 `human_eval/participant/index.html`，包含：

- 192 个 four-alternative forced-choice identity trials：目标、固定 donor、两个确定性 non-donor，并允许 `none` 和 `unjudgeable`；
- 160 个成对比较：冻结协议的四个对比共 128 项，另加 32 个探索性的 K=1 vs K=4 diverse 直接比较（`P0`）；
- 共 352 个随机顺序、随机左右位置的 blinded trials；公开包不暴露身份、方法名、条件名或源文件名。

v1 页面保留为历史工程产物，不再作为当前实验。当前 v2 位于 `human_eval_v2/`，只保留三个直接相关的 pairwise 对照：P0（diverse K=4 对 K=1）、P1（diverse K=4 对 repeated K=4）和 P3（global/patch reciprocal conflict）。它包含 96 个唯一 pair，分成 `A01–A05`、`B01–B05` 十份各 48 题的表单；每个 pair 恰好获得 5 个展示位置。每份表单的 identity、prompt、base seed、contrast 和左右方向均平衡，而且同一参与者不会重复看到同一张候选生成图。

页面只问哪张图更像三张 held-out reference 中的同一个人，允许 A、B、相同和无法判断。10 名不同参与者完成 10 份表单，共 480 次判断；分析按唯一 pair、identity、contrast 依次等权聚合，并计算 identity-bootstrap 95% 区间。原始 participant code 的重复经收集者确认来自不同的人；分析只在副本中以 form-derived pseudonym 消除代码碰撞，原始 response 未修改。完整定义、结果和运行命令见 [H1 人类感知身份相似度实验](../docs/HUMAN_IDENTITY_LIKENESS_V2.md)。

| 对照 | 偏好分数 | 95% CI | focal / tie / comparator | 无法判断 |
|---|---:|---:|---:|---:|
| P0：diverse K=4 vs K=1 | 0.47969 | [0.38698, 0.57083] | 35 / 56 / 36 | 33 |
| P1：diverse K=4 vs repeated K=4 | 0.50573 | [0.43880, 0.58256] | 37 / 54 / 36 | 33 |
| P3：global-target vs patch-target | 0.38385 | [0.13906, 0.66615] | 35 / 3 / 53 | 69 |

所有区间均覆盖 0.5。P0/P1 是感知层面的 null pilot；P3 是方向上偏向 patch-target、但缺失率高且身份异质的未决结果。

## 对“哪种信息更重要”的当前回答

按现有自动证据，最合理的优先级是：

1. **Global face-recognition embedding 是主要的自动识别器身份控制信号。** 冲突输出有 93.75% 被独立 LVFace 判为跟随 global 身份；这不代表人类判断也由它主导。
2. **当前多样参考没有显示人类感知收益。** 自动 target cosine 的小幅增益没有转化为 P0/P1 的可检测人评优势。
3. **Patch/local cues 是一个待验证的竞争解释，不是唯一后续方向。** P3 可判断票偏向 patch-target，而高无法判断率、跨 0.5 的区间和身份异质性阻止强结论；冲突条件也不是单通路充分性测试。当前人评没有真实照片候选参照，尚不能直接估计生成图的人类 likeness 缺口。
4. **真正缺失的很可能不是“身份是否存在”，而是身份内部的细粒度保真。** 正常生成条件的 Rank-1 全部正确，但与真实身份模板仍有约 `0.34–0.35` 的 cosine gap，并伴随约 9% 的身份间 spread 收缩。

第 4 点是测量层面的诊断，不是对某个神经网络层或预训练均值的机制结论。最终要回答“哪种信息让人类看来更像”，必须把人类 pairwise likeness 作为主要结果，再检验它是否由 global embedding、参考多样性、局部形状/纹理或其他条件解释。

## 局限

- 只有 8 个、且已被开发过程接触过的 FEI 身份；8-way Rank-1 很容易饱和，也无法支持人口层面的统计推断或公平性结论。
- 只有一个生成器、一个 adapter checkpoint、两个 prompts、两个 seeds 和一个 merge/CFG 配置；没有 identity-scale、merge-step、遮挡、年龄或极端姿态 sweep。
- P1 同时改变姿态、表情、光照和背景，不能把差异归因到单一 nuisance。
- Global/patch conflict 是 OOD intervention，只能比较相对通路控制，不能代表 PhotoMaker 正常输入分布。
- LVFace 对本方法是独立 evaluator，但仍是单一 face-recognition family；正式生成结论还需要固定的 AdaFace/CurricularFace-family sensitivity evaluator。
- InsightFace 是 conditioner-aligned evaluator，可能高估与生成器训练目标一致的特征。
- 真实 K=1 基线和生成结果之间的 embedding gap 不能直接换算成人类相似度；v2 人评已完成，但只有 8 个身份，三个 identity-bootstrap 区间均覆盖 0.5。
- Cohort geometry 只有 8 个身份，effective rank 最高为 7；它是描述性诊断，不识别 literal average face 或预训练机制。
- 本实验没有内部 activation ablation，不能定位 QFormer、fusion、U-Net 或 ViT layer 的信息损失，也不能把 E2 的 null matched-layer 结果改写成正结果。
- FEI、InsightFace-dependent weights、生成图和 embeddings 受研究用途与隐私约束，均保持本地且 Git-ignored；本报告不包含人脸图像。

## 下一步

1. 冻结本轮人评，不在相同 8 个身份上继续加人、调 prompt/donor 或改变分析规则。
2. 按 [2026-09-05 中文研究计划](E:/comp776_3D/docs/RESEARCH_PLAN_ZH.md)，先在新身份上建立真实候选参照，分开本人与陌生图库评分，测量中性与微笑条件下的 likeness 缺口；开发预试验与正式确认分离。P0/P1 暂不继续扩展。
3. 根据可靠诊断选择一种受控干预；global/local 只保留为竞争解释。Mapper/router 仍需独立的人类证据，不是完成核心诊断课题的必要产物。

## 结构化证据

自动结论来自正式运行中的 `generation_manifest.jsonl`、`performance.json`、`evaluation_v3/generation_validation.json`、`evaluation_v3/generated_identity_metrics.csv`、`evaluation_v3/metrics_by_condition_prompt.csv`、`evaluation_v3/paired_bootstrap.json`、`evaluation_v3/real_reference_baseline.csv`、`evaluation_v3/cohort_geometry.csv` 和 `analysis_v3/summary.json`。配置、asset lock、代码快照、Python package lock、环境/GPU 元数据和输入 manifest 均随运行保存。

原始/对齐人脸、生成图片、embeddings、模型权重和完整运行目录保持 Git-ignored。本报告只包含聚合统计，不读取、嵌入或发布任何人脸图像。
