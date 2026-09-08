# H1：人类感知身份相似度实验

> 状态：10 份表单评分与冻结分析已完成（探索性结果）
>
> 数据来源：`20260903T050000Z_photomaker_v2_generation_pilot`
>
> 研究范围：8 个 FEI 身份上的 PhotoMaker V2 探索性实验
>
> 已构建包：`artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot/human_eval_v2`（study `study_9804f75f76f29ffb62e4`）

## 1. 实验问题

本实验只回答一个问题：**给定同一个人的三张未参与生成的真实照片，哪一种身份条件生成的照片更像这个人？**

它不是 target 图像重建，也不评价哪张图更漂亮、更真实或更符合 prompt。两张候选图使用相同身份、prompt、seed、初始 latent、生成模型和采样参数，只改变待研究的身份条件。

## 2. 三个对照

| 编号 | 候选条件 | 固定的正方向 | 回答的问题 |
|---|---|---|---|
| P0 | `k4_diverse_full` vs `k1_full` | diverse K=4 | 四张不同参考相对单张参考是否提高 likeness？ |
| P1 | `k4_diverse_full` vs `k4_repeat_full` | diverse K=4 | 在参考数量相同的情况下，新增的不同视角信息是否优于重复同一张图？ |
| P3 | `global_target_patch_donor` vs `global_donor_patch_target` | global=target | 人的身份判断在冲突输出中更倾向跟随 global embedding 还是 patch 图像通道？ |

不加入 P2（重复 K=4 对 K=1）、P4（K=1 对 text-only）、4-AFC、画质题或 prompt 题。P0 是实际总效果，P1 隔离“不同参考带来的新信息”，P3 定位两个身份通道的相对影响；其余题目不直接增加对当前问题的解释力。

P3 是人为制造的通道冲突，只能解释机制，不能说明正常输入时某个通道单独就足够。

## 3. 实验材料

- 8 个固定 FEI 身份；
- 2 个固定 prompts；
- 2 个固定 generation seeds；
- 每个对照有 `8 × 2 × 2 = 32` 个唯一 pair；
- 三个对照合计 96 个唯一 pair；
- 每个身份展示同一组 3 张 held-out 真实照片：正面微笑、左侧四分之三和右侧四分之三表情照；
- reference 使用 640×480 原图，候选项使用冻结的 1024×1024 生成图；不重新生成、不筛图。

Reference panel 中的照片没有送入生成器。参与者看不到 identity、condition、原文件名、donor 或模型字段，公开媒体使用内容散列后的匿名文件名。

## 4. 分组与左右平衡

96 个 pair 分成 A、B 两个互补内容块，每块 48 题。P0/P3 使用 `prompt index XOR seed index` 分块，P1 使用相反分块。这样同一张 `k4_diverse` 图虽然分别参与 P0 和 P1，却不会被同一参与者重复看到。每个内容块制作 5 个独立展示版本，因此共 10 份表单：`A01–A05` 和 `B01–B05`。

每份表单恰好包含：

- 48 题；
- P0、P1、P3 各 16 题；
- 每个身份 6 题；
- 每个 prompt 24 题；
- 每个 seed 24 题；
- 正方向在候选左/右位置各 24 次，并且每个对照内部各 8 次在左、8 次在右。
- 96 个候选图位置对应 96 张不同的生成图，同一参与者不会重复看到同一候选图。

同一个唯一 pair 出现在同一内容块的 5 份表单中，左右方向按展示版本交替，所以最终得到 5 个判断，并形成 3:2 或 2:3 的左右分配。每份表单由一名不同参与者完成，计划总量为 `10 × 48 = 480` 个判断。

## 5. 参与者任务

页面始终使用同一个问题：

> Which generated face better preserves the reference identity?

参与者选择：

- A；
- B；
- Tie / equally similar；
- Unjudgeable。

说明文字要求只判断与 reference panel 的身份相似度，不以图像质量、风格、姿态、光照或背景作为答案。每位参与者填写唯一 participant ID，并完成分配给自己的 48 题。

## 6. 分析定义

每个对照的 focal 条件在构包时固定：P0/P1 为 `k4_diverse_full`，P3 为 `global_target_patch_donor`。解盲时把选择转成：

- focal 胜：1；
- 平票：0.5；
- focal 负：0；
- unjudgeable：不进入相似度分数，但单独报告数量和比例。

聚合顺序固定为：

1. 对每个唯一 pair 汇总 5 个评分；
2. 对每个 `identity × contrast` 平均四个 `prompt × seed` pair；
3. 对 8 个身份等权平均，得到该 contrast 的 pairwise preference score；
4. 以 identity 为重采样单位计算 95% bootstrap 区间。

同时报告 focal 胜、平、负、unjudgeable 的原始计数，避免只给一个平均值。中性点是 0.5：高于 0.5 表示人评更支持预先固定的 focal 条件，低于 0.5 表示更支持 comparator。

## 7. 结果边界

正式分析位于 `human_eval_v2/analysis/analysis.json`。10 名不同参与者完成了全部表单，共 480 次判断；每个唯一 pair 有 5 个评分。原始 participant code 有重复，但收集者确认对应不同的人，因此分析副本使用 form-derived 唯一匿名 ID，原始 response 保持不变，映射保存在 `analysis_inputs_unique_ids/normalization_manifest.csv`。

| 对照 | identity 等权偏好分数 | 95% identity-bootstrap CI | focal / tie / comparator | 无法判断 |
|---|---:|---:|---:|---:|
| P0：diverse K=4 vs K=1 | 0.4797 | [0.3870, 0.5708] | 35 / 56 / 36 | 33/160 |
| P1：diverse K=4 vs repeated K=4 | 0.5057 | [0.4388, 0.5826] | 37 / 54 / 36 | 33/160 |
| P3：global-target vs patch-target | 0.3839 | [0.1391, 0.6661] | 35 / 3 / 53 | 69/160 |

三个区间都覆盖中性点 0.5。P0/P1 不支持当前参考多样性带来可检测的人类 likeness 改善。P3 的可判断票更常选择 `global_donor_patch_target`，与自动识别器主要跟随 global 的结果方向不同；但 43.1% 无法判断、区间很宽且身份方向高度异质，因此只能作为“人类可能使用自动识别器未充分反映的 patch/local cues”的后续假设，不能断言 patch 通道普遍主导。

这轮结果能够说明，在当前 8 个 FEI 身份、两个 prompts 和 PhotoMaker V2 配置下，人类是否更偏向 diverse K=4，以及通道冲突时身份更跟随 global 还是 patch 来源。

它不能证明结果可以推广到自然自拍、其他人群、其他生成模型，也不能证明 ViT 的某一层删除身份或模型在生成“平均脸”。8 个身份的区间主要反映这个 pilot 内身份之间的变化，因此结果应报告为探索性证据。

## 8. 执行

当前正式包已经构建，无需再次运行。要在另一个同结构的完整生成 run 上重建：

```powershell
$run = "artifacts\runs\RUN_ID"
.\.venv\Scripts\python.exe scripts\build_grouped_human_eval.py `
  --run-dir $run `
  --manifest data\manifests\fei_full_seed20260902.csv `
  --output "$run\human_eval_v2" `
  --seed 20260904
```

将 `participant/` 中不同的 HTML 表单分别分配给 10 名参与者；分享时保留同目录 `media/` 的相对路径。收回的 JSON 放入 `human_eval_v2/responses/`。评分完成后运行：

```powershell
$responses = (Get-ChildItem "$run\human_eval_v2\responses\*.json").FullName
.\.venv\Scripts\python.exe scripts\analyze_grouped_human_eval.py `
  --answer-key "$run\human_eval_v2\private\answer_key.json" `
  --responses $responses `
  --output "$run\human_eval_v2\analysis\analysis.json"
```

本轮已经完成，不要在相同冻结输出上继续增加评分者或调整分析规则。后续确认应使用新的身份 cohort 和预先冻结的实验版本。
