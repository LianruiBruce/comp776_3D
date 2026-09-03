# PhotoMaker V2 generation identity-loss diagnostic pilot

## Status and scope

This document freezes the design of a small, output-level diagnostic using the official PhotoMaker V2 pipeline. It is an **exploratory engineering pilot**, not a confirmatory generation experiment and not a new test of the E2 layer hypothesis.

E2 did not pass its preregistered matched-readout gate: block 11 did not show a reliable internal advantage over the capacity-matched block-12 readout. Therefore this pilot uses only PhotoMaker V2's native InsightFace-plus-CLIP identity path. It does not inject the LVFace block-11 representation, train a mapper, reopen layer selection, or treat a generated-image result as evidence that the final ViT layer erased identity.

The machine-readable companion is `configs/generation_identity_pilot.yaml`. Any discrepancy is an error; the configuration fixes the executable values and this document fixes their interpretation.

The engineering basis is the [official PhotoMaker V2 identity encoder](https://github.com/TencentARC/PhotoMaker/blob/main/photomaker/model_v2.py) and [official pipeline](https://github.com/TencentARC/PhotoMaker/blob/main/photomaker/pipeline.py). The peer-reviewed [PhotoMaker V1 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.pdf) motivates testing stacked references, but its ablations must not be attributed to V2. This pilot tests V2's released implementation directly.

## Questions

The pilot asks six narrow questions:

1. Does one native full-image reference steer a generated face toward the supplied identity?
2. At a fixed reference count, does condition diversity add useful identity information beyond repeating one image?
3. When PhotoMaker V2's global 512-D face embedding and CLIP patch-image source name different people, which source does the output tend to follow?
4. Does identity steering survive a modest pose, expression, lighting and background prompt change?
5. Do conclusions agree between an evaluator independent of the PhotoMaker conditioner, the reused InsightFace recognizer, and blinded humans?
6. Do generated identity centroids become less separated across people, consistent with cohort-level identity contraction?

These questions localize symptoms only. Without internal activation interventions, output comparisons cannot identify the exact QFormer, fusion, U-Net block or denoising step at which information was lost.

## Frozen cohort

The eight identities are frozen before inspecting any generated output:

```text
fei_0058
fei_0075
fei_0092
fei_0100
fei_0115
fei_0162
fei_0168
fei_0193
```

They belong to the existing FEI `internal_eval` split. That name does not make them fresh test identities: every FEI identity has already been touched during project development. Results are consequently exploratory and cannot serve as the external confirmation required by the project gate.

At design time, all 56 required identity-condition rows below were present and marked `aligned` in `data/manifests/fei_full_seed20260902.csv`. The manifest's `source_relative_path` and `relative_path` values are resolved against `data/`, not the repository root. Generation consumes the original full images in `source_relative_path`; the previous 112x112 alignment is used only for real-query evaluation. A new failure in PhotoMaker/InsightFace preprocessing remains a failure and must not be hidden by the earlier manifest status.

### Reference and query separation

Reference images and evaluation queries are disjoint by acquisition condition.

| Role | Frozen FEI conditions | Use |
|---|---|---|
| Single reference | `frontal_neutral` | K=1 input and the repeated-reference control |
| Diverse references | `frontal_neutral`, `left_profile`, `right_profile`, `frontal_scale` | Native K=4 stack |
| Held-out real queries | `frontal_smile`, `left_three_quarter`, `right_three_quarter_expression` | Identity gallery/template only; never generator input |

No reference is selected using generated images, an evaluator score, or visual preference. No held-out query may be passed to PhotoMaker, used to choose a prompt, or used to replace a failed reference.

### Frozen donor map

Each target has exactly one donor, defined by the next identity in the frozen list with wraparound:

| Target | Donor |
|---|---|
| `fei_0058` | `fei_0075` |
| `fei_0075` | `fei_0092` |
| `fei_0092` | `fei_0100` |
| `fei_0100` | `fei_0115` |
| `fei_0115` | `fei_0162` |
| `fei_0162` | `fei_0168` |
| `fei_0168` | `fei_0193` |
| `fei_0193` | `fei_0058` |

This map is deterministic rather than a claim that donor pairs are perceptually or demographically matched. It must not be changed after looking at outputs.

## Frozen generation system

The asset lock already pins PhotoMaker code revision `060b4fcb10b76a4554edf565d6106b7e36c968f0`, PhotoMaker V2 weight revision `f5a1e5155dc02166253fa7e29d13519f5ba22eac` with SHA256 `0eeac57d7e09b95cee670a18e0d939ff303dd6f76d317c950bf584c6c9bdb8f5`, and RealVisXL V4.0 revision `26dfe44930964cd70d0a817b6d1cc945c130e38d`.

The primary execution fixes:

- official PhotoMaker V2 pipeline and native identity adapter;
- RealVisXL V4.0 FP16 base and Euler discrete scheduler;
- 1024x1024 output, batch size 1, 30 inference steps and CFG 5.0;
- PhotoMaker trigger phrase `person img`, with `img` immediately following its class word;
- `start_merge_step=10` for the five identity-input conditions;
- `start_merge_step=30` for the text-only condition;
- base seeds `776` and `1776`, converted to identity-specific effective seeds and paired across conditions and prompts only within an identity;
- model CPU offload and inference-only execution on the RTX 4080.

The official implementation uses text-only prompt embeddings while `i <= start_merge_step` and fused identity embeddings afterward. With 30 loop iterations indexed 0 through 29, a value of 30 therefore prevents fused identity embeddings from entering every denoising iteration. It is a PhotoMaker text-only control, not an unmodified base-SDXL control: the PhotoMaker-loaded pipeline, fused LoRA and base model remain the same.

Identity ordinals are zero-based positions in the frozen identity list. For base seed `b`, `effective_seed = b + identity_ordinal * 100000`. Thus `fei_0058` uses `776/1776`, `fei_0075` uses `100776/101776`, and so on through `fei_0193` at `700776/701776`. For one identity and base seed, the exact same initial latent is reused across all six conditions and both prompts. Different identities deliberately receive different latents; neither their latent hashes nor their text-only output hashes are expected to match.

## Frozen conditions

The condition names describe the provenance of the 512-D global face embedding and the pixels sent through PhotoMaker's CLIP vision path.

| Condition | Global `id_embeds` source | CLIP pixel source | References | Purpose |
|---|---|---|---:|---|
| `k1_full` | target | target `frontal_neutral` | 1 | Native single-reference positive condition |
| `k4_diverse_full` | target, condition matched | target four-condition set | 4 | Native diverse stack |
| `k4_repeat_full` | target `frontal_neutral`, repeated | target `frontal_neutral`, repeated | 4 | Equal-count no-new-information control |
| `global_target_patch_donor` | target `frontal_neutral` | donor `frontal_neutral` | 1 | Conflicting identity channels |
| `global_donor_patch_target` | donor `frontal_neutral` | target `frontal_neutral` | 1 | Reciprocal conflicting identity channels |
| `text_only` | target, computed but not used in denoising | target `frontal_neutral`, computed but not used | 1 | No denoising-time identity conditioning |

For K=4, the pixel list and external InsightFace embedding list must have identical order and length. `k4_repeat_full` repeats the same image and its same embedding exactly four times; it must not use four near-duplicate files.

### Causal interpretation boundary

The following contrasts are permitted:

- `k4_diverse_full - k4_repeat_full`: added reference-condition information at equal K;
- `k4_repeat_full - k1_full`: effect of repeating/stacking an identical signal, not diversity;
- the two reciprocal global/patch conflicts: relative output influence of the global and patch channels;
- `k1_full - text_only`: net effect of enabling the complete native identity-conditioning path;
- P1 versus P0 within a fixed condition and seed: robustness to this particular prompt nuisance change.

They do **not** separately identify the face encoder, QFormer, fusion module and U-Net injection effects. The global/patch conflicts are deliberately out-of-distribution combinations and diagnose sensitivity, not normal operating accuracy. P0 versus P1 is not an identity-scale dose response. This pilot contains no `id_scale` sweep, no `start_merge_step` sweep, no alternative generator and no learned intervention.

## Frozen prompts and factorial size

Both prompts avoid names and facial-feature descriptions. They differ only in requested scene nuisance and use the same fixed class/trigger phrase.

- **P0 / easy:** `a studio head-and-shoulders photograph of a person img, neutral expression, front view, even soft lighting, plain gray background`
- **P1 / nuisance:** `a candid head-and-shoulders photograph of a person img, three-quarter left view, smiling, warm side lighting, indoor cafe background, face unobstructed`

The shared negative prompt excludes multiple people, duplicate faces, cropped or malformed faces, text, watermarks and low resolution. It may not be changed per output.

The complete factorial is:

```text
8 identities x 6 conditions x 2 prompts x 2 seeds = 192 generated images
```

Every cell is generated and evaluated. There is no best-of-two selection, regeneration of unattractive samples, seed replacement, manual face choice or prompt rewriting.

Each run must snapshot the resolved configuration, asset lock, executable source, Python package lock, environment/GPU metadata and the 56-row input-manifest subset. It must also store a 192-row output manifest with input/output hashes and every factor, preprocessing/generation/evaluation status, aggregate and identity-level metrics, bootstrap draws or their reproducible summary, performance records, log and completion status. Generated images remain inside the Git-ignored run directory. A completed run is immutable; a fallback or correction creates a new run.

## Automated evaluation

### Evaluator separation

The current primary automatic evaluator is the pinned **official final embedding of LVFace-T Glint360K**. This is method-specifically independent for the native PhotoMaker V2 pilot: PhotoMaker uses InsightFace and CLIP features, not LVFace, for conditioning, and this pilot uses no LVFace-based generation loss. The evaluator is the official final LVFace output, never block 11, an E2 probe or a selected intermediate representation.

The existing InsightFace `w600k_r50.onnx` embedding may be reported only as a **conditioner-aligned secondary diagnostic**. Agreement with that reused evaluator alone is not evidence of identity preservation. AdaFace and/or CurricularFace remain required sensitivity evaluators before a formal generation claim, but their absence does not block this exploratory pilot. Conversely, if the pinned LVFace-T evaluator is unavailable, the run fails rather than silently promoting InsightFace to primary.

This independence statement is limited to the native PhotoMaker run. LVFace-T cannot remain the sole primary evaluator for a later generator or mapper conditioned on LVFace features; that future experiment requires AdaFace/CurricularFace and human confirmation.

For each evaluator, the three held-out query embeddings for identity `i` are L2-normalized, averaged, and normalized again to form real template `q_i`. For a generated face embedding `g`, the principal continuous score is the target-versus-closest-impostor margin:

```text
margin(g, i) = cosine(g, q_i) - max_{j != i} cosine(g, q_j)
```

Primary automated outcomes are:

1. detection-aware target margin and eight-way Rank-1 against the real query gallery;
2. target and donor similarities for both reciprocal conflict conditions;
3. global-follow, patch-follow and neither/unresolved rates based on which real template is closer;
4. paired P1-minus-P0 change in target margin within identity, condition and seed;
5. paired `k4_diverse_full - k4_repeat_full`, `k4_repeat_full - k1_full`, and `k1_full - text_only` effects;
6. generated-face detection coverage, multiple-face rate and evaluator alignment coverage.

For each identity and base seed, the initial latent tensor hash must be identical across all six conditions and both prompts. The identity-specific effective-seed schedule deliberately prevents a cross-identity latent or output-hash invariance requirement. The text-only control is interpreted through paired within-identity contrasts, not by expecting the same generated pixels for different identities.

Absolute verification thresholds are not primary in this small pilot. A TAR/FAR result may be added only if its threshold was frozen on a separate real-only calibration cohort before any generated image was scored. A threshold imported from an unrelated benchmark or calibrated on these eight identities is invalid.

No-face, multiple-face and alignment failures remain in the 192-row output manifest. They count as failures in end-to-end Rank-1/follow rates; successfully aligned-only metrics are reported separately. The runner must never silently choose the most target-like face from a multi-face output.

Prompt adherence, generated-image quality, seed diversity and maximum similarity to any input reference are secondary outcomes. CLIP-based prompt scores are also potentially model-aligned and must be accompanied by the blinded prompt judgment below.

### Cohort-level mean-face contraction

“Mean-face contraction” is operationalized as loss of **between-identity geometry**, not as a visual impression and not as a literal pixel average. For each normal target condition, prompt and seed aggregation rule, let `r_i` be identity `i`'s normalized held-out real-query template and `g_i` its normalized generated centroid. Define:

```text
B(X) = mean over i<j of (1 - cosine(x_i, x_j))
contraction_ratio = B(G) / B(R)
```

Also report mean impostor cosine shift, target-impostor margin, eight-way Rank-1, attraction toward the normalized real-cohort centroid, and the effective rank of centered identity centroids. The contraction ratio is primary for this diagnostic; effective rank is descriptive because eight identities permit rank at most seven.

Contraction is estimated principally for `k1_full`, `k4_repeat_full` and `k4_diverse_full`. The conflicting and text-only conditions are negative diagnostics, not evidence about normal multi-reference behavior. A ratio below one accompanied by higher impostor similarity and worse retrieval is consistent with cohort-level contraction. It does not establish that pretraining learned a population mean or that a particular ViT layer caused the contraction.

### Uncertainty and reporting unit

The identity is the inferential unit. First aggregate paired prompt/seed observations within identity, then perform 1,000 paired identity-cluster bootstrap resamples with seed `20260903`. Report raw identity-level values, effect sizes and percentile 95% intervals. With only eight already-exposed FEI identities, intervals are descriptive and must not be promoted to a population-level significance claim.

## Blinded human evaluation

Human evaluation is required before making even an exploratory statement about perceptual identity. It must be approved or exempted under the institution's human-subjects process before showing identifiable faces to raters.

### Identity task

- Show the generated face crop without a condition/method label.
- Show four randomized held-out real-query galleries: target, its frozen donor and two deterministically selected non-donor distractors.
- The two non-donor distractors are the identities at offsets `+2` and `+3` from the target in the frozen cyclic identity list.
- Each gallery uses the three query images, not the generator inputs.
- Ask which gallery depicts the same person, with `none` and `unjudgeable` options.
- Evaluate all 192 generated outputs; no image-quality or detector-based subset is selected for humans.
- Obtain at least five independent ratings per evaluated output.

### Prompt task

- Show the full generated image and its prompt on a separate screen from the identity task.
- Ask whether viewpoint, expression, lighting/background and the single-person requirement are satisfied.
- Do not show reference images on this screen.
- Evaluate all 192 outputs, retaining `unjudgeable` rather than dropping malformed or faceless samples.

### Paired diagnostic task

For the same identity, prompt and seed, compare randomized left/right pairs for:

- `k4_diverse_full` versus `k4_repeat_full`;
- `k4_repeat_full` versus `k1_full`;
- the two reciprocal global/patch conflict conditions;
- `k1_full` versus `text_only`.

Ask identity likeness, prompt adherence and visual quality as separate questions, each allowing a tie/unjudgeable response. Never ask one vague “which is better?” question. Randomize item order and side, hide file names and conditions, limit each rater to a preregistered incomplete block, include approximately 10% attention checks, and freeze exclusion rules before collection. Report win/tie/loss, four-way accuracy, `none`/unjudgeable rates and inter-rater agreement with identity-clustered intervals. Ratings by a depicted person, if collected, are a separate exploratory stratum and are never pooled with unfamiliar raters.

The four paired contrasts produce `4 x 8 identities x 2 prompts x 2 seeds = 128` pair items. All 128 are evaluated; pairs are not chosen from the automated scores.

## Ethics, privacy and licenses

- FEI images and PhotoMaker's InsightFace-dependent weights are research-only assets. Raw images, generated images, face crops and embeddings remain local and Git-ignored.
- Do not publish identifiable examples without checking the dataset terms and obtaining any additional consent/approval required by the institution.
- Do not infer race, ethnicity, gender identity, health, attractiveness or other sensitive traits from the eight people.
- The small fixed sample cannot support demographic fairness claims.
- Generated files must retain provenance linking identity and zero-based ordinal, source sample IDs, donor, prompt, condition, base/effective seed, initial-latent hash, code/weight hashes and failure state; public reports should use aggregates unless image publication is explicitly authorized.

## Preregistered failure and hardware fallbacks

Before the factorial run, execute a technical sentinel that covers both prompts, a normal identity condition, a conflict condition and text-only. It validates model loading, tensor/list shapes, trigger-token placement, deterministic seed replay, output provenance and peak CUDA allocation. Sentinel images are not substituted for factorial cells.

The only resolution fallback is:

1. attempt the primary 1024x1024, FP16, batch-one configuration with model CPU offload;
2. if it produces a reproducible CUDA out-of-memory error after clearing unrelated processes and restarting the worker, record the traceback and peak memory;
3. change both dimensions to 768, create a new run ID and regenerate **all 192 cells from the beginning**;
4. never pool 1024 and 768 outputs or choose the resolution from identity scores.

Steps, sampler, CFG, prompts, seed formula, condition definitions, references, identities and donor mapping do not change under the fallback. A missing pinned LVFace-T primary evaluator fails the run rather than authorizing substitution with InsightFace. Missing AdaFace/CurricularFace sensitivity models do not block this exploratory pilot, but do block a later formal generation claim. Reference detection failure, generation exception, no-face output and evaluator failure are retained with explicit status and denominator; they do not authorize a new seed or reference.

## Permitted and prohibited conclusions

The pilot may report, with effect sizes and uncertainty:

- whether native PhotoMaker identity conditioning causally changes outputs relative to its text-only mode for these eight identities;
- whether the output follows the global or patch donor more often under the reciprocal conflict intervention;
- whether diverse K=4 references differ from repeated K=4 references;
- whether P1 degrades identity steering relative to P0;
- whether independent, reused and human evaluators agree;
- whether generated identity geometry shows cohort-level contraction.

It must **not** claim:

- external confirmation, generalization beyond these eight FEI identities, or state-of-the-art performance;
- that block 11 is an identity layer, that the final ViT layer erased identity, or that E2 became positive;
- that a specific adapter/UNet component caused a failure without an internal intervention;
- literal “mean-face generation” or a pretraining population-mean mechanism from embedding contraction alone;
- demographic fairness or absence of bias;
- successful identity preservation from the reused ArcFace-family score alone;
- a generation improvement obtained by selecting attractive seeds or dropping failures;
- authorization to train a custom mapper. That remains gated on frozen external representation and multi-reference evidence with a separately licensed training dataset.

## Post-run implementation audit (2026-09-03)

The 192-image automatic protocol above is complete. The local `human_eval/participant/` artifact is only a v1 **identity-likeness master page** used to validate blinded media staging and response decoding. It contains all 192 4-AFC items and 160 pairwise items, but it is not yet the formal collection instrument: each v1 response must cover all 352 items, and the page does not implement the preregistered incomplete blocks, attention checks, or separate prompt-adherence and visual-quality screens. Do not distribute it for formal recruitment until those gaps are implemented and the applicable institutional approval or exemption is in place.

The v1 master also includes 32 direct `k1_full` versus `k4_diverse_full` pairs (code `P0`) in addition to the 128 frozen pair items. This is a useful direct diagnostic but was not one of the four pairwise contrasts enumerated above; it must remain explicitly exploratory and must not silently increase the preregistered confirmatory family.
