# Experiment Design V1 — Layer-wise and Multi-reference Face Identity

## 1. Goal

This experiment asks a narrower and falsifiable question before modifying any generator:

> Under pose, expression and illumination changes, does an intermediate face-ViT representation or a multi-image aggregation rule verify unseen identities better than a matched final-layer readout and the official final embedding?

The experiment is successful even if the answer is negative. A high same-person cosine is not enough: the representation must also reject different identities, generalize to identities excluded from model selection, and remain useful under nuisance changes.

The primary layer comparison is an intermediate matched E2 readout versus the same readout at block 12. The official final embedding is reported separately as an operational baseline because its topology, training data and capacity are different.

## 2. What related work implies

The middle column reports facts from the cited upstream work. The right column records project-specific decisions unless it explicitly says that an upstream protocol is reproduced.

| Work | Relevant implementation detail | Consequence for this project |
|---|---|---|
| [LVFace, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.html) / [official code](https://github.com/bytedance/LVFace) | The official inference architecture is a pre-LN face ViT without a CLS token; after the final norm its head flattens all spatial tokens. Full-model training reports CosFace `s=64`, `m=0.4`, AdamW `lr=1e-3`, weight decay `0.1`, polynomial decay, 120 epochs, horizontal flips, PCO/NCS and a progressive `384→128` batch schedule. | Use LVFace-T for fast development. E2 borrows the hyperspherical geometry and selected numerical defaults, but is not a reproduction of the official full-backbone training recipe. |
| [CVLFace ViT-KP-RPE](https://github.com/mk-minchul/CVLface/blob/main/cvlface/research/recognition/code/run_v1/models/vit_kprpe/__init__.py) / [KP-RPE paper](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.html) | The base model has depth 24, 112×112 inputs, patch size 8 and 512-dimensional tokens. KP-RPE is designed to improve robustness to alignment and affine changes using five landmarks. | Use it as the stronger face-ViT replication, not as the first debugging model. |
| [TransFace](https://github.com/DanJun6737/TransFace) | Its DPAP/EHSM design explicitly uses patch variation and local-token information; the official repository provides aligned 112×112 inference and pretrained ViTs. | Local tokens may contain useful information even when a global pooled vector does not; keep token-level probes as an ablation. |
| [DINOv2](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) | The official model exposes `get_intermediate_layers`, normalized class tokens and patch tokens. | Use DINOv2-S/14 only as a generic-vision control. A face-specific effect should replicate more strongly in face ViTs. |
| [PhotoMaker v1, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.pdf) | The peer-reviewed stacking ablation reports better ID fidelity than averaging, with diminishing gains and an identity–text controllability trade-off as reference count grows. | Use this as evidence for testing stacked references and reporting the trade-off; do not attribute this ablation to V2. |
| [PhotoMaker V2 official implementation](https://github.com/TencentARC/PhotoMaker/blob/main/photomaker/model_v2.py) | The implementation accepts multiple ID images and combines a 512-D face embedding with CLIP patch features through a Perceiver-style module. | Use V2 as an engineering baseline for the later generator experiment, not as the source of the V1 peer-reviewed ablation. |
| [PuLID v1.1](https://github.com/ToTheBeginning/PuLID/blob/main/pulid/pipeline_v1_1.py) / [paper](https://arxiv.org/abs/2404.16022) | The code concatenates an InsightFace `antelopev2` embedding with an EVA-CLIP global feature, retains multiple ViT hidden layers, and concatenates hidden tokens from auxiliary reference images before the ID adapter. | This is direct precedent for combining a recognition embedding, intermediate ViT features and multiple references. It also means ArcFace is not an independent evaluator for PuLID. |
| [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) / [InsightFace alignment](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py) | ArcFace uses normalized angular embeddings; InsightFace applies a fixed five-landmark similarity transform to 112×112 before recognition. | Use one fixed five-point alignment pipeline. ArcFace may evaluate a method only when ArcFace features were not used by its conditioner or loss. |
| [Intermediate Layer Classifiers for OOD Generalization, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/71c3451f6cd6a4f82bb822db25cea4fd-Abstract-Conference.html) / [official code](https://github.com/oshapio/intermediate-layer-generalization) | Freezes a backbone, fits layer-wise affine closed-set classifiers, selects with validation data and repeats training. It studies image-classification distribution shifts. | E2 borrows only the frozen-backbone, validation-only selection and repeated-run principles. Its open-set metric readout, CosFace training and identity-wise FAR cross-fitting are our face-recognition adaptation, not an ILC reproduction. |
| [Prisma, 2025 arXiv whitepaper; CVPR MIV Workshop oral/tutorial](https://arxiv.org/abs/2504.19475) / [official code](https://github.com/Prisma-Multimodal/ViT-Prisma) | Provides activation caching, circuit tools and intervention examples for supported vision-transformer families. | This is tooling precedent only. LVFace hooks are custom project code; causal claims would require a separately controlled activation-patching protocol. Attention maps alone are not causal evidence. |
| [Layer by layer, module by module: Choose both for optimal OOD probing of ViT, ICLR 2026 CAO Workshop](https://openreview.net/forum?id=4lT3aScsRJ) / [official code](https://github.com/ambroiseodt/vit-probing) | Reports that the best image-classification OOD probe site can depend on both depth and module output. This is a workshop result, not an ICLR main-conference conclusion or face-verification study. | Keep full-block outputs as the preregistered E2 primary site; treat attention/MLP hook-site sweeps as later exploratory work with a new external confirmation. It does not establish identity storage or causal use. |

## 3. Hypotheses

- **H1 — depth effect:** identity verification changes materially with ViT depth under nuisance variation.
- **H2 — matched-readout accessibility:** with identical frozen-backbone metric readouts trained on the same train identities, an intermediate block transfers to unseen-identity verification better than block 12. This supports accessibility under this supervised readout family, not storage, erasure or causal use.
- **H3 — multiple references:** four diverse, quality-controlled references outperform a single reference and simple final-layer averaging.
- **H4 — external replication:** the selected representation and aggregation rule retain their direction of improvement on a second dataset and a second face ViT.
- **H5 — generation gate:** only a representation that passes H3 and H4 should be considered for generator conditioning.

## 4. Datasets and split policy

### 4.1 Development: FEI original images

Use the [official FEI original set](https://fei.edu.br/~cet/facedatabase.html): 200 identities × 14 color images, with expression changes, approximately 10% scale variation and profile rotation up to about 180 degrees.

- Research-only; never commit or redistribute images.
- Create a new immutable manifest containing identity, acquisition slot, expression if known, detection result, five landmarks, derived yaw/pitch, brightness, blur, alignment status and SHA256.
- Split identities once with seed `20260902`: 100 train / 50 dev / 50 internal evaluation.
- The earlier aligned-FEI smoke exposed all identity groups. Therefore the FEI internal evaluation is a development check, not a fresh confirmatory test.

The 14 acquisition slots are consistent across identities. Before running any model comparison, the protocol freezes slots `01, 03, 08, 10, 12, 14` as queries and `02, 04, 05, 06, 07, 09, 11, 13` as the reference pool. This covers frontal, left/right profile, expression and illumination changes. The nuisance names are visually derived protocol labels rather than publisher annotations and are marked as such in the manifest.

### 4.2 External confirmation: Extended Yale Face Database B+

Use the [official UCSD Extended Yale B+ description](https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b): 28 subjects, nine poses and 64 illumination conditions. Use the cropped or officially supplied alignment metadata variant and record the exact archive/version in the asset lock.

- Replicate grayscale into three channels only at model input.
- Do not tune a layer, threshold, pooling rule or reference policy on Yale B+.
- Run only the configuration frozen from FEI dev.
- Report results by pose and illumination severity as well as the aggregate.

## 5. Preprocessing

1. Detect one face and five landmarks using one pinned InsightFace detector.
2. Align to the ArcFace 112×112 template with the official similarity transform.
3. Save detector score, selected box, landmarks, transform matrix and failure reason in the manifest.
4. Never silently remove a failure. Report both conditional metrics on successfully aligned faces and end-to-end detection/alignment coverage.
5. Feed the same aligned pixels to each face-recognition backbone, then apply that backbone's documented normalization.
6. Do not choose reference images using test identity similarity or the final evaluator.

## 6. Stage A — Layer profiling

Freeze every backbone. For each Transformer block `l`, extract:

1. **Official final embedding** — the main baseline, available only for the last block.
2. **Token mean** — post-block tokens, the trained final `model.norm`, mean pooling and L2 normalization; applying that final norm to an intermediate block is an off-distribution E1 diagnostic.
3. **Final-head probe** — apply the official final head to every block; keep only as a diagnostic because the head was trained for the last block.
4. **Equal-capacity supervised metric readout** — raw post-block tokens, a fresh per-layer `LayerNorm → token mean → Linear(d, 512, bias=False) → L2`, and an identical training-only CosFace classifier for every layer, including the final layer.

The E2 readout is not called a strict linear probe because its purpose is to learn a 512-D open-set angular embedding, rather than only a closed-set affine classifier. “Equal capacity” refers only to the 12 E2 readouts: it does not mean capacity-matched to the official LVFace head, which preserves token positions and is much larger. For an efficient but exact implementation, cache the mean of non-affine, per-token LayerNorm outputs, standardized independently over each token's channel dimension with the same epsilon. The fresh trainable LayerNorm affine is then applied as `gamma * cached_mean + beta` before the projection; this is algebraically identical to applying the affine to every token before their mean. This cache factorization is a project engineering choice.

The following recipe is a project-preregistered diagnostic, not the official LVFace training procedure and not the ILC protocol:

For the equal-capacity readout:

- keep the ViT frozen;
- never reuse the final trained `model.norm` affine for an intermediate block;
- train the fresh LayerNorm affine, projection and classifier only on all usable images of the 100 FEI train identities;
- use CosFace `s=64, m=0.4`, AdamW `lr=1e-3`, betas `(0.9, 0.999)`, weight decay `0.1`, polynomial decay with project-chosen power `1.0` to zero, batch size 128, float32 and exactly 120 epochs; the scalar defaults are borrowed from LVFace where available, but fixed batch size, no augmentation and the decay power are project choices;
- save every layer/seed training loss and accuracy curve, since a common budget does not guarantee equal optimization difficulty;
- use seeds `{776, 1776, 2776, 3776, 4776}` and reset initialization and minibatch order so a given seed is matched across all layers;
- do not use augmentation, hyperparameter search or per-layer early stopping in the formal E2 run;
- discard the identity classifier and evaluate the 512-D projection on unseen identities.

Split the 50 dev identities into five fixed identity folds shared by all layers and seeds. For each held-out fold, calibrate the FAR threshold on the other four folds and score the held-out identities. Select exactly one common layer using mean cross-fitted template-to-query `TAR@FAR=0.01` across all five probe seeds. Ties are resolved by mean template d-prime, then mean ROC-AUC, then the shallower layer. No seed-specific winner is used. After the layer is locked, calibrate one threshold per seed on all dev identities and run internal evaluation only for the selected layer, the equal-capacity final layer and the official final embedding.

E2 includes three diagnostics that cannot select the layer: matched untrained projections, identity labels independently permuted among samples within each acquisition condition, and sample-independent Gaussian features passed through the same scoring path. Reusing one common identity permutation across conditions would merely rename classes and is not a valid control.

## 7. Stage B — Multi-reference identity templates

Keep the six query images fixed. From the eight reference candidates, evaluate nested `K ∈ {1, 2, 4, 8}` reference sets.

Compare these aggregation rules:

| ID | Template construction |
|---|---|
| M0 | One official-final reference embedding |
| M1 | L2-normalized mean of `K` official-final embeddings |
| M2 | L2-normalized mean of `K` embeddings from the dev-selected layer |
| M3 | Quality/diversity-weighted mean from the selected layer; weights use detection, blur and pose metadata only |
| M4 | PhotoMaker-inspired stacked set aggregator: retain all `K` image embeddings and map them to one 512-D template with a small two-layer attention/resampler |
| M5 | Parameter-matched MLP baseline for M4, so gains cannot be attributed only to more trainable parameters |

Train M3–M5 on train identities only. Freeze all decisions on dev. Random reference selection uses five pre-recorded seeds; quality/diversity selection is deterministic. Report the marginal gain from `K=1→2`, `2→4` and `4→8`, because related work suggests diminishing returns.

For every query, compare it with its own identity template and every other identity template. This is a template-to-image verification/retrieval problem, not leave-one-out image classification.

## 8. Metrics and statistical protocol

Required metrics:

- detection/alignment coverage and failure counts;
- end-to-end TAR with a failed query detection counted as a false reject, in addition to aligned-only recognition metrics;
- positive/negative cosine mean, standard deviation and gap;
- d-prime, ROC-AUC and EER;
- thresholds calibrated on FEI dev negatives, then frozen for FEI internal evaluation and Yale B+; report TAR and observed FAR at target FAR 0.01;
- template-to-query Rank-1 and Recall@5; E2's completed artifact serialized Rank-1 but omitted an explicit Recall@5 field, although Rank-1=1 entails Recall@5=1 for its frozen main methods; the reusable metric code now serializes Recall@5 for E3 and later runs;
- the same metrics stratified by yaw/pose, illumination and expression;
- throughput and peak CUDA memory.

Statistics:

- use 1,000 identity-level paired bootstrap resamples for 95% confidence intervals;
- never treat pairwise comparisons as independent samples;
- report mean and standard deviation across five probe/aggregator seeds;
- preregister two multi-reference contrasts before seeing internal evaluation: H3a compares the same aggregation rule at `K=4` versus `K=1`; H3b compares `M4` versus `M1` at `K=4`;
- external Yale B+ receives only the frozen winner and baselines M0/M1.

For E2, first average each identity's selected-minus-equal-capacity-block-12 TAR difference over the five probe seeds, then bootstrap identities; detection/alignment failures count as false rejects in this comparison. The official final embedding is a separate operational baseline and is not used for the paired H2 claim. FEI internal support requires a selected layer below 12, a positive difference whose identity-level 95% interval excludes zero, and a positive direction in at least four of five seeds. This remains internal-development evidence. Support for H1 or H3 requires the corresponding identity-level interval to exclude zero on frozen external confirmation. Otherwise report a null result; do not search additional test layers.

## 9. Backbone order on one RTX 4080

1. **LVFace-T** — implement and debug all metrics and probes.
2. **CVLFace ViT-Base KP-RPE / AdaFace / WebFace4M** — 24-layer face-specific replication. Its official configuration uses 112×112 inputs, patch size 8, depth 24, 512-D tokens and 16 heads.
3. **DINOv2-S/14** — optional generic control using the official `get_intermediate_layers` API.
4. **TransFace-S** — optional face-ViT replication if CVLFace integration is blocked.

All of these are inference or small-probe workloads; do not train a face backbone from scratch on the 4080.

## 10. Stage C — Minimal generation experiment

Generation begins only if the selected multi-reference representation improves the frozen external evaluation.

### C1. No-training reference ablation

Run official PhotoMaker V2 first because it natively accepts an arbitrary number of references. Compare `K={1,2,4}` and the random/quality/diversity reference policies using the same prompts and seeds. Run PuLID v1.1 as a second baseline if it fits the environment.

Use 12 prompts covering viewpoint, expression, lighting, accessories, background and non-photorealistic style, with four fixed seeds per identity/method. Keep resolution, sampler, steps, CFG and identity scale fixed.

### C2. Proposed adapter, only after C1

Freeze the generator and face ViT. Train only a small mapper that converts M4's selected-layer set representation into the same number and dimension of identity tokens expected by the chosen adapter. This requires a separate, appropriately licensed identity-training dataset; FEI alone is too small for a generator-conditioning claim.

Generation metrics:

- primary identity similarity from a recognizer not used by the conditioner or its loss;
- face detection coverage;
- prompt similarity (CLIP-T or equivalent);
- generated-face diversity across seeds;
- maximum similarity to any input image as a copying indicator;
- blinded pairwise human identity/preference judgments for the final comparison.

PhotoMaker V2 and PuLID both consume InsightFace-style identity embeddings in their official implementations. Therefore ArcFace similarity may be reported as a secondary diagnostic for those baselines, but an AdaFace/CurricularFace checkpoint must be the primary independent evaluator. ArcFace can be primary for a proposed LVFace-only conditioner that never uses ArcFace during training or inference.

## 11. Minimal run matrix

| Run | Purpose | Output decision |
|---|---|---|
| E0 | Existing FEI-aligned LVFace-T smoke | Pipeline gate; already complete |
| E1 | FEI-full LVFace-T raw layer sweep | Candidate layer/representation |
| E2 | FEI-full LVFace-T equal-capacity metric readouts | Test matched-readout accessibility at intermediate versus final blocks |
| E3 | FEI-full multi-reference `K × aggregator` | Freeze one aggregation rule |
| E4 | Yale B+ frozen external evaluation | Go/no-go for the core representation claim |
| E5 | CVLFace ViT-B frozen replication | Backbone generalization |
| E6 | PhotoMaker V2 reference-selection ablation | Go/no-go for a custom generator mapper |

Recommended proposed configuration names:

```text
configs/fei_full_lvface_layers.yaml
configs/fei_full_lvface_probes.yaml
configs/fei_full_lvface_multiref.yaml
configs/yale_bplus_frozen_eval.yaml
configs/cvlface_vitb_frozen_replication.yaml
configs/photomaker_v2_reference_ablation.yaml
```

Each run should additionally save `preprocess_coverage.csv`, `metrics_by_condition.csv`, a structured bootstrap artifact (`bootstrap_intervals.csv` for E1 or `selected_vs_final_bootstrap.json` for E2), and the already established config/source/environment snapshots.

## 12. Main validity threats

- FEI identities may overlap unknown face-recognition pretraining data; Yale B+ reduces but does not prove absence of overlap.
- FEI train and evaluation identities use the same acquisition slots; E2 tests identity-disjoint transfer under observed nuisances, not held-out-nuisance OOD generalization in the ILC sense.
- Applying the final LVFace head to intermediate blocks biases the comparison; the equal-capacity probe is mandatory.
- Even the E2 mean-pooled readout discards spatial token arrangement. A positive intermediate-layer result establishes accessibility to this readout family, not global identity loss in the final block.
- Extreme profile failures can make results appear better if failed faces are dropped.
- FEI's homogeneous background may allow non-face shortcuts.
- All FEI identities were exposed during smoke debugging, so only a second dataset can support confirmation.
- An evaluator reused by the conditioner will inflate generation identity scores.
- More references can improve identity similarity while harming prompt controllability and output diversity.

## 13. Immediate implementation order

1. E0, E1 and the five-seed E2 are complete. E2 repeated the complete backbone extraction exactly and spot-checked exact probe retraining only for block 1/seed 776; it did not retrain all 60 main probes twice.
2. Add identity-disjoint nuisance probes and the preregistered multi-reference E3 study without reopening E2 layer selection.
3. Run only frozen choices on Yale B+.
4. Replicate the frozen layer protocol with CVLFace ViT-B.
5. Begin generator work only if the external representation and multi-reference gates pass.

## 14. E2 execution record

E2 is complete under `configs/fei_full_lvface_probes.yaml`. The immutable run is `artifacts/runs/20260903T022924Z_fei_full_lvface_equal_capacity_probes`, and the curated result is `reports/E2_FEI_FULL_LVFACE_PROBES.md`.

Dev selected block 11, but its failure-aware internal end-to-end identity TAR advantage over the matched block-12 readout was `+0.00067`, with a paired identity-bootstrap 95% interval of `[0.00000, 0.00200]` and a positive direction for only one of five seeds. The preregistered H2 gate therefore did not pass. This execution record does not change the frozen protocol after seeing internal results: block 11 remains the dev-selected candidate for preregistered downstream comparisons, block 12 remains its matched control, and the official final embedding remains the operational baseline.

Known non-primary serialization deviation: the broad metrics section requested Recall@5, while the immutable E2 run stored only Rank-1. Every frozen main E2 method had Rank-1=1, which logically entails Recall@5=1. The reusable metric code now serializes Recall@5 for E3 and later runs; the immutable E2 run was not edited retroactively.
