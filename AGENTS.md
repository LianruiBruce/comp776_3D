# AGENTS.md — Identity Layers Lab

This is the canonical project synchronization file requested as `agent.md`. The repository is developed on a case-insensitive Windows filesystem, so only the conventional uppercase `AGENTS.md` is maintained.

## Research objective

Determine which identity information is lost between reference photographs and generated faces, and which conditioning information improves human-perceived likeness. Layer-wise analysis of a face-recognition Vision Transformer is an upstream diagnostic, not the final research target.

The falsifiable first hypothesis is:

> Some intermediate face-ViT representations yield better same-identity compactness versus different-identity separation than the final representation under controlled nuisance changes.

Do not describe a layer as an “identity layer” merely because same-person cosine is high. It must also separate different identities and be evaluated on identities excluded from layer selection.

For generation, automatic recognizers may establish identity steering and localize information channels, but only a blinded human study may support a claim about perceived likeness.

## Current verified state

- Date: 2026-09-03 (America/New_York).
- Hardware: NVIDIA GeForce RTX 4080, 16,376 MiB reported by `nvidia-smi`.
- Python: 3.11.5 in repository-local `.venv`.
- PyTorch: 2.8.0+cu128; CUDA is available on the RTX 4080.
- Smoke data: FEI manually aligned frontal images, 200 identities × 2 conditions, official archives downloaded locally and excluded from Git.
- Smoke model: LVFace-T Glint360K, 12 blocks, 19,137,792 parameters.
- Weight integrity: SHA256 `1f5b73d6f7a18e431d95d2ccabb4d00dca2a6f2004eb74de2c5788d386cb59ec` verified.
- Model/code compatibility: strict state-dict loading passes; a CUDA forward produces finite `[2, 512]` embeddings; observed peak allocated memory was about 152 MiB for that diagnostic.
- Latest aligned-subset smoke run: `artifacts/runs/20260901T051845Z_smoke_fei_aligned_lvface_t`; `status.json` reports `complete`.
- Formal-run performance: all 400 samples × 12 blocks in 0.7705 s (519.14 images/s), with 370,290,688 bytes (353.14 MiB) peak CUDA memory. A second complete extraction produced byte-identical arrays for all 24 layer/representation outputs; final-head parity difference was also exactly 0.0.
- Dev selection by d-prime chose block 12 for both probes. The official-head probe reached dev/test d-prime 10.8937/10.5241 and test ROC-AUC/LOO Top-1 of 1.0/1.0. Token mean reached dev/test d-prime 4.0304/4.3819 and test ROC-AUC/LOO Top-1 of 0.99805/0.9875.
- With thresholds fixed from validation negatives at target FAR 0.01, the official-head probe achieved test FAR/TAR 0.00769/1.00; token mean achieved 0.00256/0.95.
- The smoke result does **not** support an intermediate-block advantage under neutral-versus-smile variation. It validates the layer-analysis pipeline but is not evidence about pose, illumination, or generation quality. The curated interpretation is `reports/SMOKE_FEI_LVFACE_T.md`.
- E1 data: all four official FEI original archives are pinned locally, 200 identities × 14 acquisition slots. The fixed seed `20260902` assigns 100 train / 50 dev / 50 internal-evaluation identities; six slots are queries and eight are references.
- E1 preprocessing: pinned InsightFace commit `7fadd420c2351d0ffa8cac403421c1a3ed733365`, SCRFD-10G detector SHA256 `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`, ONNX Runtime GPU 1.26.0 and the ArcFace five-point 112×112 transform. It aligned 2,782/2,800 samples (99.357%); all 18 failures were the lowest-light slot and remain recorded.
- Latest formal run: `artifacts/runs/20260903T013013Z_fei_full_lvface_layers`; `status.json` reports `complete`. The curated interpretation is `reports/E1_FEI_FULL_LVFACE_T.md`.
- E1 extraction: 2,782 samples × 12 blocks in 4.31 s (644.80 images/s), 370,290,688 bytes peak CUDA allocation. A complete repeat produced byte-identical arrays for all 24 representations; final-head parity was exactly 0.0.
- E1 selection: the official-head diagnostic selected block 12 and achieved internal frozen template TAR/FAR 1.0000/0.00743. Raw token mean selected block 11; internal frozen TAR/FAR was 0.9626/0.01347 versus block 12 at 0.9150/0.03276.
- For raw token mean, the block-11 minus block-12 identity-mean TAR difference was +0.0467 with a paired 1,000-resample 95% bootstrap CI of [+0.0167, +0.0767]. This is FEI internal-development evidence only, not external confirmation.
- E1 does not establish an identity layer: the official final head remains much stronger, so the block-11 token-mean gain may reflect readout mismatch or representation reorganization. This motivated the now-complete E2 matched-readout test below.
- Latest E2 smoke run: `artifacts/runs/20260903T025857Z_smoke_fei_full_lvface_equal_capacity_probes`; `status.json` reports `complete`. It validates the hardened config/checkpoint/order checks and serializes `template_recall_at_5` throughout the current runner.
- Latest formal run: `artifacts/runs/20260903T022924Z_fei_full_lvface_equal_capacity_probes`; `status.json` reports `complete`. The curated interpretation is `reports/E2_FEI_FULL_LVFACE_PROBES.md`.
- E2 trained the same fresh-LayerNorm/token-mean/bias-free 256→512 angular readout at all 12 frozen blocks for seeds `{776,1776,2776,3776,4776}`. All usable images of 100 train identities were used for 120 fixed epochs; 50 dev identities selected one common layer with identity-fold cross-fitted TAR@FAR=.01. This is a project-specific supervised metric readout, not an official LVFace/ILC recipe or a classic affine linear probe.
- Dev selected block 11: blocks 10 and 11 tied at mean cross-fitted TAR 0.99864, then block 11 won by mean template d-prime 6.7883 versus 6.7118. Four of five seed-level dev winners were block 11; one was block 10. Block 12 reached TAR 0.99728 and d-prime 6.4137.
- After `selection.json` was written, internal evaluation extracted only blocks 11 and 12. Block 11 versus matched block 12 had failure-aware end-to-end identity TAR 0.98000 versus 0.97933, delta +0.00067 with paired identity-bootstrap 95% CI `[0.00000, 0.00200]`; only 1/5 seed directions was positive. The preregistered internal H2 gate therefore **did not pass**.
- On aligned internal queries, block-11 matched readouts reached mean TAR/FAR 1.00000/0.01427; matched block 12 reached 0.99932/0.01805. The heterogeneous official final embedding remained the stronger operational baseline at TAR/FAR 1.00000/0.00743 and template d-prime 9.6011. Do not use the official embedding as the equal-capacity H2 comparator.
- E2 controls behaved coherently but require care: Gaussian features were near chance (cross-fitted TAR 0.01361, ROC-AUC 0.53334), while untrained and shuffled-label projections retained substantial identity geometry already present in late frozen features. Correct-label training improved separation, but the high absolute late-layer performance is not wholly attributable to probe supervision.
- E2 reproducibility/performance: 2,088 train+dev images × 12 blocks extracted in 2.488 s at 839.09 images/s and repeated byte-identically; all probe/control training took 539.93 s with 73,099,264 bytes peak CUDA allocation after offloading the backbone. Frozen internal extraction processed 694 images at blocks 11/12 in 1.016 s. Probe repeat and final-head parity differences were exactly 0.0.
- The immutable E2 tables omitted an explicit Recall@5 column; all frozen main methods had Rank-1=1, which entails Recall@5=1. Reusable template metrics now serialize `template_recall_at_5` for E3 and later runs; completed E2 artifacts remain unchanged.
- The output-level generation pilot is frozen in `configs/generation_identity_pilot.yaml`: official PhotoMaker V2 on RealVisXL V4, 8 already-exposed FEI `internal_eval` identities, 6 identity-channel/reference conditions, 2 prompts and 2 seeds, for 192 images. It does not use an E2 layer or a learned project mapper.
- Latest formal generation run: `artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot`. Generation completed 192/192 at 1024x1024 with no failed cells; every output produced exactly one detected face. Total generation time was 1,788.8318 s (mean 9.3168 s/image), with peak CUDA allocation/reservation of 6,866,930,688/14,524,874,752 bytes.
- The latest valid automatic evaluation is `evaluation_v3/` and its identity-bootstrap summary is `analysis_v3/`; root status is `automatic_evaluation_complete_human_pending`. The earlier `evaluation/` is a preserved failed validator attempt. All six core numerical outputs from the first successful `evaluation_v2/` and the reproducibility rerun `evaluation_v3/` are byte-identical.
- With independent LVFace evaluation, K=1, diverse K=4 and repeated K=4 all reached 8-way Rank-1 1.0. Their target cosine similarities were 0.56125, 0.57434 and 0.56045, versus 0.91316 for a real K=1 reference scored against held-out real queries. Thus categorical identity retrieval is saturated while fine-grained reference-to-query similarity remains substantially below the real-photo baseline.
- Diverse K=4 minus K=1 target similarity was +0.01309 with identity-bootstrap 95% CI `[-0.01081, +0.03813]`; diverse K=4 minus repeated K=4 was +0.01389 `[+0.00012, +0.02965]`. The corresponding independent target-versus-impostor margin intervals both crossed zero, and repeated K=4 was indistinguishable from K=1. Treat the multi-view gain as small exploratory evidence, not a perceptual conclusion.
- Native K=1 minus text-only was +0.55228 target cosine `[+0.51139, +0.59246]` and +0.9375 Rank-1 `[+0.84375, +1.00000]` under independent LVFace, establishing that the released identity path strongly steers the generated identity.
- Reciprocal out-of-distribution channel conflicts show that the 512-D global InsightFace embedding dominates the CLIP pixel/patch identity route: independent LVFace followed the global source in 60/64 outputs (93.75%); the reused conditioner evaluator did so in 63/64 (98.44%). This localizes output influence, but does not prove which normal-path details humans use for likeness.
- Normal generated identity cohorts retained only 90.64%, 90.19% and 91.33% of the real cohort's between-identity spread for K=1, diverse K=4 and repeated K=4 under LVFace. Generated samples were *less* similar to the literal real-cohort centroid, so report this only as between-identity embedding contraction, never attraction to a cohort mean face.
- A blinded identity-likeness master package is built at `human_eval/participant/`, with 192 four-alternative identity trials and 160 pairwise trials (352 total), plus a separate private key and response analyzer. Of the pairwise items, 128 implement the four frozen contrasts and 32 direct K=1-versus-diverse-K=4 (`P0`) items are exploratory. The package validates the anonymous media/response path but is not yet the formal collection instrument: v1 requires full coverage from each rater and lacks the preregistered incomplete blocks, attention checks and separate prompt/quality screens. No real responses have been collected; no human-likeness claim is currently permitted.

## Decisions and rationale

1. **LVFace-T before CVLFace ViT-B.** LVFace-T exposes all 12 blocks, has an official 76.6 MB checkpoint, and is fast enough to debug the entire pipeline. CVLFace ViT-B/AdaFace/WebFace4M is the planned stronger 24-layer replication after the smoke experiment.
2. **FEI aligned before larger data.** It is only about 12 MB and supplies 200 identities with neutral/smile pairs. It is adequate for pipeline validation, not for full pose/illumination claims.
3. **Split by identity.** The fixed seed is 776 with 120/40/40 dev/val/test identities. Dev selects layers, val calibrates frozen thresholds, and test evaluates only the frozen representations and thresholds.
4. **No silent sample dropping.** Future detection/alignment failures must remain in the manifest and detection coverage must be reported. Publisher alignment is not a detector result: the FEI smoke manifest records `face_detection_attempted=false` and a null detection outcome.
5. **No evaluator reuse for generation claims.** A generator conditioned or trained with a face recognizer must be evaluated with an independent ArcFace/AdaFace/CurricularFace checkpoint plus human judgment.
6. **Intermediate tokens have no CLS token.** LVFace uses 144 spatial tokens. Token mean is an ablation, not an official identity embedding. Applying the official head to intermediate layers is also an analysis probe; only the final-layer use matches training.
7. **Remove final-head bias with matched readouts.** E2 compares identical 512-D supervised angular readouts trained on train identities with a frozen backbone; final-head projections remain diagnostic only. “Equal capacity” applies across the 12 E2 readouts, not between an E2 readout and the much larger official LVFace head.
8. **Separate development from confirmation.** FEI full supplies 14 controlled images per identity and is the development set. All FEI identities were touched by smoke debugging, so only a frozen external Yale B+ run may support confirmation.
9. **Gate generation.** PhotoMaker V2 reference selection is the first generator experiment because it natively supports multiple references. A custom mapper is not trained unless multi-reference gains replicate externally.
10. **Evaluator independence is method-specific.** PhotoMaker V2 and PuLID consume InsightFace-style identity features, so ArcFace cannot be their sole primary evaluator; use an independent AdaFace/CurricularFace model and retain ArcFace only as a secondary diagnostic.
11. **Pin detector code as source, not a pip wheel.** InsightFace is a clean checkout at a fixed official commit. Its wheel declares CPU `onnxruntime`, which conflicts with the one-runtime-package rule; the project instead uses the pinned source with `onnxruntime-gpu==1.26.0`, the newest PyPI line compatible with CUDA 12.8 before ORT 1.27 switched to CUDA 13.
12. **Treat E1 as a readout result.** Block 11 improves raw token mean over block 12 internally, while the trained official head is best at block 12. Do not interpret this as final-layer identity loss until equal-capacity probes and external replication agree.
13. **Treat E2 as a null matched-layer result.** Dev selects block 11, but its frozen internal advantage over the matched block-12 readout is only one accepted query across all five seed runs, the identity-bootstrap interval touches zero and only one seed direction is positive. This does not support final-layer identity erasure. Freeze the result; do not search internal identities for another layer.
14. **Separate upstream facts from project design.** LVFace supplies the frozen face ViT, official inference baseline, angular geometry and some numerical defaults. Fresh mean-pooled readouts, fixed batch 128/no augmentation, power-1 decay, five matched seeds, identity cross-fitting and the H2 gate are project decisions. ILC contributes general frozen-probe/validation-selection precedent, not this face-verification protocol.
15. **Pivot the generation question to observable identity loss.** Because E2 did not support reliable final-layer erasure, the first generator study uses the released PhotoMaker V2 identity path and intervenes on reference diversity and its global-versus-patch sources. Do not inject block 11 or train a mapper based on the null E2 result.
16. **Separate categorical identity, continuous fidelity and perceived likeness.** Rank-1, target cosine/margin and blinded human choices answer different questions. Saturated Rank-1 is not evidence that a portrait looks like the person, and automatic cosine is not a human judgment.
17. **Freeze the eight-identity pilot after automatic analysis.** The global-channel dominance and small diverse-reference gain are hypotheses for external and human confirmation. Do not tune prompts, seeds, donors, merge step or reference selection on these generated outputs.

## Filesystem contract

```text
configs/                 versioned experiment configs and immutable asset lock
data/                    dataset card and versioned manifests only
  raw/                   downloaded archives; never commit
  processed/             derived images; never commit
  manifests/             non-image provenance tables; commit when curated
docs/                    hypotheses, protocol, validity threats
models/                  model card; no weights
src/id_layers/           reusable research package
scripts/                 asset bootstrap and maintenance entry points
tests/                   deterministic unit tests
artifacts/cache/         weights and external caches; never commit
artifacts/runs/          complete machine-generated runs; never commit
reports/                 curated, reviewable findings and figures
third_party/LVFace/      pinned external checkout; never commit
third_party/InsightFace/ pinned external checkout; never commit
third_party/PhotoMaker/  pinned external checkout; never commit
```

Every completed run must contain a resolved config, asset-lock and executable-source snapshots, a full Python package lock, environment metadata, manifest snapshot, metrics, frozen validation-calibrated operating points, selection and performance records, log, and completion status. Do not manually edit a completed run; create a new run.

## Reproducibility and leakage rules

- Resolve all relative paths against the repository root.
- Record source URL, upstream revision, SHA256, preprocessing, identity split, and random seed.
- Never select a layer, pooling rule, threshold, or hyperparameter on the held-out test identities.
- Profile all layers only on dev/val. Calibrate operating thresholds on val negatives, then run test only for the frozen representation. The FEI smoke test identities were exposed during early pipeline debugging and are not a fresh confirmatory set.
- For FEI-full E1/E2, train is reserved for learned probes, dev selects layers and calibrates thresholds, and `internal_eval` receives only frozen choices. Because all FEI identities have been touched during development, this split is not a fresh external test.
- Report positive and negative pair counts along with aggregate scores.
- Count face-detection failures rather than deleting them. FEI aligned currently uses the dataset's manual alignment plus the upstream LVFace resize policy; label this clearly.
- Raw/processed face data and pretrained weights are local research assets and must remain Git-ignored.
- FEI and LVFace pretrained weights are for research/non-commercial use according to their publishers. Do not redistribute them.
- Generated faces, aligned evaluation crops, embeddings, blinded media, participant responses and private answer keys remain under Git-ignored run directories. Obtain the applicable institutional approval or exemption before showing identifiable FEI faces to raters.
- Preserve user changes and unrelated work. Use `apply_patch` for source/document edits.

## Update protocol for agents

After any material experiment or design change, update:

1. **Current verified state** with exact evidence, not plans.
2. **Decisions and rationale** if a model, dataset, metric, or split changes.
3. The latest run path and the next concrete action below.

Do not paste long logs here; link to structured artifacts or a curated report.

## Next actions

1. Upgrade the v1 human master package to balanced incomplete blocks with attention checks and separate prompt/quality screens; obtain the applicable institutional approval/exemption, freeze assignment/exclusion rules, then collect at least five independent blinded ratings per item.
2. Add an independent AdaFace/CurricularFace-family sensitivity evaluator and an external identity cohort before promoting any automatic generation result beyond exploratory evidence.
3. Use the human results to decide whether the next intervention targets diverse local appearance/geometry cues or only the global identity embedding; do not choose this from reused InsightFace scores alone.
4. Add pinned Yale B+ asset metadata and run only the frozen block-11/matched-block-12/official representation baselines and a frozen multi-reference rule for external confirmation.
5. Replicate the frozen layer protocol with CVLFace ViT-B; train a generator mapper only after external representation and human-perception gates pass.
