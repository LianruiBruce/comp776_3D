# AGENTS.md — Identity Layers Lab

This is the canonical project synchronization file requested as `agent.md`. The repository is developed on a case-insensitive Windows filesystem, so only the conventional uppercase `AGENTS.md` is maintained.

## Research objective

Determine which representations inside a face-recognition Vision Transformer retain identity across nuisance changes, then use those representations for multi-reference identity conditioning in a frozen image generator.

The falsifiable first hypothesis is:

> Some intermediate face-ViT representations yield better same-identity compactness versus different-identity separation than the final representation under controlled nuisance changes.

Do not describe a layer as an “identity layer” merely because same-person cosine is high. It must also separate different identities and be evaluated on identities excluded from layer selection.

## Current verified state

- Date: 2026-09-01 (America/New_York).
- Hardware: NVIDIA GeForce RTX 4080, 16,376 MiB reported by `nvidia-smi`.
- Python: 3.11.5 in repository-local `.venv`.
- PyTorch: 2.8.0+cu128; CUDA is available on the RTX 4080.
- Smoke data: FEI manually aligned frontal images, 200 identities × 2 conditions, official archives downloaded locally and excluded from Git.
- Smoke model: LVFace-T Glint360K, 12 blocks, 19,137,792 parameters.
- Weight integrity: SHA256 `1f5b73d6f7a18e431d95d2ccabb4d00dca2a6f2004eb74de2c5788d386cb59ec` verified.
- Model/code compatibility: strict state-dict loading passes; a CUDA forward produces finite `[2, 512]` embeddings; observed peak allocated memory was about 152 MiB for that diagnostic.
- Latest formal run: `artifacts/runs/20260901T051845Z_smoke_fei_aligned_lvface_t`; `status.json` reports `complete`.
- Formal-run performance: all 400 samples × 12 blocks in 0.7705 s (519.14 images/s), with 370,290,688 bytes (353.14 MiB) peak CUDA memory. A second complete extraction produced byte-identical arrays for all 24 layer/representation outputs; final-head parity difference was also exactly 0.0.
- Dev selection by d-prime chose block 12 for both probes. The official-head probe reached dev/test d-prime 10.8937/10.5241 and test ROC-AUC/LOO Top-1 of 1.0/1.0. Token mean reached dev/test d-prime 4.0304/4.3819 and test ROC-AUC/LOO Top-1 of 0.99805/0.9875.
- With thresholds fixed from validation negatives at target FAR 0.01, the official-head probe achieved test FAR/TAR 0.00769/1.00; token mean achieved 0.00256/0.95.
- The smoke result does **not** support an intermediate-block advantage under neutral-versus-smile variation. It validates the layer-analysis pipeline but is not evidence about pose, illumination, or generation quality. The curated interpretation is `reports/SMOKE_FEI_LVFACE_T.md`.

## Decisions and rationale

1. **LVFace-T before CVLFace ViT-B.** LVFace-T exposes all 12 blocks, has an official 76.6 MB checkpoint, and is fast enough to debug the entire pipeline. CVLFace ViT-B/AdaFace/WebFace4M is the planned stronger 24-layer replication after the smoke experiment.
2. **FEI aligned before larger data.** It is only about 12 MB and supplies 200 identities with neutral/smile pairs. It is adequate for pipeline validation, not for full pose/illumination claims.
3. **Split by identity.** The fixed seed is 776 with 120/40/40 dev/val/test identities. Dev selects layers, val calibrates frozen thresholds, and test evaluates only the frozen representations and thresholds.
4. **No silent sample dropping.** Future detection/alignment failures must remain in the manifest and detection coverage must be reported. Publisher alignment is not a detector result: the FEI smoke manifest records `face_detection_attempted=false` and a null detection outcome.
5. **No evaluator reuse for generation claims.** A generator conditioned or trained with a face recognizer must be evaluated with an independent ArcFace/AdaFace/CurricularFace checkpoint plus human judgment.
6. **Intermediate tokens have no CLS token.** LVFace uses 144 spatial tokens. Token mean is an ablation, not an official identity embedding. Applying the official head to intermediate layers is also an analysis probe; only the final-layer use matches training.

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
```

Every completed run must contain a resolved config, asset-lock and executable-source snapshots, a full Python package lock, environment metadata, manifest snapshot, metrics, frozen validation-calibrated operating points, selection and performance records, log, and completion status. Do not manually edit a completed run; create a new run.

## Reproducibility and leakage rules

- Resolve all relative paths against the repository root.
- Record source URL, upstream revision, SHA256, preprocessing, identity split, and random seed.
- Never select a layer, pooling rule, threshold, or hyperparameter on the held-out test identities.
- Profile all layers only on dev/val. Calibrate operating thresholds on val negatives, then run test only for the frozen representation. The FEI smoke test identities were exposed during early pipeline debugging and are not a fresh confirmatory set.
- Report positive and negative pair counts along with aggregate scores.
- Count face-detection failures rather than deleting them. FEI aligned currently uses the dataset's manual alignment plus the upstream LVFace resize policy; label this clearly.
- Raw/processed face data and pretrained weights are local research assets and must remain Git-ignored.
- FEI and LVFace pretrained weights are for research/non-commercial use according to their publishers. Do not redistribute them.
- Preserve user changes and unrelated work. Use `apply_patch` for source/document edits.

## Update protocol for agents

After any material experiment or design change, update:

1. **Current verified state** with exact evidence, not plans.
2. **Decisions and rationale** if a model, dataset, metric, or split changes.
3. The latest run path and the next concrete action below.

Do not paste long logs here; link to structured artifacts or a curated report.

## Next actions

1. Add an independent ArcFace-family evaluator and five-point alignment for non-aligned data.
2. Run FEI original for pose/expression/illumination variation; add Extended Yale Face B for controlled illumination if access remains stable.
3. Replicate the layer curve with CVLFace ViT-B AdaFace WebFace4M.
4. Add identity-level bootstrap confidence intervals and a final-head-independent, equal-capacity intermediate-layer probe.
5. Test multi-reference aggregation using only dev/val identities, then freeze the rule before test evaluation.
6. Only then connect selected multi-layer/multi-reference features to PhotoMaker/PuLID or a small frozen-generator adapter.
