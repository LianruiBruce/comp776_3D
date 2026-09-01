# FEI aligned × LVFace-T smoke experiment

## Status and scope

- Run ID: `20260901T051845Z_smoke_fei_aligned_lvface_t`
- Status: complete
- Date: 2026-09-01 (America/New_York)
- Hardware: NVIDIA GeForce RTX 4080
- Data: FEI manually aligned frontal subset, 200 identities × neutral/smile
- Identity-disjoint split: 120 dev / 40 validation / 40 test identities, fixed seed 776
- Model: LVFace-T Glint360K, 12 Transformer blocks, 19,137,792 parameters
- Layer selection: maximize d-prime on dev identities independently for each representation

This is a pipeline and representation-analysis smoke test. It does not test generation, broad pose or illumination invariance, or real-world face verification.

The FEI test identities were exposed to full layer curves in the early diagnostic implementation. They remain useful for this smoke result but must not be treated as untouched confirmatory identities in later model development.

## Result

Both analysis probes selected block 12. The experiment therefore does not support the hypothesis that an intermediate block is better than the final block under this dataset's neutral-versus-smile change.

| Representation | Dev-selected block | Dev d-prime | Dev ROC-AUC | Test d-prime | Test ROC-AUC | Test EER | Test LOO Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official head applied per block | 12 | 10.8937 | 0.999948 | 10.5241 | 1.000000 | 0.000000 | 1.0000 |
| Normalized token mean | 12 | 4.0304 | 0.999025 | 4.3819 | 0.998053 | 0.027724 | 0.9875 |

For the official-head probe at block 12, test same-identity cosine was 0.8081 versus 0.0258 for different identities. For token mean, test same-identity cosine was 0.9989 versus 0.9945 for different identities. Thus a high same-identity cosine by itself is not evidence that a representation preserves discriminative identity.

The official head is trained for the final block only. Applying it to earlier blocks is an analysis probe, not a claim that those blocks were trained to be valid head inputs. Token mean is likewise an ablation because LVFace-T has spatial tokens rather than a CLS token.

Validation-negative calibration at target FAR 0.01 gives a frozen threshold of 0.19833 for the official-head probe. On test it yields FAR 0.00769 and TAR 1.00. Token mean's frozen threshold is 0.99744; on test it yields FAR 0.00256 and TAR 0.95. These are the thresholded operating-point results; the per-split EER and TAR@FAR columns are descriptive ROC summaries, not deployed thresholds.

## Runtime and reproducibility

- 400 images × 12 hooked blocks: 0.7705 seconds
- Throughput: 519.14 images/s
- Peak allocated CUDA memory: 370,290,688 bytes (353.14 MiB)
- Repeated-forward maximum absolute difference: 0.0
- Final-block projected embedding versus official model output: maximum absolute difference 0.0
- A second complete extraction produced byte-identical arrays for every one of the 24 layer/representation outputs. Shared layer metrics also matched the earlier diagnostic runs.
- The run stores a 22-file executable source snapshot, resolved asset lock, complete `pip freeze --all`, CUDA/cuDNN/driver metadata, and per-array embedding fingerprints. The main Git worktree was dirty because this new project had not yet been committed; the exact local run state is therefore reconstructed from this snapshot rather than the older Git revision alone.

Canonical artifact hashes:

| Artifact | SHA256 |
|---|---|
| `metrics_by_layer.csv` | `8f152a77e0df6aa01be525224d51fdb3329d64ac89022c7dbe2b987cffbbd72c` |
| `selection.json` | `7832a14954e803f0e223d168d38f3c08273940b519d6eeba781880e9c7c56ff9` |
| `frozen_operating_points.json` | `9197931dfe6f9538465abcbd60692f3c7bec4b9584cdae2e2ddd8060609440b8` |
| `summary.json` | `43d812362c63e2cd7ccbc0a77acb6d172a3d21073457a360cae611afb6267d5d` |

The immutable local run is stored at `artifacts/runs/20260901T051845Z_smoke_fei_aligned_lvface_t/` and contains the per-layer metrics, frozen operating points, selection record, source/environment snapshots and dev layer curves. Run artifacts are intentionally Git-ignored because they are reproducible machine outputs, so these paths become available after a teammate reproduces the run rather than through GitHub.

## Validity limits and next experiment

FEI aligned has only two near-frontal conditions per person and the smoke preprocessing directly resizes the publisher-aligned crop to 112 × 112. Its near-perfect scores indicate that the task is easy, not that the model has solved identity preservation in generation.

The test split contains only 40 positive pairs, while negative pairs share identities and are statistically dependent. These are point estimates without identity-level bootstrap intervals. Possible identity overlap between FEI and LVFace's Glint360K training data is also unknown.

The next informative experiment should add controlled pose/illumination variation, five-point face alignment, an independent ArcFace-family evaluator, identity-level bootstrap intervals, and an equal-capacity probe that is not tied to LVFace's final head. Layer and multi-reference aggregation rules must be chosen on dev, checked on validation, and frozen before the held-out test and later generator evaluation.
