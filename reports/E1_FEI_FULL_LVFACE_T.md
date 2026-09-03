# E1 — FEI Full Layer Sweep with LVFace-T

## Status

- Final immutable run: `artifacts/runs/20260903T013013Z_fei_full_lvface_layers`
- Run state: `complete`
- Scope: FEI development evidence only; not an external confirmation and not a generation result.

## Question

Does a fixed, layer-wise representation retain identity better at an intermediate LVFace-T block than at the final block when pose, expression, scale and illumination vary?

## Protocol

- Data: official FEI original images, 200 identities × 14 acquisition slots.
- Identity split: seed `20260902`, 100 train / 50 dev / 50 internal evaluation.
- Fixed query slots: `01, 03, 08, 10, 12, 14`.
- Fixed reference slots: `02, 04, 05, 06, 07, 09, 11, 13`.
- Alignment: pinned InsightFace commit `7fadd420c2351d0ffa8cac403421c1a3ed733365`, SCRFD-10G model SHA256 `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`, ArcFace five-point 112×112 transform.
- Backbone: pinned LVFace-T Glint360K, 12 blocks.
- Layer selection: maximize dev template-to-query TAR at FAR 0.01; ties use template d-prime, then ROC-AUC, then the shallower layer.
- Thresholds: calibrate from dev negatives and apply unchanged to the internal-evaluation identities.
- Uncertainty: 1,000 paired identity-level bootstrap resamples.

The query/reference condition names are derived protocol labels based on the fixed FEI acquisition sequence, not publisher annotations.

## Preprocessing coverage

| Item | Result |
|---|---:|
| Source images | 2,800 |
| Successfully detected and aligned | 2,782 |
| End-to-end coverage | 99.357% |
| Failures | 18 |

All 18 failures were `no_face_detected` for acquisition slot 14, the lowest-light frontal condition: six train, six dev and six internal-evaluation images. They remain in `preprocess_coverage.csv`; no failed sample was silently deleted. Extreme left and right profiles were detected for every identity.

## Layer-selection result

| Representation | Dev-selected block | Dev template TAR@FAR=.01 | Dev template d-prime | Internal frozen TAR | Internal observed FAR |
|---|---:|---:|---:|---:|---:|
| Official head applied per block | 12 | 1.0000 | 8.9242 | 1.0000 | 0.00743 |
| Raw token mean | 11 | 0.9558 | 3.2995 | 0.9626 | 0.01347 |
| Raw token mean, final-block baseline | 12 | 0.8571 | 2.8648 | 0.9150 | 0.03276 |

For raw token mean, block 11 improved identity-mean frozen TAR over block 12 by `+0.0467`, with paired 95% bootstrap CI `[+0.0167, +0.0767]` across the 50 internal-evaluation identities. Block 11 also had a lower observed internal FAR than block 12, so the TAR gain was not obtained by accepting more impostor comparisons.

The official-head probe selected block 12. Its final-block use is the only one matching LVFace training, and it was substantially stronger than token mean.

## Nuisance breakdown

Using the global dev-calibrated threshold, block-11 token mean obtained the following internal-evaluation TAR:

| Query condition | Coverage | Frozen TAR | Observed FAR |
|---|---:|---:|---:|
| Low-light frontal | 0.88 | 0.9773 | 0.00835 |
| Smile frontal | 1.00 | 1.0000 | 0.01592 |
| Extreme left profile | 1.00 | 0.9400 | 0.00367 |
| Left three-quarter | 1.00 | 1.0000 | 0.02571 |
| Extreme right profile | 1.00 | 0.8600 | 0.00286 |
| Right three-quarter / expression | 1.00 | 1.0000 | 0.02367 |

The official final head achieved frozen TAR 1.0 for every successfully aligned query condition. Detection coverage, rather than recognition, was its remaining failure mode in this run.

## Reproducibility and performance

- All 2,782 usable samples × 12 blocks were extracted in 4.31 seconds: 644.80 images/s.
- Peak allocated CUDA memory was 370,290,688 bytes (353.14 MiB).
- Repeating the full extraction produced exact equality for all 24 layer/representation arrays.
- Final block plus official head exactly matched the official model forward output.
- Switching from the InsightFace wheel to the pinned official source checkout produced identical embedding fingerprints, layer metrics and bootstrap results.

## Interpretation

E1 provides internal-development evidence that a naive token-mean readout degrades from block 11 to block 12 under the FEI nuisance protocol. It does **not** show that LVFace loses identity information globally: the trained official head reads the final block extremely well. A more plausible interpretation is that identity information is reorganized or is not linearly exposed by simple spatial averaging at the final block.

Therefore E1 does not yet establish an “identity layer.” The required next experiment is E2: train the same small 512-D probe at every frozen block using train identities, select on dev, and evaluate once on the internal split. External Yale B+ confirmation and a second face ViT are still required before making a general claim.

## Structured evidence

The run contains `dataset_manifest.csv`, `preprocess_coverage.csv`, `metrics_by_layer.csv`, `metrics_by_condition.csv`, `bootstrap_intervals.csv`, `selection.json`, `frozen_operating_points.json`, exact embedding fingerprints, environment/package locks, asset locks and executable-source snapshots. Raw images, aligned images, model weights and run artifacts remain excluded from Git.
