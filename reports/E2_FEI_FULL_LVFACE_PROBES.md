# E2 — FEI Full Equal-capacity LVFace-T Readouts

## Status

- Final immutable run: `artifacts/runs/20260903T022924Z_fei_full_lvface_equal_capacity_probes`
- Run state: `complete`
- Decision: the internal H2 gate did **not** pass.
- Scope: FEI internal-development evidence only; not external confirmation, a causal representation claim or a generation result.

## Question

When the LVFace-T backbone is frozen and every block receives the same supervised metric-readout family, does an intermediate block transfer to unseen-identity verification better than the matched block-12 readout?

This is the fair H2 comparison. The official LVFace final embedding is reported separately as an operational baseline because it uses a different, much larger head that retains token position and was pretrained with the backbone.

## Protocol

- Data: official FEI original images, 200 identities × 14 fixed acquisition slots.
- Identity split: seed `20260902`, 100 train / 50 dev / 50 internal evaluation.
- Usable samples: 1,394 train, 694 dev and 694 internal evaluation.
- Frozen backbone: pinned LVFace-T Glint360K, 12 blocks and 19,137,792 parameters.
- Readout at every block: fresh trainable LayerNorm affine on a cached non-affine token normalization, token mean, bias-free `256→512` projection and L2 normalization. A training-only 100-class CosFace classifier is discarded for evaluation.
- Optimization: five matched seeds; CosFace `s=64, m=0.4`; AdamW; 120 fixed epochs; no augmentation or early stopping.
- Selection: five identity folds on dev. Each held-out fold uses a FAR threshold calibrated only on the other four folds. Select one global layer by five-seed mean cross-fitted template TAR@FAR=.01, then d-prime, ROC-AUC and shallower depth.
- Frozen evaluation: write `selection.json` first, then extract internal features only for the selected block and block 12. Calibrate one threshold per seed on all dev identities.
- H2 uncertainty: average each internal identity's selected-minus-block-12 end-to-end TAR over seeds, then perform 1,000 paired identity bootstrap resamples. Detection failures count as false rejects.

The E2 training recipe is a project-specific face-verification adaptation. It borrows angular geometry and some numerical defaults from LVFace and experimental principles from intermediate-layer probing work; it is not an official LVFace or ILC reproduction. Dev cross-fitted scoring used 294 positive and 2,646 negative pairs per layer/seed. Full dev template d-prime/AUC and aligned internal template metrics each used 294 positive and 14,406 negative pairs.

## Preprocessing coverage

| Split | Source images | Usable | Failed |
|---|---:|---:|---:|
| Train | 1,400 | 1,394 | 6 |
| Dev | 700 | 694 | 6 |
| Internal evaluation | 700 | 694 | 6 |
| **Total** | **2,800** | **2,782** | **18** |

All failures were `no_face_detected` for the predefined lowest-light frontal query. They remain in the manifest. Internal low-light query coverage was 44/50; all other query conditions had 50/50 coverage.

## Dev selection

| Block | Mean cross-fitted TAR | Seed SD | Mean template d-prime | Mean ROC-AUC |
|---:|---:|---:|---:|---:|
| 9 | 0.99660 | 0.00000 | 6.3330 | 0.999831 |
| 10 | **0.99864** | 0.00186 | 6.7118 | **0.999962** |
| **11** | **0.99864** | 0.00186 | **6.7883** | 0.999952 |
| 12 | 0.99728 | 0.00152 | 6.4137 | 0.999878 |

Blocks 10 and 11 tied on the preregistered primary metric. The first tiebreak, mean template d-prime, selected block 11. Per-seed winners were block 10 for seed 776 and block 11 for the other four seeds. This is one common selected block, not five seed-specific internal-evaluation choices.

## Frozen internal evaluation

| Method | Seeds | Aligned TAR mean ± SD | Observed FAR mean ± SD | End-to-end TAR mean ± SD | Template d-prime mean ± SD | Rank-1 mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| Equal-capacity block 11 | 5 | 1.00000 ± 0.00000 | 0.01427 ± 0.00144 | 0.98000 ± 0.00000 | 6.7982 ± 0.0583 | 1.0000 ± 0.0000 |
| Equal-capacity block 12 | 5 | 0.99932 ± 0.00152 | 0.01805 ± 0.00092 | 0.97933 ± 0.00149 | 6.4585 ± 0.0633 | 1.0000 ± 0.0000 |
| Official final embedding | 1 | 1.00000 | **0.00743** | 0.98000 | **9.6011** | 1.0000 |
| Final pretrained-norm token mean diagnostic | 1 | 0.91497 | 0.03276 | 0.89667 | 2.9728 | 1.0000 |

All operating thresholds in this table were calibrated on dev negatives at target FAR .01 and then frozen. The probes did not maintain FAR=.01 on internal identities: their observed FARs drifted to 1.43% and 1.80%. The selected probe's lower mean internal FAR and higher d-prime than the matched final probe are descriptive, while its mean EER was worse (`0.00654 ± 0.00024` versus `0.00353 ± 0.00032`); no uncertainty comparison was preregistered for these secondary metrics. The official final embedding remains the strongest operational representation by d-prime and observed FAR. All four rows have Rank-1=1, which entails Recall@5=1, but the immutable E2 CSV omitted an explicit Recall@5 column; that serialization gap is fixed for subsequent runs rather than retroactively editing this one.

The paired H2 estimate was:

| Quantity | Result |
|---|---:|
| Block-11 mean end-to-end identity TAR | 0.98000 |
| Block-12 mean end-to-end identity TAR | 0.97933 |
| Selected minus matched block 12 | +0.00067 |
| Identity-bootstrap 95% interval | [0.00000, 0.00200] |
| Seeds with a positive direction | 1/5 |

The interval includes zero and only one seed has a positive direction. Therefore the preregistered internal H2 gate fails. Selecting block 11 on dev does not by itself establish a meaningful intermediate-layer advantage on internal evaluation.

## Controls and optimization audit

- A single sample-independent Gaussian-feature control produced cross-fitted TAR `0.01361`, ROC-AUC `0.53334`, d-prime `0.11047` and Rank-1 `0.02721`. This realization is consistent with a signal-free null and reveals no obvious scoring-path shortcut, but one draw cannot prove the absence of every leakage route.
- Untrained random projections still preserved substantial late-layer identity structure. Their five-seed mean cross-fitted TAR was approximately `0.932` at block 10, `0.940` at block 11 and `0.850` at block 12.
- The independently condition-wise shuffled-label control, run only at seed 776, also remained strong: TAR was `0.98980`, `0.99320` and `0.97959` at blocks 10, 11 and 12. Its block-11 final training accuracy was near 100-class chance (`0.00861`), which is consistent with failure to fit the contradictory targets. High verification performance is still possible because the mapping may retain identity geometry already present in the frozen face backbone. Relative to the matched true-label run, block-11 cross-fitted d-prime increased from `5.5334` to `6.8110`; this is an incremental supervision diagnostic, not a significance test.
- Untrained and shuffled-control curves also favor the late block-10/11 region. This late-depth trend is consistent with substantial pre-existing backbone/readout geometry, but the controls do not identify its mechanism or causal role and it cannot be promoted to an identity-layer claim.
- The candidate blocks were not visibly under-trained: blocks 10–12 reached training accuracy 1.0 for every seed, with final mean losses `0.00261`, `0.00153` and `0.00130`. Earlier blocks were harder, so equal epoch budgets should not be confused with equal optimization difficulty.

## Reproducibility and performance

- Train+dev extraction: 2,088 images × 12 blocks in 2.488 s (`839.09` images/s), peak CUDA allocation 379,793,408 bytes.
- Repeating the complete train+dev extraction produced byte-identical arrays for all 36 layer/representation outputs. Probe retraining was spot-checked only at block 1/seed 776, not repeated for all 60 main probes.
- Probe training and controls: 539.93 s; peak CUDA allocation 73,099,264 bytes after moving the backbone to CPU.
- Frozen internal extraction: 694 images, blocks 11 and 12 only, in 1.016 s (`683.11` images/s), peak CUDA allocation 319,696,896 bytes.
- Final-head parity difference was exactly 0.0. Repeating block-1/seed-776 training produced identical parameter hashes and embeddings.
- The run saved 60 main checkpoints, 12 shuffled-label checkpoints and one Gaussian checkpoint; all 73 state hashes were distinct and recorded.

## Interpretation

E1 showed that an untrained token-mean diagnostic drops from block 11 to block 12. E2 shows that this gap almost disappears after both layers receive the same supervised angular readout. Identity is therefore highly accessible from both late layers under this readout family; the current evidence does not support the stronger story that the final block erases identity.

The block-11 selection and higher separation remain useful engineering observations, but they are not an “identity layer” result. They do not establish where identity is stored or what features causally drive LVFace predictions. The next representation claim must use a frozen external dataset and a second face ViT. A generator experiment should not be justified by E2 alone.

## Structured evidence

The immutable run contains resolved config and asset locks, source and package snapshots, full preprocessing coverage, dev fold assignments, every layer/seed metric, cross-fitted operating points, training curves, control metrics, checkpoint hashes, frozen internal metrics, condition metrics, failure-aware identity TAR, the paired bootstrap, extraction/access audits and performance metadata. Images, weights, checkpoints, embeddings and full run artifacts remain Git-ignored; this report contains only aggregate results.
