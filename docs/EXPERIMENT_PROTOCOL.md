# Experiment protocol

## Research question

Given several photographs of the same person under different nuisance conditions, can a face-recognition ViT layer or layer mixture represent identity more reliably than the conventional final representation, and can that representation improve identity-conditioned generation without copying nuisance attributes?

## Hypotheses

- **H1 — layer effect:** identity separation varies materially across face-ViT depth.
- **H2 — invariance trade-off:** the layer maximizing same/different separation is not necessarily the layer with the largest raw same-person cosine.
- **H3 — multi-reference effect:** quality-aware aggregation across images and layers improves held-out identity retrieval over final-layer averaging.
- **H4 — generation effect:** a selected representation improves independent face similarity while preserving prompt compliance and face diversity.

The current smoke run tests only whether H1 is measurable under a neutral-versus-smile perturbation. It does not test H2–H4 completely.

## Stage 1: layer profiling

1. Use identities, not images, as the split unit.
2. Apply one documented face alignment and normalization pipeline to every sample.
3. Extract every Transformer block plus the official final embedding in evaluation mode.
4. For each layer and representation on dev/val compute positive/negative cosine distributions, d-prime, ROC-AUC, EER, within-split TAR@FAR, and leave-one-out Top-1.
5. Select layers on dev identities only. Calibrate operating thresholds on val negative pairs, freeze both layer and threshold, then evaluate only those selected representations on test once. Do not produce or inspect full test layer curves in confirmatory experiments.
6. Repeat with at least two face-specific ViTs and one generic ViT control.

## Stage 2: nuisance probes

Use controlled or derived labels for yaw, pitch, illumination, expression, occlusion and image quality. Train linear probes with identity-disjoint folds. A desirable identity representation has high identity verification performance but low nuisance predictability. Estimated pose/quality labels must be marked as derived, not ground truth.

## Stage 3: multi-reference aggregation

Compare equal-compute baselines:

- final layer, normalized image mean;
- best fixed intermediate layer;
- all-layer mean;
- stacked image embeddings;
- learned sparse layer weights;
- image-quality and layer-aware attention pooling.

All learned aggregation uses train/dev identities only. Ablations must separate gains from extra parameters and from simply adding more reference images.

## Stage 4: generation

Start with frozen SDXL-based PhotoMaker/PuLID baselines because they fit a 16GB GPU and have mature identity adapters. Train only a small mapper/adapter or LoRA. Use a FLUX/PuLID-FLUX DiT replication after the representation hypothesis is supported.

Generation evaluation must include an identity evaluator not used by the conditioner or loss, prompt compliance, detection coverage, face diversity/copying analysis, fixed prompts/seeds/sample counts, identity-level confidence intervals, and blinded human identity judgments.

## Threats to validity

- Face-recognition similarities can reward adversarial or unnatural faces.
- Celebrity datasets may overlap generator pretraining.
- Tight face alignment can hide sensitivity to detection failures.
- Pairwise confidence intervals can be too narrow because pairs are not independent; identity-level bootstrap is preferred for final reporting.
- A head trained for the final layer may unfairly disadvantage intermediate layers.
- FEI aligned is too controlled for substantive claims; it is only the engineering gate.
