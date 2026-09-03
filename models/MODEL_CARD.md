# Model card

## LVFace-T Glint360K

- Paper/repository: <https://github.com/bytedance/LVFace>
- Official weights: <https://huggingface.co/bytedance-research/LVFace>
- Architecture: face-recognition Vision Transformer, 12 blocks, 144 spatial tokens, hidden size 256, final embedding 512.
- Parameters verified locally: 19,137,792.
- Input: aligned RGB face, 112×112, normalized channel-wise by `(x - 0.5) / 0.5` after conversion to `[0, 1]`.
- There is no CLS token.

Pinned code revision, weight revision and SHA256 are in `configs/assets.lock.yaml`. The code is MIT licensed. The upstream repository describes the pretrained weights as non-commercial research assets; weights are never committed here.

## InsightFace SCRFD-10G preprocessing model

- Purpose: face detection and five-landmark localization only; it is not an identity evaluator.
- Code: official InsightFace checkout pinned at commit `7fadd420c2351d0ffa8cac403421c1a3ed733365`.
- Model pack: official `buffalo_l` release; only `det_10g.onnx` is loaded.
- Detector SHA256: `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`.
- Runtime: `onnxruntime-gpu==1.26.0`, with `CUDAExecutionProvider` required before preprocessing begins.
- Output: one deterministically selected face, five landmarks, and the official ArcFace 112×112 similarity transform.

The InsightFace code is MIT licensed; the publisher restricts provided pretrained models to non-commercial research. The checkout, model archive and ONNX file remain local and Git-ignored. Every run records the detector result, box, landmarks, transform, quality proxies and failure reason.

## InsightFace buffalo_l recognition model

- Purpose in the generation pilot: provide the global face-recognition vector consumed by PhotoMaker V2.
- File: `w600k_r50.onnx` from the same pinned official `buffalo_l` archive.
- SHA256: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
- Evaluation role: secondary conditioner-reuse diagnostic only, never the sole evidence for a generation identity claim.

The model is kept local under `artifacts/cache/models/insightface/`. Although InsightFace source code is MIT licensed, the publisher's pretrained model terms restrict these weights to non-commercial research. That restriction also applies when the vector is used through PhotoMaker V2.

## PhotoMaker V2 identity adapter

- Repository: <https://github.com/TencentARC/PhotoMaker>.
- Pinned code revision: `060b4fcb10b76a4554edf565d6106b7e36c968f0`.
- Adapter repository: <https://huggingface.co/TencentARC/PhotoMaker-V2>.
- Adapter revision: `f5a1e5155dc02166253fa7e29d13519f5ba22eac`.
- Adapter SHA256: `0eeac57d7e09b95cee670a18e0d939ff303dd6f76d317c950bf584c6c9bdb8f5`.
- Local purpose: frozen, training-free multi-reference generation and controlled routing of a target/donor identity through the InsightFace-global and CLIP-patch inputs.
- Runtime used here: fp16 SDXL pipeline, fused adapter LoRA, model CPU offload, Euler scheduler, 30 steps, CFG 5 and 1024×1024 output.

PhotoMaker V2 combines the InsightFace-derived global representation with visual patch information before inserting identity tokens into the text-conditioned diffusion path. The conflict routes in this repository are project diagnostics, not an upstream PhotoMaker benchmark. The `text_only` control keeps the same PhotoMaker-loaded pipeline but delays identity merging past all 30 denoising steps; it is not an untouched base-SDXL control.

The upstream PhotoMaker code and adapter are Apache-2.0, but PhotoMaker V2 explicitly relies on InsightFace. Use of this stack in this repository is therefore limited by the InsightFace pretrained-model research restriction as well as the FEI data terms. The checkout and adapter are Git-ignored and must not be redistributed from this repository.

## RealVisXL V4 fp16 base model

- Repository: <https://huggingface.co/SG161222/RealVisXL_V4.0>.
- Pinned revision: `26dfe44930964cd70d0a817b6d1cc945c130e38d`.
- Files used: locked fp16 text encoders, UNet and VAE plus scheduler/tokenizer/config files.
- License identifier recorded by the asset lock: `CreativeML-OpenRAIL++`.

This base model is used only with the frozen PhotoMaker V2 adapter. Every snapshot file used by the experiment is allow-listed and hash-verified in `configs/assets.lock.yaml`; do not replace it with a similarly named Hugging Face revision and call the run comparable.

## Generation evaluator separation

The formal generation pilot uses the official final LVFace-T embedding as its primary automatic identity evaluator because neither PhotoMaker V2 conditioning nor generation loss uses LVFace. InsightFace `w600k_r50.onnx`, which participates in conditioning, is retained only as a secondary sensitivity diagnostic. Automatic face-recognition metrics are not human judgments; any statement that one condition “looks more like the person” requires the preregistered blinded human evaluation.

## Intermediate representations and readouts

The layer experiments register read-only forward hooks on `model.blocks[0:12]` and evaluate:

1. `token_mean`: final model LayerNorm applied to the block output, followed by mean over 144 tokens and L2 normalization.
2. `head_projected`: final model LayerNorm and the official flattening feature head applied to a block output, followed by L2 normalization.
3. `probe_ln0_mean`: raw post-block tokens standardized independently over channels by a non-affine LayerNorm with epsilon `1e-5`, then averaged over tokens without L2 normalization. This is a cacheable input for E2, not an embedding by itself.

None should be called an official intermediate identity embedding. `token_mean` was not the training head, `head_projected` applies a head trained for the final block to earlier blocks, and `probe_ln0_mean` is only an unnormalized probe input. The final block's `head_projected` vector is numerically checked against the official forward output.

E2 attaches the same project-designed readout to `probe_ln0_mean` at every block: trainable LayerNorm affine parameters, a bias-free `256→512` projection and L2 normalization, plus a training-only CosFace classifier. The backbone stays frozen and the classifier is discarded after training. This family has 131,584 inference-time trainable parameters per layer and is capacity-matched only across E2 layers; it is neither the official LVFace head nor a reproduction of the LVFace full-model training procedure.

The formal five-seed E2 run selected block 11 on dev, but its failure-aware internal end-to-end TAR advantage over the matched block-12 readout was only `+0.00067`, with 95% identity-bootstrap interval `[0.00000, 0.00200]` and a positive direction in only one of five seeds. The preregistered H2 gate did not pass. See `reports/E2_FEI_FULL_LVFACE_PROBES.md`.

## Planned model ladder

- CVLFace ViT-B AdaFace WebFace4M: stronger 24-layer face-specific replication.
- TransFace-S: face-generation-related comparison because it has been integrated into FaceChain.
- DINOv2 ViT-B/14: generic-vision negative/control backbone.
- Independent AdaFace/CurricularFace-family checkpoint: additional sensitivity evaluator for later generated-image confirmation.
