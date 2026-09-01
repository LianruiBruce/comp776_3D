# Model card

## LVFace-T Glint360K

- Paper/repository: <https://github.com/bytedance/LVFace>
- Official weights: <https://huggingface.co/bytedance-research/LVFace>
- Architecture: face-recognition Vision Transformer, 12 blocks, 144 spatial tokens, hidden size 256, final embedding 512.
- Parameters verified locally: 19,137,792.
- Input: aligned RGB face, 112×112, normalized channel-wise by `(x - 0.5) / 0.5` after conversion to `[0, 1]`.
- There is no CLS token.

Pinned code revision, weight revision and SHA256 are in `configs/assets.lock.yaml`. The code is MIT licensed. The upstream repository describes the pretrained weights as non-commercial research assets; weights are never committed here.

## Intermediate representations

The smoke experiment registers read-only forward hooks on `model.blocks[0:12]` and evaluates:

1. `token_mean`: final model LayerNorm applied to the block output, followed by mean over 144 tokens and L2 normalization.
2. `head_projected`: final model LayerNorm and the official flattening feature head applied to a block output, followed by L2 normalization.

Neither construction should be called an official intermediate identity embedding. `token_mean` was not the training head, and `head_projected` applies a head trained for the final block to earlier blocks. The final block's `head_projected` vector is numerically checked against the official forward output.

## Planned model ladder

- CVLFace ViT-B AdaFace WebFace4M: stronger 24-layer face-specific replication.
- TransFace-S: face-generation-related comparison because it has been integrated into FaceChain.
- DINOv2 ViT-B/14: generic-vision negative/control backbone.
- Independent ArcFace/AdaFace/CurricularFace checkpoint: evaluation only for later generated images.
