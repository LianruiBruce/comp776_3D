# Third-party cache

- `LVFace/` is the pinned, untracked LVFace model-code checkout.
- `InsightFace/` is the pinned, untracked detection/alignment-code checkout used by E1.

Both official repositories and exact commits are recorded in `configs/assets.lock.yaml`. Run `scripts/bootstrap_assets.py --accept-research-terms --profile e1` to create or verify both.

Do not modify the cached checkout. Any required adaptation belongs in `src/id_layers/` with a clear upstream reference.
