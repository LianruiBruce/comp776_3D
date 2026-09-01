# Dataset card

## FEI manually aligned frontal subset

- Publisher: FEI Artificial Intelligence Laboratory, Brazil.
- Homepage: <https://fei.edu.br/~cet/facedatabase.html>
- Local use: pipeline smoke experiment only.
- Contents used: 200 identities, two frontal manually aligned RGB images per identity. The `a` and `b` suffixes are recorded as neutral and smile conditions.
- Original image size observed locally: 260×360.
- Terms: the publisher states that the database is available for research purposes and restricts reproduction/distribution. Treat it as research-only.

Consequences for this repository:

- ZIP archives and extracted images are Git-ignored.
- No face thumbnails are committed.
- The bootstrap script requires explicit acknowledgement of research terms.
- Only provenance, checksums, manifest schema and aggregate results may be versioned.
- Publication of example faces must be reviewed against the publisher's current terms.

The immutable URLs and archive SHA256 values are in `configs/assets.lock.yaml`.

## Split protocol

Identity IDs are shuffled once using NumPy `default_rng(seed=776)` and assigned as:

- dev: 120 identities, used for representation/layer selection;
- val: 40 identities, used for stability checks and any future threshold tuning;
- test: 40 identities, report-only.

Both images of one identity always remain in the same split. The generated manifest records sample ID, identity ID, condition, split, relative path, SHA256, source and alignment status. No face detector is run for this publisher-aligned smoke subset, so `face_detection_attempted=false` and `face_detected` is null rather than being reported as a successful detection.

## Known limitations

- Only two conditions per identity are available.
- Background and acquisition conditions are highly controlled.
- Images are manually aligned but not transformed to a standard five-landmark ArcFace template by this smoke pipeline.
- Direct 112×112 resize follows the upstream LVFace demo behavior but changes aspect ratio.
- Therefore this dataset cannot establish pose, illumination, occlusion or in-the-wild robustness.

## Planned extensions

1. FEI original: all 2,800 images for broader pose/expression/illumination variation.
2. Yale Face Database B: controlled pose and illumination metadata.
3. DigiFace-1M: larger synthetic identity-disjoint replication.
4. LFW or another licensed real-world set: external sanity check only, not layer selection.
