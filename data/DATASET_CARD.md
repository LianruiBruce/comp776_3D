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

## FEI original set for E1/E2 development

- Contents: 200 identities × 14 color images, 2,800 total, original size 640×480.
- Local use: pose/expression/scale/illumination layer profiling and multi-reference development.
- Assets: four official `originalimages_part*.zip` archives, each pinned by SHA256.
- Split: NumPy `default_rng(seed=20260902)`, 100 train / 50 dev / 50 internal evaluation identities.
- Fixed query slots: `01, 03, 08, 10, 12, 14`.
- Fixed reference slots: `02, 04, 05, 06, 07, 09, 11, 13`.

The slot condition names in the manifest are derived from visual inspection of the consistent acquisition sequence; they are not publisher-supplied pose labels. All FEI identities have been touched during project development, so `internal_eval` is not a fresh external test.

E1 and E2 use the same pinned SCRFD-10G detection and ArcFace five-point alignment. The frozen manifest contains 2,782/2,800 aligned images: 1,394 train, 694 dev and 694 internal evaluation. All 18 failures were no-face detections for slot 14, the lowest-light frontal image; failures remain in the manifest and coverage tables and count as false rejects in E2's end-to-end internal bootstrap. See `reports/E1_FEI_FULL_LVFACE_T.md` and `reports/E2_FEI_FULL_LVFACE_PROBES.md`.

## FEI subset for the PhotoMaker V2 generation pilot

- Scope: exploratory internal diagnostic only; it is not a new test set or external confirmation.
- Identities: 8 fixed identities from the existing `internal_eval` split. All FEI identities had already been touched during development.
- Input snapshot: 56 unique rows, consisting of four generator-reference conditions and three held-out real query conditions for each identity.
- Single-reference condition: `frontal_neutral`.
- Diverse four-reference condition: `frontal_neutral`, `left_profile`, `right_profile` and `frontal_scale`.
- Repeat control: the exact `frontal_neutral` image repeated four times.
- Held-out identity gallery: `frontal_smile`, `left_three_quarter` and `right_three_quarter_expression`.
- Donors: a frozen cyclic mapping among the same eight identities for global-versus-patch channel-conflict trials.

The generator receives the source-resolution copies recorded by `source_relative_path`; automatic identity evaluation uses the separately five-point-aligned `relative_path` query images. The frozen design combines 8 identities, 6 conditions, 2 prompts and 2 seeds, producing 192 generated images. Those images are model outputs, not additional FEI ground truth, and must not be used to enlarge the recognition train/dev sets.

Generated faces can still depict recognizable biometric identity. The source references, generated images, aligned crops, embeddings, human-evaluation media, response JSON files and researcher answer key therefore remain in `artifacts/runs/` or other Git-ignored local paths. They are not public release artifacts and must not be uploaded to GitHub or inserted into reports by default.

An offline v1 identity-likeness master package is prepared but currently has no real participant ratings. It validates blinded media staging, but the formal instrument still needs the preregistered incomplete blocks, attention checks and separate prompt/quality screens. Before recruitment, obtain the applicable institutional approval or exemption, informed consent as required, and freeze assignment/exclusion rules. Give participants only their blinded block; keep the answer key separate. Do not infer sensitive attributes or make demographic subgroup claims from FEI or the generated outputs.

The exact automatic-only interpretation and formal run ID are recorded in `reports/GENERATION_IDENTITY_PILOT.md`.

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
- The eight-identity generation subset is too small and development-exposed for population, fairness or external generalization claims.
- Face-recognition similarity is only an automatic proxy; no perceptual-likeness claim is valid until the blinded human study is completed.

## Planned extensions

1. Extended Yale Face Database B+: controlled pose and illumination external confirmation.
2. DigiFace-1M: larger synthetic identity-disjoint replication.
3. LFW or another licensed real-world set: external sanity check only, not layer selection.
