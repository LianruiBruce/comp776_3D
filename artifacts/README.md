# Artifacts

`cache/`, `tmp/`, and `runs/` are local and Git-ignored.

A complete run follows this contract:

```text
artifacts/runs/<run_id>/
  assets.lock.resolved.yaml
  config.resolved.yaml
  dataset_manifest.csv
  preprocess_coverage.csv
  environment.json
  environment.lock.txt
  metrics_by_layer.csv
  metrics_by_condition.csv        # when the dataset defines nuisance/query conditions
  selection.json
  source_snapshot/
  source_snapshot_manifest.json
  summary.json
  plots/
  run.log
  status.json
```

Layer-sweep runs additionally contain `extraction.json`, `frozen_operating_points.json`, `bootstrap_intervals.csv` and optional `embeddings.npz`.

Equal-capacity E2 runs additionally contain:

```text
  development_manifest.csv
  internal_evaluation_manifest.csv
  dev_identity_folds.json
  extraction_development.json
  extraction_internal_frozen.json
  metrics_by_layer_seed.csv
  crossfit_operating_points.csv
  dev_identity_tar.csv
  internal_frozen_metrics.csv
  internal_identity_tar.csv
  selected_vs_final_bootstrap.json
  control_metrics.csv
  training_history.csv
  probe_checkpoints.json
  probe_repeat.json
  access_audit.json
  performance.json
  checkpoints/
  probe_inputs_*.npz             # only when save_embeddings=true
  frozen_embeddings.npz          # only when save_embeddings=true
```

For E2, the official result columns are the dev-calibrated `frozen_*` fields plus the failure-aware `end_to_end_tar`. Unprefixed internal ROC/EER/TAR fields are descriptive post-hoc diagnostics and must not be used for model selection or the H2 claim.

PhotoMaker V2 generation runs use a separate output contract:

```text
artifacts/runs/<generation_run_id>/
  assets.lock.resolved.yaml
  config.resolved.yaml
  dataset_manifest.csv             # frozen 56-row input subset
  environment.json
  environment.lock.txt
  source_snapshot/
  source_snapshot_manifest.json
  generation_manifest.jsonl        # one row per planned output, including routes/hashes/status
  performance.json
  run_events.jsonl
  images/                           # identifiable generated faces; never commit
  status.json
  evaluation/                       # name is selected by --evaluation-name
    generated_identity_metrics.csv
    metrics_by_condition_prompt.csv
    real_reference_baseline.csv
    cohort_geometry.csv
    paired_bootstrap.json
    generation_validation.json
    query_coverage.csv
    evaluation_embeddings.npz
    *_aligned/                      # private derived face crops
    summary.json
    status.json
  analysis/                         # selected by --output-name
    condition_aggregates.csv
    identity_condition_means.csv
    summary.json
    status.json
  human_eval/
    participant/
      index.html
      blinded_trials.json
      media/                        # contains identifiable faces; never commit
    private/
      answer_key.json               # researcher-only; never give to raters
    responses/                      # present only after approved data collection
    analysis.json                   # present only after response analysis
```

The verified formal generation run is `artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot`: 192/192 images completed, with `evaluation_v3/` and `analysis_v3/` as the latest automatic-analysis entry points. Its `evaluation/` directory is a preserved failed validator attempt, not a valid result; the first successful `evaluation_v2/` has byte-identical SHA256 values to v3 for all nine numerical/validation core artifacts. A blinded v1 identity-likeness master package is built, but it is not yet the block-assigned formal instrument and no real response files have been collected, so the root status remains `automatic_evaluation_complete_human_pending`.

Evaluation/alignment crops, embeddings, generated images, blinded media, participant responses and the private key all remain inside the Git-ignored run. Only aggregate, non-identifying conclusions may be manually curated into `reports/` after license, privacy and institutional-review requirements are satisfied.

Curate stable findings into `reports/`; never hand-edit completed run artifacts.
