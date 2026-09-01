# Artifacts

`cache/`, `tmp/`, and `runs/` are local and Git-ignored.

A complete run follows this contract:

```text
artifacts/runs/<run_id>/
  assets.lock.resolved.yaml
  config.resolved.yaml
  dataset_manifest.csv
  environment.json
  environment.lock.txt
  extraction.json
  frozen_operating_points.json
  metrics_by_layer.csv
  selection.json
  source_snapshot/
  source_snapshot_manifest.json
  summary.json
  plots/
  run.log
  status.json
```

Curate stable findings into `reports/`; never hand-edit completed run artifacts.
