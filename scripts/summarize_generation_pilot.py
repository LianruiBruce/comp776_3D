from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from id_layers.config import repository_root
from id_layers.generation_metrics import (
    identity_bootstrap_mean,
    paired_identity_bootstrap,
)
from id_layers.utils import sha256_file, write_json

EVALUATORS = ["lvface_independent", "insightface_conditioner"]
METRICS = ["target_similarity", "target_impostor_margin", "end_to_end_rank1"]
CONDITION_CONTRASTS = [
    ("k4_diverse_full", "k1_full", "diverse_k4_minus_k1"),
    ("k4_diverse_full", "k4_repeat_full", "diverse_k4_minus_repeat_k4"),
    ("k4_repeat_full", "k1_full", "repeat_k4_minus_k1"),
    ("k1_full", "text_only", "native_k1_minus_text_only"),
]
NATIVE_CONDITIONS = ["k1_full", "k4_diverse_full", "k4_repeat_full"]
CONFLICT_CONDITIONS = [
    "global_target_patch_donor",
    "global_donor_patch_target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen identity-level contrasts from generation evaluation."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-name", default="evaluation_v2")
    parser.add_argument("--output-name", default="analysis")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    return parser.parse_args()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, group in scored.groupby("condition", sort=True):
        for evaluator in EVALUATORS:
            rows.append(
                {
                    "condition": condition,
                    "evaluator": evaluator,
                    "image_count": int(len(group)),
                    "identity_count": int(group["identity_id"].nunique()),
                    "detection_rate": float(group["face_detected"].astype(bool).mean()),
                    "single_face_rate": float(
                        group["end_to_end_valid"].astype(bool).mean()
                    ),
                    **{
                        metric: float(group[f"{evaluator}__{metric}"].mean())
                        for metric in METRICS
                    },
                }
            )
    return pd.DataFrame(rows)


def _condition_contrasts(
    scored: pd.DataFrame, resamples: int, seed: int
) -> list[dict[str, Any]]:
    results = []
    for evaluator in EVALUATORS:
        for metric in METRICS:
            for condition_a, condition_b, name in CONDITION_CONTRASTS:
                result = paired_identity_bootstrap(
                    scored,
                    identity_column="identity_id",
                    condition_column="condition",
                    value_column=f"{evaluator}__{metric}",
                    condition_a=condition_a,
                    condition_b=condition_b,
                    resamples=resamples,
                    seed=seed,
                )
                results.append(
                    {
                        "contrast_type": "generation_condition",
                        "contrast": name,
                        "evaluator": evaluator,
                        "metric": metric,
                        **result,
                    }
                )
    return results


def _prompt_contrasts(
    scored: pd.DataFrame, resamples: int, seed: int
) -> list[dict[str, Any]]:
    results = []
    for condition in sorted(scored["condition"].unique()):
        subset = scored[scored["condition"] == condition]
        for evaluator in EVALUATORS:
            for metric in METRICS:
                result = paired_identity_bootstrap(
                    subset,
                    identity_column="identity_id",
                    condition_column="prompt_id",
                    value_column=f"{evaluator}__{metric}",
                    condition_a="p1_nuisance",
                    condition_b="p0_easy",
                    resamples=resamples,
                    seed=seed,
                )
                results.append(
                    {
                        "contrast_type": "prompt_within_condition",
                        "condition": condition,
                        "contrast": "p1_nuisance_minus_p0_easy",
                        "evaluator": evaluator,
                        "metric": metric,
                        **result,
                    }
                )
    return results


def _channel_steering(
    scored: pd.DataFrame, resamples: int, seed: int
) -> list[dict[str, Any]]:
    results = []
    for condition in CONFLICT_CONDITIONS:
        subset = scored[scored["condition"] == condition].copy()
        for evaluator in EVALUATORS:
            value = f"{evaluator}__global_minus_patch"
            difference = identity_bootstrap_mean(
                subset,
                identity_column="identity_id",
                value_column=value,
                resamples=resamples,
                seed=seed,
            )
            subset["global_follow"] = (subset[value] > 0).astype(float)
            follow = identity_bootstrap_mean(
                subset,
                identity_column="identity_id",
                value_column="global_follow",
                resamples=resamples,
                seed=seed,
            )
            results.append(
                {
                    "condition": condition,
                    "evaluator": evaluator,
                    "global_minus_patch": difference,
                    "global_follow_rate": follow,
                    "image_count": int(len(subset)),
                }
            )
    return results


def _real_reference_gaps(
    scored: pd.DataFrame,
    baseline: pd.DataFrame,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    for evaluator in EVALUATORS:
        for metric in ["target_similarity", "target_impostor_margin"]:
            value = f"{evaluator}__{metric}"
            real = baseline[["identity_id", value]].copy()
            real["condition"] = "real_k1_reference"
            for condition in NATIVE_CONDITIONS:
                generated = scored.loc[
                    scored["condition"] == condition,
                    ["identity_id", "condition", value],
                ]
                combined = pd.concat([generated, real], ignore_index=True)
                result = paired_identity_bootstrap(
                    combined,
                    identity_column="identity_id",
                    condition_column="condition",
                    value_column=value,
                    condition_a=condition,
                    condition_b="real_k1_reference",
                    resamples=resamples,
                    seed=seed,
                )
                results.append(
                    {
                        "contrast_type": "generated_minus_real_reference",
                        "condition": condition,
                        "evaluator": evaluator,
                        "metric": metric,
                        **result,
                    }
                )
    return results


def main() -> None:
    args = parse_args()
    root = repository_root()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    run_dir = run_dir.resolve()
    evaluation_dir = run_dir / args.evaluation_name
    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "status.json", {"state": "running"})
    try:
        metrics_path = evaluation_dir / "generated_identity_metrics.csv"
        baseline_path = evaluation_dir / "real_reference_baseline.csv"
        geometry_path = evaluation_dir / "cohort_geometry.csv"
        scored = pd.read_csv(metrics_path)
        baseline = pd.read_csv(baseline_path)
        geometry = pd.read_csv(geometry_path)
        aggregate = _aggregate(scored)
        aggregate.to_csv(output_dir / "condition_aggregates.csv", index=False)
        identity_means = scored.groupby(
            ["identity_id", "condition"], as_index=False
        ).agg(
            **{
                f"{evaluator}__{metric}": (f"{evaluator}__{metric}", "mean")
                for evaluator in EVALUATORS
                for metric in METRICS
            }
        )
        identity_means.to_csv(output_dir / "identity_condition_means.csv", index=False)
        summary = {
            "interpretation_scope": (
                "exploratory eight-identity automatic analysis; no human ratings ingested"
            ),
            "automatic_metrics_only": True,
            "condition_aggregates": _records(aggregate),
            "condition_contrasts": _condition_contrasts(
                scored, args.bootstrap_resamples, args.bootstrap_seed
            ),
            "prompt_contrasts": _prompt_contrasts(
                scored, args.bootstrap_resamples, args.bootstrap_seed
            ),
            "channel_steering": _channel_steering(
                scored, args.bootstrap_resamples, args.bootstrap_seed
            ),
            "generated_minus_real_reference": _real_reference_gaps(
                scored,
                baseline,
                args.bootstrap_resamples,
                args.bootstrap_seed,
            ),
            "cohort_geometry": _records(geometry),
            "inputs": {
                "per_image_metrics": {
                    "path": metrics_path.relative_to(run_dir).as_posix(),
                    "sha256": sha256_file(metrics_path),
                },
                "real_reference_baseline": {
                    "path": baseline_path.relative_to(run_dir).as_posix(),
                    "sha256": sha256_file(baseline_path),
                },
                "cohort_geometry": {
                    "path": geometry_path.relative_to(run_dir).as_posix(),
                    "sha256": sha256_file(geometry_path),
                },
            },
            "bootstrap": {
                "unit": "identity",
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "interval": "percentile_95",
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "status.json", {"state": "complete"})
    except Exception as error:
        write_json(
            output_dir / "status.json",
            {"state": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
