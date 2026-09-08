"""Continuous MediaPipe-native pose diagnostics with circular angle handling.

These are uncalibrated Euler-coordinate proxies, not physical pose ground truth.
Signed changes wrap to [-180, 180); bootstrap intervals are unwrapped around
their circular point estimate and may legitimately cross the +/-180 boundary.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .auto_analysis import ROOT, _project_path, _read


def wrap_degrees(value):
    return (np.asarray(value, dtype=float) + 180.0) % 360.0 - 180.0


def circular_mean(values) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=float)
    if np.isinf(array).any():
        raise ValueError("Infinite angle")
    array = array[np.isfinite(array)]
    if not len(array):
        return None, None
    z = np.exp(1j * np.deg2rad(array)).mean()
    length = float(abs(z))
    if length < 1e-8:
        return None, length
    return float(wrap_degrees(np.rad2deg(np.angle(z)))), length


def circular_identity_interval(
    values: pd.Series, *, seed: int = 20260905, resamples: int = 2000
) -> dict[str, Any]:
    if not values.index.is_unique or resamples < 1:
        raise ValueError("Need unique identity means and positive resample count")
    values = values.sort_index()
    valid = values.dropna().to_numpy(dtype=float)
    estimate, length = circular_mean(valid)
    result = {
        "identity_count": len(values),
        "valid_identity_count": len(valid),
        "estimate_degrees": estimate,
        "resultant_length": length,
        "ci95_lower_centered_degrees": None,
        "ci95_upper_centered_degrees": None,
        "valid_bootstrap_replicates": 0,
    }
    if estimate is not None and len(valid) >= 2:
        rng = np.random.default_rng(seed)
        draws = valid[rng.integers(0, len(valid), (resamples, len(valid)))]
        z = np.exp(1j * np.deg2rad(draws)).mean(axis=1)
        good = abs(z) >= 1e-8
        if not good.any():
            return result
        changes = wrap_degrees(np.rad2deg(np.angle(z[good])) - estimate)
        low, high = np.quantile(changes, [0.025, 0.975])
        result.update(
            ci95_lower_centered_degrees=float(estimate + low),
            ci95_upper_centered_degrees=float(estimate + high),
            valid_bootstrap_replicates=int(good.sum()),
        )
    return result


def _summarize(frame: pd.DataFrame, groups: list[str], value: str, resamples: int, seed: int):
    identities, summaries = [], []
    for labels, rows in frame.groupby(groups, sort=True, dropna=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        tags = dict(zip(groups, labels, strict=True))
        means = {}
        for identity, part in rows.groupby("identity_id", sort=True):
            estimate, length = circular_mean(part[value])
            means[identity] = estimate
            identities.append(
                {
                    **tags,
                    "identity_id": identity,
                    "estimate_degrees": estimate,
                    "resultant_length": length,
                    "row_count": len(part),
                    "valid_row_count": int(part[value].notna().sum()),
                }
            )
        summaries.append(
            {
                **tags,
                "row_count": len(rows),
                "valid_row_count": int(rows[value].notna().sum()),
                **circular_identity_interval(
                    pd.Series(means, dtype=float), seed=seed, resamples=resamples
                ),
            }
        )
    return pd.DataFrame(identities), pd.DataFrame(summaries)


def analyze_pose_collection(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    new_expression_protocol: bool = False,
    resamples: int = 2000,
    seed: int = 20260905,
) -> dict[str, Any]:
    evaluation_dir, output_dir = _project_path(evaluation_dir), _project_path(output_dir)
    protected = ROOT / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot"
    if output_dir.is_relative_to(evaluation_dir) or output_dir.is_relative_to(protected):
        raise ValueError("Pose analysis cannot modify source experiments")
    if output_dir.exists():
        raise FileExistsError("Pose analysis output already exists")
    landmarks = _read(evaluation_dir / "landmarks.csv", required=True)
    if landmarks.sample_id.duplicated().any():
        raise ValueError("Duplicate landmark sample ID")
    metadata_columns = ["sample_id", "identity_id", "condition", "prompt_id", "base_seed"]
    metadata = _read(evaluation_dir / "identity_metrics.csv", required=True)[metadata_columns]
    metadata = metadata.drop_duplicates()
    if metadata.sample_id.duplicated().any():
        raise ValueError("Recognizer metadata disagree for a generated sample")
    start = time.perf_counter()
    rows = []
    for source in landmarks.to_dict("records"):
        error, values = None, [None, None, None]
        if source.get("status") != "complete":
            error = source.get("error", "landmarker_failed")
        else:
            try:
                values = json.loads(source["pose_extrinsic_xyz_degrees"])
                if len(values) != 3 or not np.isfinite(np.asarray(values, dtype=float)).all():
                    raise ValueError("Need three finite Euler angles")
                values = wrap_degrees(values).tolist()
            except (TypeError, ValueError) as exc:
                error, values = f"invalid_pose:{type(exc).__name__}", [None, None, None]
        rows.append(
            {
                **source,
                "pose_x_degrees": values[0],
                "pose_y_degrees": values[1],
                "pose_z_degrees": values[2],
                "pose_parse_error": error,
            }
        )
    frame = pd.DataFrame(rows)
    generated = metadata.merge(
        frame.drop(columns=[c for c in metadata_columns if c != "sample_id" and c in frame]),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    generated["kind"] = "generated"
    real = frame.loc[~frame.sample_id.isin(metadata.sample_id)].copy()
    real["prompt_id"] = "real_photo"
    all_images = pd.concat([generated, real], ignore_index=True)
    long = all_images.melt(
        id_vars=["sample_id", "identity_id", "kind", "condition", "prompt_id"],
        value_vars=["pose_x_degrees", "pose_y_degrees", "pose_z_degrees"],
        var_name="axis",
        value_name="angle_degrees",
    )
    identity, summary = _summarize(
        long, ["kind", "condition", "prompt_id", "axis"], "angle_degrees", resamples, seed
    )
    tables = {"pose_per_image": all_images, "pose_identity": identity, "pose_summary": summary}
    if new_expression_protocol:
        if set(metadata.prompt_id) != {"neutral", "smile"}:
            raise ValueError("Pose contrast requires the frozen neutral/smile protocol")
        pair_keys = ["identity_id", "condition", "base_seed"]
        if generated.duplicated([*pair_keys, "prompt_id"]).any():
            raise ValueError("Duplicate expression pose cell")
        pairs = []
        for axis in ("pose_x_degrees", "pose_y_degrees", "pose_z_degrees"):
            matrix = generated.pivot(index=pair_keys, columns="prompt_id", values=axis)
            matrix = matrix.reindex(columns=["neutral", "smile"])
            matrix["difference_degrees"] = wrap_degrees(matrix.smile - matrix.neutral)
            matrix["axis"] = axis
            pairs.extend(matrix.reset_index().to_dict("records"))
        pairs = pd.DataFrame(pairs)
        identity, summary = _summarize(pairs, ["axis"], "difference_degrees", resamples, seed)
        tables.update(
            pose_expression_pairs=pairs,
            pose_expression_identity=identity,
            pose_expression_summary=summary,
        )
    output_dir.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output_dir / (name + ".csv"), index=False)
    result = {
        "status": "complete",
        "evidence_type": "DEVELOPMENT_AUTOMATIC_POSE_PROXY",
        "source_directory": str(evaluation_dir),
        "new_expression_protocol": new_expression_protocol,
        "source_sample_count": len(all_images),
        "generated_sample_count": len(generated),
        "valid_generated_pose_count": int(generated.pose_x_degrees.notna().sum()),
        "seed": seed,
        "resamples": resamples,
        "axis_convention": (
            "MediaPipe native closest-proper-rotation extrinsic XYZ Euler coordinates; "
            "uncalibrated, not physical pose ground truth"
        ),
        "circular_rule": (
            "Within-identity circular mean, identity-equal circular grand mean; paired shortest "
            "signed differences [-180,180); bootstrap CI unwrapped around point estimate"
        ),
        "ambiguity_rule": (
            "Resultant length <1e-8 gives undefined mean, retained as missing; "
            "no chosen pose-success threshold"
        ),
        "source_sha256": {
            name: hashlib.sha256((evaluation_dir / name).read_bytes()).hexdigest()
            for name in ("landmarks.csv", "identity_metrics.csv")
        },
        "cpu_wall_seconds": time.perf_counter() - start,
    }
    for name in ("summary.json", "status.json", "resolved_config.json"):
        (output_dir / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
