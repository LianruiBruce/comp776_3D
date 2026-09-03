from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from id_layers.config import repository_root
from id_layers.generation_evaluation import (
    InsightFaceGeneratedFaceEncoder,
    LVFaceFinalEncoder,
)
from id_layers.generation_metrics import (
    build_identity_templates,
    cohort_geometry,
    gallery_identity_metrics,
    paired_identity_bootstrap,
)
from id_layers.utils import (
    environment_metadata,
    python_package_lock,
    sha256_file,
    snapshot_research_sources,
    write_json,
)

QUERY_CONDITIONS = [
    "frontal_smile",
    "left_three_quarter",
    "right_three_quarter_expression",
]
CONTRASTS = [
    ("k4_diverse_full", "k1_full"),
    ("k4_diverse_full", "k4_repeat_full"),
    ("k1_full", "text_only"),
]
EVALUATORS = ["lvface_independent", "insightface_conditioner"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PhotoMaker V2 generation pilot without dropping failures."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/manifests/fei_full_seed20260902.csv"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="Run-relative evaluation directory; use a new name after a failed attempt.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"No generation records in {path}")
    frame = pd.DataFrame(records)
    required = {
        "sample_id",
        "identity_id",
        "donor_identity_id",
        "condition",
        "prompt_id",
        "base_seed",
        "effective_seed",
        "output_relative_path",
        "status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Generation manifest is missing fields: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Generation sample_id values must be unique")
    return frame


def _resolve_inside(base: Path, value: str) -> Path:
    target = (base / value).resolve()
    if target != base.resolve() and base.resolve() not in target.parents:
        raise ValueError(f"Path escapes expected root {base}: {value}")
    return target


def _validate_generation_matrix(
    generated: pd.DataFrame, run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    identities = list(config["data"]["identities"])
    conditions = [item["name"] for item in config["generation"]["conditions"]]
    prompts = list(config["generation"]["prompts"])
    seeds = [int(value) for value in config["generation"]["seeds"]]
    expected_cells = {
        (identity, condition, prompt, seed)
        for identity in identities
        for condition in conditions
        for prompt in prompts
        for seed in seeds
    }
    observed_cells = {
        (str(row.identity_id), str(row.condition), str(row.prompt_id), int(row.base_seed))
        for row in generated.itertuples(index=False)
    }
    if len(generated) != len(expected_cells) or observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        unexpected = sorted(observed_cells - expected_cells)
        raise RuntimeError(
            "Generation factorial is incomplete or contaminated: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if generated["status"].tolist().count("complete") != len(generated):
        raise RuntimeError("Every formal generation row must have status=complete")

    latent_groups = generated.groupby(["identity_id", "base_seed"])
    bad_latent_groups = []
    for key, group in latent_groups:
        if (
            group["latent_sha256"].nunique() != 1
            or group["effective_seed"].nunique() != 1
            or len(group) != len(conditions) * len(prompts)
        ):
            bad_latent_groups.append(key)
    if bad_latent_groups:
        raise RuntimeError(f"Unpaired latent groups: {bad_latent_groups}")
    for ordinal, identity in enumerate(identities):
        expected_effective = {seed + ordinal * 100_000 for seed in seeds}
        observed_effective = set(
            generated.loc[generated["identity_id"] == identity, "effective_seed"].astype(int)
        )
        if observed_effective != expected_effective:
            raise RuntimeError(f"Effective-seed schedule mismatch for {identity}")

    route_expectations = {
        "k1_full": ("target", "target", 1),
        "k4_diverse_full": ("target", "target", 4),
        "k4_repeat_full": ("target", "target", 4),
        "global_target_patch_donor": ("target", "donor", 1),
        "global_donor_patch_target": ("donor", "target", 1),
        "text_only": (None, None, 1),
    }
    verified_outputs = 0
    for row in generated.itertuples(index=False):
        global_role, patch_role, reference_count = route_expectations[row.condition]
        target = str(row.identity_id)
        donor = str(row.donor_identity_id)
        expected_global = (
            target if global_role == "target" else donor if global_role == "donor" else None
        )
        expected_patch = (
            target if patch_role == "target" else donor if patch_role == "donor" else None
        )
        observed_global = (
            None
            if pd.isna(row.global_embedding_identity_id)
            else str(row.global_embedding_identity_id)
        )
        observed_patch = (
            None
            if pd.isna(row.patch_image_identity_id)
            else str(row.patch_image_identity_id)
        )
        if (
            observed_global != expected_global
            or observed_patch != expected_patch
        ):
            raise RuntimeError(f"Identity-channel route mismatch for {row.sample_id}")
        if (
            len(row.global_reference_sample_ids) != reference_count
            or len(row.patch_reference_sample_ids) != reference_count
        ):
            raise RuntimeError(f"Reference-count mismatch for {row.sample_id}")
        if row.condition == "k4_repeat_full" and (
            len(set(row.global_reference_sample_ids)) != 1
            or len(set(row.patch_reference_sample_ids)) != 1
        ):
            raise RuntimeError(f"K4 repeat is not exact for {row.sample_id}")
        output = _resolve_inside(run_dir, str(row.output_relative_path))
        if not output.is_file() or sha256_file(output) != str(row.output_sha256):
            raise RuntimeError(f"Generated output hash mismatch for {row.sample_id}")
        verified_outputs += 1
    return {
        "status": "passed",
        "expected_cells": len(expected_cells),
        "observed_unique_cells": len(observed_cells),
        "complete_rows": int((generated["status"] == "complete").sum()),
        "paired_latent_groups": int(len(latent_groups)),
        "verified_output_hashes": verified_outputs,
        "identity_count": len(identities),
        "condition_count": len(conditions),
        "prompt_count": len(prompts),
        "base_seed_count": len(seeds),
    }


def _make_face_encoder(root: Path, lock: dict[str, Any]) -> InsightFaceGeneratedFaceEncoder:
    insightface = lock["models"]["insightface_buffalo_l_scrfd10g"]
    pack = root / "artifacts/cache/models/insightface/models/buffalo_l"
    return InsightFaceGeneratedFaceEncoder(
        checkout=root / "third_party/InsightFace",
        cache_root=root / "artifacts/cache/models/insightface",
        model_name="buffalo_l",
        detector_path=pack / insightface["detector_filename"],
        recognition_path=pack / insightface["recognition_filename"],
        locked_model=insightface,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def _make_lvface_encoder(root: Path, lock: dict[str, Any]) -> LVFaceFinalEncoder:
    return LVFaceFinalEncoder(
        {
            "architecture": "vit_t",
            "third_party_path": str(root / "third_party/LVFace"),
            "weights_path": str(
                root
                / "artifacts/cache/models/LVFace/LVFace-T_Glint360K/"
                "LVFace-T_Glint360K.pt"
            ),
        },
        lock["models"]["lvface_t_glint360k"],
    )


def _query_embeddings(
    *,
    root: Path,
    evaluation_dir: Path,
    dataset: pd.DataFrame,
    identity_ids: list[str],
    face_encoder: InsightFaceGeneratedFaceEncoder,
    lvface_encoder: LVFaceFinalEncoder,
) -> tuple[dict[str, tuple[np.ndarray, list[str]]], pd.DataFrame]:
    query = dataset[
        dataset["identity_id"].isin(identity_ids)
        & dataset["condition"].isin(QUERY_CONDITIONS)
    ].copy()
    counts = query.groupby("identity_id")["condition"].nunique()
    missing = [identity for identity in identity_ids if counts.get(identity, 0) != 3]
    if missing:
        raise ValueError(f"Each identity needs all three held-out query conditions: {missing}")
    query = query.sort_values(["identity_id", "condition"]).reset_index(drop=True)

    conditioner_values: list[np.ndarray] = []
    lvface_paths: list[Path] = []
    coverage: list[dict[str, Any]] = []
    for row in query.itertuples(index=False):
        source = _resolve_inside(root / "data", str(row.source_relative_path))
        # Manifest paths are relative to data/, e.g. processed/fei_full/source/...
        aligned = _resolve_inside(root / "data", str(row.relative_path))
        diagnostic_target = (
            evaluation_dir / "query_conditioner_aligned" / f"{row.sample_id}.png"
        )
        embedding, metadata = face_encoder.encode_and_align(source, diagnostic_target)
        coverage.append(
            {
                "sample_id": row.sample_id,
                "identity_id": row.identity_id,
                "condition": row.condition,
                **metadata,
            }
        )
        if embedding is None:
            raise RuntimeError(
                f"Pinned conditioner failed on held-out real query {row.sample_id}; "
                "the gallery cannot be constructed"
            )
        conditioner_values.append(embedding)
        lvface_paths.append(aligned)

    labels = query["identity_id"].tolist()
    conditioner_templates, conditioner_order = build_identity_templates(
        np.stack(conditioner_values), labels, identity_ids
    )
    lvface_values = lvface_encoder.encode_paths(lvface_paths)
    lvface_templates, lvface_order = build_identity_templates(
        lvface_values, labels, identity_ids
    )
    if conditioner_order != lvface_order:
        raise RuntimeError("Evaluator galleries have inconsistent identity ordering")
    return {
        "insightface_conditioner": (conditioner_templates, conditioner_order),
        "lvface_independent": (lvface_templates, lvface_order),
    }, pd.DataFrame(coverage)


def _generated_embeddings(
    *,
    run_dir: Path,
    evaluation_dir: Path,
    generated: pd.DataFrame,
    face_encoder: InsightFaceGeneratedFaceEncoder,
    lvface_encoder: LVFaceFinalEncoder,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    metadata_rows: list[dict[str, Any]] = []
    conditioner_values: list[np.ndarray] = []
    valid_indices: list[int] = []
    aligned_paths: list[Path] = []
    for index, row in generated.iterrows():
        if str(row["status"]) not in {"generated", "complete"}:
            metadata_rows.append(
                {
                    "generation_row": int(index),
                    "face_detected": False,
                    "detected_face_count": 0,
                    "alignment_status": "failed",
                    "alignment_failure_reason": f"generation_status:{row['status']}",
                    "end_to_end_valid": False,
                }
            )
            continue
        source = _resolve_inside(run_dir, str(row["output_relative_path"]))
        if not source.is_file():
            raise FileNotFoundError(f"Generated output is missing: {source}")
        aligned_target = evaluation_dir / "generated_aligned" / f"{row['sample_id']}.png"
        embedding, metadata = face_encoder.encode_and_align(source, aligned_target)
        metadata_rows.append({"generation_row": int(index), **metadata})
        if embedding is not None:
            conditioner_values.append(embedding)
            valid_indices.append(int(index))
            aligned_paths.append(aligned_target)

    metadata = pd.DataFrame(metadata_rows).set_index("generation_row")
    if not valid_indices:
        raise RuntimeError("No generated image contained a detectable face")
    lvface_values = lvface_encoder.encode_paths(aligned_paths)
    return (
        metadata,
        {
            "insightface_conditioner": np.stack(conditioner_values),
            "lvface_independent": lvface_values,
        },
        np.asarray(valid_indices, dtype=np.int64),
    )


def _real_reference_baseline(
    *,
    root: Path,
    evaluation_dir: Path,
    dataset: pd.DataFrame,
    generated: pd.DataFrame,
    identity_ids: list[str],
    face_encoder: InsightFaceGeneratedFaceEncoder,
    lvface_encoder: LVFaceFinalEncoder,
    galleries: dict[str, tuple[np.ndarray, list[str]]],
) -> pd.DataFrame:
    """Score the real K1 inputs against held-out queries as a contextual ceiling."""
    references = dataset[
        dataset["identity_id"].isin(identity_ids)
        & (dataset["condition"] == "frontal_neutral")
    ].copy()
    if len(references) != len(identity_ids):
        raise ValueError("Expected exactly one frontal-neutral real reference per identity")
    donor_by_identity = (
        generated.groupby("identity_id")["donor_identity_id"].nunique().to_dict()
    )
    if any(count != 1 for count in donor_by_identity.values()):
        raise ValueError("Each target identity must have exactly one frozen donor")
    donor_lookup = (
        generated.groupby("identity_id")["donor_identity_id"].first().to_dict()
    )
    references = references.sort_values("identity_id").reset_index(drop=True)
    conditioner_values: list[np.ndarray] = []
    lvface_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for row in references.itertuples(index=False):
        source = _resolve_inside(root / "data", str(row.source_relative_path))
        aligned = _resolve_inside(root / "data", str(row.relative_path))
        output = evaluation_dir / "real_reference_aligned" / f"{row.sample_id}.png"
        embedding, metadata = face_encoder.encode_and_align(source, output)
        if embedding is None or not bool(metadata["end_to_end_valid"]):
            raise RuntimeError(f"Real K1 reference is not a single valid face: {row.sample_id}")
        conditioner_values.append(embedding)
        lvface_paths.append(aligned)
        rows.append(
            {
                "sample_id": row.sample_id,
                "identity_id": row.identity_id,
                "donor_identity_id": donor_lookup[row.identity_id],
                **metadata,
            }
        )
    baseline = pd.DataFrame(rows)
    embeddings = {
        "insightface_conditioner": np.stack(conditioner_values),
        "lvface_independent": lvface_encoder.encode_paths(lvface_paths),
    }
    for evaluator in EVALUATORS:
        templates, template_ids = galleries[evaluator]
        metrics = gallery_identity_metrics(
            embeddings[evaluator],
            baseline["identity_id"].tolist(),
            templates,
            template_ids,
            donor_ids=baseline["donor_identity_id"].tolist(),
            end_to_end_valid=[True] * len(baseline),
        )
        for column in metrics:
            baseline[f"{evaluator}__{column}"] = metrics[column]
    return baseline


def _attach_scores(
    generated: pd.DataFrame,
    detection: pd.DataFrame,
    valid_indices: np.ndarray,
    generated_embeddings: dict[str, np.ndarray],
    galleries: dict[str, tuple[np.ndarray, list[str]]],
) -> pd.DataFrame:
    output = generated.join(detection, how="left")
    for evaluator in EVALUATORS:
        templates, template_ids = galleries[evaluator]
        valid = output.loc[valid_indices]
        metrics = gallery_identity_metrics(
            generated_embeddings[evaluator],
            valid["identity_id"].tolist(),
            templates,
            template_ids,
            donor_ids=valid["donor_identity_id"].tolist(),
            end_to_end_valid=valid["end_to_end_valid"].astype(bool).tolist(),
        )
        metrics.index = valid_indices
        for column in metrics:
            output[f"{evaluator}__{column}"] = metrics[column]
        output[f"{evaluator}__aligned_rank1"] = output[
            f"{evaluator}__aligned_rank1"
        ].fillna(False)
        output[f"{evaluator}__end_to_end_rank1"] = output[
            f"{evaluator}__end_to_end_rank1"
        ].fillna(False)

        target_minus_donor = output[f"{evaluator}__target_minus_donor"]
        output[f"{evaluator}__global_minus_patch"] = np.where(
            output["condition"] == "global_target_patch_donor",
            target_minus_donor,
            np.where(
                output["condition"] == "global_donor_patch_target",
                -target_minus_donor,
                np.nan,
            ),
        )
    return output


def _aggregate_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, prompt_id), group in scored.groupby(["condition", "prompt_id"]):
        base = {
            "condition": condition,
            "prompt_id": prompt_id,
            "generated_count": int(len(group)),
            "face_detection_rate": float(group["face_detected"].fillna(False).mean()),
            "single_face_rate": float(group["end_to_end_valid"].fillna(False).mean()),
        }
        for evaluator in EVALUATORS:
            global_minus_patch = group[f"{evaluator}__global_minus_patch"]
            rows.append(
                {
                    **base,
                    "evaluator": evaluator,
                    "aligned_rank1": float(group[f"{evaluator}__aligned_rank1"].mean()),
                    "end_to_end_rank1": float(
                        group[f"{evaluator}__end_to_end_rank1"].mean()
                    ),
                    "target_similarity": float(
                        group[f"{evaluator}__target_similarity"].mean()
                    ),
                    "target_impostor_margin": float(
                        group[f"{evaluator}__target_impostor_margin"].mean()
                    ),
                    "target_minus_donor": float(
                        group[f"{evaluator}__target_minus_donor"].mean()
                    ),
                    "global_minus_patch": float(
                        global_minus_patch.mean()
                    ),
                    "global_follow_rate": float((global_minus_patch.dropna() > 0).mean())
                    if bool(global_minus_patch.notna().any())
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_contrasts(
    scored: pd.DataFrame, resamples: int, seed: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for evaluator in EVALUATORS:
        for value_name in [
            "target_impostor_margin",
            "end_to_end_rank1",
            "target_similarity",
        ]:
            value_column = f"{evaluator}__{value_name}"
            for condition_a, condition_b in CONTRASTS:
                result = paired_identity_bootstrap(
                    scored,
                    identity_column="identity_id",
                    condition_column="condition",
                    value_column=value_column,
                    condition_a=condition_a,
                    condition_b=condition_b,
                    resamples=resamples,
                    seed=seed,
                )
                results.append({"evaluator": evaluator, "metric": value_name, **result})
    return results


def _geometry_rows(
    scored: pd.DataFrame,
    valid_indices: np.ndarray,
    generated_embeddings: dict[str, np.ndarray],
    galleries: dict[str, tuple[np.ndarray, list[str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_frame = scored.loc[valid_indices].copy()
    valid_frame["embedding_row"] = np.arange(len(valid_frame))
    for evaluator in EVALUATORS:
        real_templates, template_ids = galleries[evaluator]
        real_lookup = {identity: real_templates[i] for i, identity in enumerate(template_ids)}
        for condition, group in valid_frame.groupby("condition"):
            identities = sorted(group["identity_id"].unique())
            centroids: list[np.ndarray] = []
            comparable_ids: list[str] = []
            for identity in identities:
                subset = group[group["identity_id"] == identity]
                embedding_rows = subset["embedding_row"].to_numpy(dtype=np.int64)
                centroid = generated_embeddings[evaluator][embedding_rows].mean(axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm > 0:
                    centroids.append(centroid / norm)
                    comparable_ids.append(identity)
            if len(centroids) < 2:
                continue
            generated_geometry = cohort_geometry(np.stack(centroids))
            comparable_real = np.stack([real_lookup[identity] for identity in comparable_ids])
            real_geometry = cohort_geometry(comparable_real)
            real_cohort_centroid = comparable_real.mean(axis=0)
            real_cohort_centroid /= np.linalg.norm(real_cohort_centroid)
            generated_cohort_attraction = float(
                np.mean(np.stack(centroids) @ real_cohort_centroid)
            )
            real_cohort_attraction = float(
                np.mean(comparable_real @ real_cohort_centroid)
            )
            rows.append(
                {
                    "condition": condition,
                    "evaluator": evaluator,
                    "identity_count": generated_geometry.identity_count,
                    "generated_between_identity_spread": (
                        generated_geometry.between_identity_spread
                    ),
                    "real_between_identity_spread": real_geometry.between_identity_spread,
                    "contraction_ratio": (
                        generated_geometry.between_identity_spread
                        / real_geometry.between_identity_spread
                    ),
                    "generated_mean_impostor_similarity": (
                        generated_geometry.mean_impostor_similarity
                    ),
                    "real_mean_impostor_similarity": real_geometry.mean_impostor_similarity,
                    "generated_effective_rank": (
                        generated_geometry.covariance_effective_rank
                    ),
                    "real_effective_rank": real_geometry.covariance_effective_rank,
                    "generated_real_cohort_centroid_similarity": (
                        generated_cohort_attraction
                    ),
                    "real_real_cohort_centroid_similarity": real_cohort_attraction,
                    "cohort_centroid_attraction_shift": (
                        generated_cohort_attraction - real_cohort_attraction
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    root = repository_root()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    run_dir = run_dir.resolve()
    run_status_path = run_dir / "status.json"
    run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
    allowed_states = {
        "generation_complete_evaluation_pending",
        "automatic_evaluation_failed",
        "automatic_evaluation_complete_human_pending",
    }
    if run_status.get("state") not in allowed_states:
        raise RuntimeError(
            "Automatic evaluation requires generation pending or a recorded failed attempt; "
            f"observed {run_status.get('state')!r}"
        )
    manifest_path = run_dir / "generation_manifest.jsonl"
    generated = _load_jsonl(manifest_path)
    resolved_generation_config = yaml.safe_load(
        (run_dir / "config.resolved.yaml").read_text(encoding="utf-8")
    )
    expected_generated = int(
        resolved_generation_config["experiment"]["expected_generated_images"]
    )
    if len(generated) != expected_generated:
        raise RuntimeError(
            f"Expected {expected_generated} generated records, observed {len(generated)}"
        )
    if Path(args.evaluation_name).name != args.evaluation_name or args.evaluation_name in {
        ".",
        "..",
    }:
        raise ValueError("evaluation-name must be one path-safe directory name")
    evaluation_dir = run_dir / args.evaluation_name
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("generation_evaluation")
    write_json(evaluation_dir / "status.json", {"state": "running"})

    try:
        evaluation_source_manifest = snapshot_research_sources(
            root, evaluation_dir / "source_snapshot"
        )
        write_json(
            evaluation_dir / "source_snapshot_manifest.json", evaluation_source_manifest
        )
        write_json(evaluation_dir / "environment.json", environment_metadata(root))
        (evaluation_dir / "environment.lock.txt").write_text(
            python_package_lock(), encoding="utf-8"
        )
        generation_validation = _validate_generation_matrix(
            generated, run_dir, resolved_generation_config
        )
        write_json(evaluation_dir / "generation_validation.json", generation_validation)
        lock = yaml.safe_load((root / "configs/assets.lock.yaml").read_text(encoding="utf-8"))
        dataset_path = (
            args.dataset_manifest
            if args.dataset_manifest.is_absolute()
            else root / args.dataset_manifest
        )
        dataset = pd.read_csv(dataset_path)
        identity_ids = sorted(generated["identity_id"].unique().tolist())
        logger.info("Loading pinned conditioner and independent evaluator")
        face_encoder = _make_face_encoder(root, lock)
        lvface_encoder = _make_lvface_encoder(root, lock)
        logger.info("Building held-out real-query galleries for %d identities", len(identity_ids))
        galleries, query_coverage = _query_embeddings(
            root=root,
            evaluation_dir=evaluation_dir,
            dataset=dataset,
            identity_ids=identity_ids,
            face_encoder=face_encoder,
            lvface_encoder=lvface_encoder,
        )
        query_coverage.to_csv(evaluation_dir / "query_coverage.csv", index=False)
        real_reference_baseline = _real_reference_baseline(
            root=root,
            evaluation_dir=evaluation_dir,
            dataset=dataset,
            generated=generated,
            identity_ids=identity_ids,
            face_encoder=face_encoder,
            lvface_encoder=lvface_encoder,
            galleries=galleries,
        )
        real_reference_baseline.to_csv(
            evaluation_dir / "real_reference_baseline.csv", index=False
        )
        logger.info("Detecting and encoding %d generated outputs", len(generated))
        detection, generated_embeddings, valid_indices = _generated_embeddings(
            run_dir=run_dir,
            evaluation_dir=evaluation_dir,
            generated=generated,
            face_encoder=face_encoder,
            lvface_encoder=lvface_encoder,
        )
        scored = _attach_scores(
            generated, detection, valid_indices, generated_embeddings, galleries
        )
        scored.to_csv(evaluation_dir / "generated_identity_metrics.csv", index=False)
        aggregate = _aggregate_metrics(scored)
        aggregate.to_csv(evaluation_dir / "metrics_by_condition_prompt.csv", index=False)
        bootstraps = _bootstrap_contrasts(
            scored, args.bootstrap_resamples, args.bootstrap_seed
        )
        write_json(evaluation_dir / "paired_bootstrap.json", bootstraps)
        geometry = _geometry_rows(
            scored, valid_indices, generated_embeddings, galleries
        )
        pd.DataFrame(geometry).to_csv(evaluation_dir / "cohort_geometry.csv", index=False)
        np.savez_compressed(
            evaluation_dir / "evaluation_embeddings.npz",
            **{
                **{f"generated__{key}": value for key, value in generated_embeddings.items()},
                **{f"query_templates__{key}": value[0] for key, value in galleries.items()},
            },
        )
        elapsed_values = [float(value) for value in generated["elapsed_seconds"]]
        performance = {
            "generated_count": len(elapsed_values),
            "summed_generation_seconds": float(sum(elapsed_values)),
            "mean_generation_seconds": float(statistics.fmean(elapsed_values)),
            "median_generation_seconds": float(statistics.median(elapsed_values)),
            "minimum_generation_seconds": float(min(elapsed_values)),
            "maximum_generation_seconds": float(max(elapsed_values)),
            "peak_cuda_allocated_bytes": int(generated["peak_cuda_bytes"].max()),
            "peak_cuda_reserved_bytes": int(
                generated["peak_cuda_reserved_bytes"].max()
            ),
        }
        write_json(run_dir / "performance.json", performance)
        summary = {
            "interpretation_scope": (
                "exploratory FEI internal pilot; automatic metrics are not human judgments"
            ),
            "generated_count": int(len(scored)),
            "detected_count": int(scored["face_detected"].fillna(False).sum()),
            "single_face_count": int(scored["end_to_end_valid"].fillna(False).sum()),
            "identity_count": len(identity_ids),
            "query_conditions": QUERY_CONDITIONS,
            "primary_evaluator": "lvface_independent",
            "secondary_evaluator": "insightface_conditioner_evaluator_reuse_diagnostic",
            "dataset_manifest": str(dataset_path.relative_to(root).as_posix()),
            "dataset_manifest_sha256": sha256_file(dataset_path),
            "files": {
                "per_image": "generated_identity_metrics.csv",
                "aggregate": "metrics_by_condition_prompt.csv",
                "bootstrap": "paired_bootstrap.json",
                "geometry": "cohort_geometry.csv",
                "query_coverage": "query_coverage.csv",
                "real_reference_baseline": "real_reference_baseline.csv",
            },
        }
        write_json(evaluation_dir / "summary.json", summary)
        write_json(evaluation_dir / "status.json", {"state": "complete"})
        successful_run_status = {
            key: value
            for key, value in run_status.items()
            if key not in {"error", "error_type"}
        }
        write_json(
            run_status_path,
            {
                **successful_run_status,
                "state": "automatic_evaluation_complete_human_pending",
                "automatic_evaluation": f"{args.evaluation_name}/summary.json",
            },
        )
        logger.info("Generation evaluation complete: %s", evaluation_dir)
    except Exception as error:
        write_json(
            evaluation_dir / "status.json",
            {"state": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        write_json(
            run_status_path,
            {
                **run_status,
                "state": "automatic_evaluation_failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
