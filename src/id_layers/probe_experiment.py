from __future__ import annotations

import gc
import hashlib
import logging
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .config import load_asset_lock, repository_root
from .data import FaceManifestDataset, prepare_dataset_manifest
from .metrics import (
    frozen_template_threshold_metrics,
    template_query_metrics,
    template_query_score_frame,
)
from .modeling import (
    check_determinism,
    compare_extraction_results,
    extract_layer_embeddings,
    load_lvface_model,
)
from .probing import (
    EqualCapacityAngularProbe,
    ProbeTrainingConfig,
    aggregate_and_select_layer,
    assign_identity_folds,
    bootstrap_selected_vs_final_identity_tar,
    build_matched_probe,
    calibrate_far_threshold,
    cross_fitted_template_tar,
    train_equal_capacity_probe,
)
from .utils import (
    environment_metadata,
    python_package_lock,
    set_reproducibility,
    sha256_file,
    snapshot_research_sources,
    utc_run_id,
    write_json,
    write_yaml,
)

plt.switch_backend("Agg")


def _configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("id_layers.probes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _public_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    private = [
        column
        for column in ("absolute_path", "source_absolute_path")
        if column in frame
    ]
    return frame.drop(columns=private)


def _loader(
    manifest: pd.DataFrame, config: dict[str, Any], seed: int
) -> tuple[FaceManifestDataset, DataLoader]:
    dataset = FaceManifestDataset(manifest, config["data"]["preprocessing"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["model"]["num_workers"]),
        pin_memory=config["model"]["device"] == "cuda",
        generator=generator,
    )
    return dataset, loader


def _training_config(config: dict[str, Any]) -> ProbeTrainingConfig:
    probe = config["probe"]
    optimizer = probe["optimizer"]
    loss = probe["loss"]
    return ProbeTrainingConfig(
        output_dim=int(probe["output_dim"]),
        epochs=int(probe["epochs"]),
        batch_size=int(probe["batch_size"]),
        learning_rate=float(optimizer["learning_rate"]),
        beta1=float(optimizer["betas"][0]),
        beta2=float(optimizer["betas"][1]),
        weight_decay=float(optimizer["weight_decay"]),
        polynomial_power=float(optimizer["polynomial_power"]),
        scale=float(loss["scale"]),
        margin=float(loss["margin"]),
        device=str(config["model"]["device"]),
        deterministic=bool(config["experiment"]["deterministic"]),
    )


def _probe_embeddings(
    probe: EqualCapacityAngularProbe,
    features: np.ndarray,
    device_name: str,
    batch_size: int = 1024,
) -> np.ndarray:
    device = torch.device(device_name)
    probe = probe.to(device).eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.asarray(features[start : start + batch_size], dtype=np.float32)
            ).to(device)
            parts.append(probe(batch).cpu().numpy().astype(np.float32, copy=False))
    probe.cpu()
    return np.concatenate(parts, axis=0)


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _save_probe_checkpoint(
    path: Path,
    result: Any,
    *,
    layer: int | str,
    control: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe_state = {
        key: value.detach().cpu().contiguous()
        for key, value in result.probe.state_dict().items()
    }
    classifier_state = {
        key: value.detach().cpu().contiguous()
        for key, value in result.classifier.state_dict().items()
    }
    torch.save(
        {
            "probe_state": probe_state,
            "classifier_state": classifier_state,
            "class_labels": result.class_labels,
            "seed": result.seed,
            "layer": layer,
            "control": control,
            "training_config": result.config,
            "optimizer_steps": result.optimizer_steps,
        },
        path,
    )
    return {
        "path": path.as_posix(),
        "layer": layer,
        "seed": int(result.seed),
        "control": control,
        "probe_state_sha256": _state_sha256(probe_state),
        "classifier_state_sha256": _state_sha256(classifier_state),
        "file_sha256": sha256_file(path),
    }


def _load_probe(
    path: Path,
    input_dim: int,
    output_dim: int,
    *,
    expected_layer: int | str | None = None,
    expected_seed: int | None = None,
    expected_control: str | None = None,
    expected_file_sha256: str | None = None,
    expected_probe_state_sha256: str | None = None,
) -> EqualCapacityAngularProbe:
    if expected_file_sha256 is not None:
        observed_file_sha256 = sha256_file(path)
        if observed_file_sha256 != expected_file_sha256:
            raise RuntimeError(
                f"Probe checkpoint file checksum mismatch for {path}: "
                f"{observed_file_sha256} != {expected_file_sha256}"
            )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_metadata = {
        "layer": expected_layer,
        "seed": expected_seed,
        "control": expected_control,
    }
    for key, expected in expected_metadata.items():
        if expected is not None and payload.get(key) != expected:
            raise RuntimeError(
                f"Probe checkpoint {key} mismatch for {path}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    observed_state_sha256 = _state_sha256(payload["probe_state"])
    if (
        expected_probe_state_sha256 is not None
        and observed_state_sha256 != expected_probe_state_sha256
    ):
        raise RuntimeError(
            f"Probe state checksum mismatch for {path}: "
            f"{observed_state_sha256} != {expected_probe_state_sha256}"
        )
    probe = EqualCapacityAngularProbe(input_dim, output_dim)
    probe.load_state_dict(payload["probe_state"], strict=True)
    return probe.eval()


def _crossfit_measurement(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    folds: dict[str, int],
    target_far: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result = cross_fitted_template_tar(
        embeddings,
        manifest["identity_id"].astype(str).tolist(),
        manifest["protocol_role"].astype(str).tolist(),
        folds,
        target_far=target_far,
    )
    standard = template_query_metrics(embeddings, manifest, [target_far])
    row = {
        "cross_fitted_tar": result.tar,
        "cross_fitted_observed_far": result.observed_far,
        "cross_fitted_d_prime": result.d_prime,
        "cross_fitted_roc_auc": result.roc_auc,
        "cross_fitted_positive_pairs": result.positive_pairs,
        "cross_fitted_negative_pairs": result.negative_pairs,
        **standard,
    }
    fold_rows = [asdict(record) for record in result.folds]
    identity_rows = [
        {"identity_id": identity, "tar": tar}
        for identity, tar in result.identity_tar.items()
    ]
    return row, fold_rows, identity_rows


def _condition_wise_shuffled_labels(
    manifest: pd.DataFrame, seed: int
) -> np.ndarray:
    labels = manifest["identity_id"].astype(str).to_numpy(copy=True)
    shuffled = labels.copy()
    rng = np.random.default_rng(seed)
    for condition in sorted(manifest["condition"].astype(str).unique()):
        indices = np.flatnonzero(manifest["condition"].astype(str).to_numpy() == condition)
        shuffled[indices] = labels[indices][rng.permutation(len(indices))]
    if np.array_equal(labels, shuffled):
        raise RuntimeError("Shuffled-label control did not change any training label")
    return shuffled


def _frozen_evaluation(
    *,
    method: str,
    layer: int,
    seed: int | None,
    calibration_embeddings: np.ndarray,
    calibration_manifest: pd.DataFrame,
    evaluation_embeddings: np.ndarray,
    evaluation_manifest: pd.DataFrame,
    full_manifest: pd.DataFrame,
    evaluation_split: str,
    far_targets: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = frozen_template_threshold_metrics(
        calibration_embeddings,
        calibration_manifest,
        evaluation_embeddings,
        evaluation_manifest,
        far_targets,
    )
    standard = template_query_metrics(
        evaluation_embeddings, evaluation_manifest, far_targets
    )
    calibration_frame, calibration_negative, _ = template_query_score_frame(
        calibration_embeddings, calibration_manifest
    )
    del calibration_frame
    evaluation_frame, evaluation_negative, _ = template_query_score_frame(
        evaluation_embeddings, evaluation_manifest
    )

    metric_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    source_queries = full_manifest[
        (full_manifest["split"].astype(str) == evaluation_split)
        & (full_manifest["protocol_role"].astype(str) == "query")
    ]
    for target_far in far_targets:
        threshold = calibrate_far_threshold(calibration_negative, target_far)
        operating_point = frozen["operating_points"][f"far_{target_far:g}"]
        metric_rows.append(
            {
                "method": method,
                "layer": layer,
                "seed": seed,
                "target_far": target_far,
                "unprefixed_template_metrics_scope": (
                    "posthoc_internal_diagnostic_not_frozen"
                ),
                **standard,
                **{
                    f"frozen_{key}": value
                    for key, value in operating_point.items()
                },
            }
        )

        accepted = evaluation_frame["positive_score"].to_numpy(np.float64) >= threshold
        accepted_frame = evaluation_frame[["identity_id", "condition"]].copy()
        accepted_frame["identity_id"] = accepted_frame["identity_id"].astype(str)
        accepted_frame["accepted"] = accepted.astype(np.float64)
        for identity, source_group in source_queries.groupby("identity_id", sort=True):
            identity = str(identity)
            group = accepted_frame[accepted_frame["identity_id"] == identity]
            source_count = int(len(source_group))
            usable_count = int(source_group["usable_for_model"].astype(bool).sum())
            accepted_count = int(group["accepted"].sum())
            identity_rows.append(
                {
                    "method": method,
                    "layer": layer,
                    "seed": seed,
                    "target_far": target_far,
                    "identity_id": identity,
                    "source_query_count": source_count,
                    "usable_query_count": usable_count,
                    "scored_query_count": int(len(group)),
                    "accepted_query_count": accepted_count,
                    "aligned_only_tar": (
                        float(group["accepted"].mean()) if len(group) else 0.0
                    ),
                    "end_to_end_tar": accepted_count / source_count,
                    # The paired H2 bootstrap treats detection/alignment failures as rejects.
                    "tar": accepted_count / source_count,
                }
            )

        for condition in sorted(source_queries["condition"].astype(str).unique()):
            condition_mask = (
                (evaluation_manifest["protocol_role"].astype(str).to_numpy() == "reference")
                | (
                    (evaluation_manifest["protocol_role"].astype(str).to_numpy() == "query")
                    & (evaluation_manifest["condition"].astype(str).to_numpy() == condition)
                )
            )
            condition_manifest = evaluation_manifest.loc[condition_mask].reset_index(
                drop=True
            )
            condition_values = evaluation_embeddings[condition_mask]
            condition_query_count = int(
                (condition_manifest["protocol_role"].astype(str) == "query").sum()
            )
            condition_source = source_queries[
                source_queries["condition"].astype(str) == condition
            ]
            source_count = int(len(condition_source))
            usable_count = int(condition_source["usable_for_model"].astype(bool).sum())
            if condition_query_count == 0:
                condition_rows.append(
                    {
                        "method": method,
                        "layer": layer,
                        "seed": seed,
                        "split": evaluation_split,
                        "query_condition": condition,
                        "target_far": target_far,
                        "threshold": threshold,
                        "source_query_count": source_count,
                        "usable_query_count": usable_count,
                        "scored_query_count": 0,
                        "query_coverage": usable_count / source_count,
                        "scored_query_coverage": 0.0,
                        "accepted_query_count": 0,
                        "aligned_only_tar": None,
                        "end_to_end_tar": 0.0,
                        "aligned_observed_far": None,
                        "aligned_negative_pairs": 0,
                    }
                )
                continue
            condition_frame, condition_negative, _ = template_query_score_frame(
                condition_values, condition_manifest
            )
            condition_positive = condition_frame["positive_score"].to_numpy(np.float64)
            scored_count = int(len(condition_frame))
            accepted_count = int((condition_positive >= threshold).sum())
            condition_rows.append(
                {
                    "method": method,
                    "layer": layer,
                    "seed": seed,
                    "split": evaluation_split,
                    "query_condition": condition,
                    "target_far": target_far,
                    "threshold": threshold,
                    "source_query_count": source_count,
                    "usable_query_count": usable_count,
                    "scored_query_count": scored_count,
                    "query_coverage": usable_count / source_count,
                    "scored_query_coverage": scored_count / source_count,
                    "accepted_query_count": accepted_count,
                    "aligned_only_tar": float((condition_positive >= threshold).mean()),
                    "end_to_end_tar": accepted_count / source_count,
                    "aligned_observed_far": float(
                        (condition_negative.astype(np.float64) >= threshold).mean()
                    ),
                    "aligned_negative_pairs": int(condition_negative.size),
                }
            )

    if len(evaluation_negative) == 0:
        raise RuntimeError("Internal evaluation produced no negative pairs")
    return metric_rows, identity_rows, condition_rows


def _plot_probe_curves(summary: pd.DataFrame, run_dir: Path) -> str:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = summary["layer"].to_numpy()
    axes[0].errorbar(
        x,
        summary["cross_fitted_tar_mean"],
        yerr=summary["cross_fitted_tar_std"].fillna(0.0),
        marker="o",
        capsize=3,
    )
    axes[0].set_ylabel("Cross-fitted TAR @ FAR=0.01")
    axes[0].set_ylim(0.0, 1.02)
    axes[1].errorbar(
        x,
        summary["template_d_prime_mean"],
        yerr=summary["template_d_prime_std"].fillna(0.0),
        marker="o",
        capsize=3,
    )
    axes[1].set_ylabel("Template d-prime")
    for axis in axes:
        axis.set_xlabel("Transformer block")
        axis.grid(alpha=0.25)
        axis.set_xticks(x)
    figure.tight_layout()
    path = plot_dir / "equal_capacity_probe_layers.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.relative_to(run_dir).as_posix()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def run_probe_experiment(config: dict[str, Any], run_id: str | None = None) -> Path:
    root = repository_root()
    experiment = config["experiment"]
    probe_config = config["probe"]
    run_id = run_id or utc_run_id(experiment["name"])
    run_dir = Path(experiment["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = _configure_logging(run_dir)
    write_yaml(run_dir / "config.resolved.yaml", config)
    write_json(run_dir / "status.json", {"state": "running"})

    try:
        seed = int(experiment["seed"])
        set_reproducibility(seed, bool(experiment["deterministic"]))
        write_json(
            run_dir / "source_snapshot_manifest.json",
            snapshot_research_sources(root, run_dir / "source_snapshot"),
        )
        asset_lock = load_asset_lock(config)
        write_yaml(run_dir / "assets.lock.resolved.yaml", asset_lock)

        if probe_config.get("precision") != "float32":
            raise ValueError("E2 supports probe.precision=float32 only")
        if probe_config.get("augmentation") != "none":
            raise ValueError("Formal E2 requires augmentation=none")
        if bool(probe_config.get("early_stopping")):
            raise ValueError("Formal E2 forbids per-layer early stopping")
        if bool(probe_config.get("projection_bias")):
            raise ValueError("Formal E2 requires a bias-free projection")

        identity_split_seed = int(config["data"].get("identity_split_seed", seed))
        logger.info("Preparing and validating the immutable dataset manifest")
        full_manifest = prepare_dataset_manifest(
            config["data"], asset_lock, identity_split_seed
        )
        public_manifest = _public_manifest(full_manifest)
        public_manifest.to_csv(run_dir / "dataset_manifest.csv", index=False)
        public_manifest.to_csv(run_dir / "preprocess_coverage.csv", index=False)
        usable = full_manifest[full_manifest["usable_for_model"].astype(bool)].copy()
        usable = usable.reset_index(drop=True)
        if usable.empty:
            raise RuntimeError("No sample passed detection/alignment preprocessing")

        train_split = "train"
        dev_split = str(config["evaluation"]["selection_split"])
        internal_split = str(config["evaluation"]["final_evaluation_split"])
        development_manifest = usable[
            usable["split"].astype(str).isin([train_split, dev_split])
        ].reset_index(drop=True)
        if internal_split in set(development_manifest["split"].astype(str)):
            raise RuntimeError("Internal-evaluation samples entered layer development")
        _public_manifest(development_manifest).to_csv(
            run_dir / "development_manifest.csv", index=False
        )

        development_dataset, development_loader = _loader(
            development_manifest, config, seed
        )
        logger.info("Loading pinned frozen LVFace model")
        model = load_lvface_model(
            config["model"], asset_lock["models"][config["model"]["name"]]
        )
        determinism = check_determinism(
            model, development_dataset[0]["image"].unsqueeze(0), config["model"]["device"]
        )
        logger.info("Extracting every layer on train+dev identities only")
        development_extraction = extract_layer_embeddings(
            model,
            development_loader,
            config["model"]["device"],
            config["model"]["representations"],
            int(config["data"]["preprocessing"]["image_size"]),
        )
        development_extraction.metadata["determinism"] = determinism
        if bool(experiment.get("verify_full_extraction_repeat", False)):
            logger.info("Repeating train+dev extraction for exact reproducibility")
            repeated = extract_layer_embeddings(
                model,
                development_loader,
                config["model"]["device"],
                config["model"]["representations"],
                int(config["data"]["preprocessing"]["image_size"]),
            )
            development_extraction.metadata["full_extraction_repeat"] = (
                compare_extraction_results(development_extraction, repeated)
            )
            del repeated
        write_json(
            run_dir / "extraction_development.json", development_extraction.metadata
        )

        layer_count = int(development_extraction.metadata["layer_count"])
        input_representation = str(probe_config["input_representation"])
        input_dim = int(probe_config["input_dim"])
        for layer in range(1, layer_count + 1):
            observed_dim = development_extraction.embeddings[
                (layer, input_representation)
            ].shape[1]
            if observed_dim != input_dim:
                raise RuntimeError(
                    f"Layer {layer} probe input width {observed_dim} != {input_dim}"
                )
        if development_extraction.metadata["probe_layer_norm_eps"] != float(
            probe_config["layer_norm_eps"]
        ):
            raise RuntimeError("Cached probe LayerNorm epsilon differs from the config")

        train_mask = (
            development_manifest["split"].astype(str).to_numpy() == train_split
        )
        dev_mask = development_manifest["split"].astype(str).to_numpy() == dev_split
        train_manifest = development_manifest.loc[train_mask].reset_index(drop=True)
        dev_manifest = development_manifest.loc[dev_mask].reset_index(drop=True)
        train_labels = train_manifest["identity_id"].astype(str).to_numpy()
        dev_folds = assign_identity_folds(
            dev_manifest["identity_id"].astype(str).tolist(),
            n_splits=int(probe_config["dev_folds"]),
            seed=int(probe_config["dev_fold_seed"]),
        )
        write_json(run_dir / "dev_identity_folds.json", dev_folds)

        # The frozen backbone is moved off GPU while the small readouts are trained.
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        training = _training_config(config)
        seeds = [int(value) for value in probe_config["seeds"]]
        target_far = float(config["evaluation"]["far_targets"][0])
        metric_rows: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        dev_identity_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        history_rows: list[dict[str, Any]] = []
        checkpoint_records: list[dict[str, Any]] = []
        checkpoint_by_layer_seed: dict[tuple[int, int], Path] = {}
        checkpoint_record_by_layer_seed: dict[tuple[int, int], dict[str, Any]] = {}
        repeat_record: dict[str, Any] | None = None
        controls = probe_config.get("controls", {})
        shuffled_seeds = {
            int(value)
            for value in controls.get("shuffled_labels", {}).get("seeds", [])
        }
        training_started = time.perf_counter()

        for probe_seed in seeds:
            shuffled_labels = (
                _condition_wise_shuffled_labels(train_manifest, probe_seed)
                if probe_seed in shuffled_seeds
                else None
            )
            for layer in range(1, layer_count + 1):
                all_features = development_extraction.embeddings[
                    (layer, input_representation)
                ]
                train_features = all_features[train_mask]
                dev_features = all_features[dev_mask]
                logger.info(
                    "Training equal-capacity probe: layer=%02d seed=%d", layer, probe_seed
                )

                if bool(controls.get("untrained_projection", False)):
                    untrained, _ = build_matched_probe(
                        input_dim,
                        int(probe_config["output_dim"]),
                        int(train_manifest["identity_id"].nunique()),
                        seed=probe_seed,
                        scale=float(probe_config["loss"]["scale"]),
                        margin=float(probe_config["loss"]["margin"]),
                    )
                    untrained_embeddings = _probe_embeddings(
                        untrained, dev_features, config["model"]["device"]
                    )
                    measurement, _, _ = _crossfit_measurement(
                        untrained_embeddings, dev_manifest, dev_folds, target_far
                    )
                    control_rows.append(
                        {
                            "control": "untrained_projection",
                            "layer": layer,
                            "seed": probe_seed,
                            **measurement,
                        }
                    )

                result = train_equal_capacity_probe(
                    train_features,
                    train_labels,
                    seed=probe_seed,
                    config=training,
                )
                checkpoint_path = (
                    run_dir
                    / "checkpoints"
                    / f"layer_{layer:02d}__seed_{probe_seed}.pt"
                )
                checkpoint_by_layer_seed[(layer, probe_seed)] = checkpoint_path
                checkpoint_record = _save_probe_checkpoint(
                    checkpoint_path, result, layer=layer, control="trained"
                )
                checkpoint_records.append(checkpoint_record)
                checkpoint_record_by_layer_seed[(layer, probe_seed)] = checkpoint_record
                history_rows.extend(
                    {
                        "control": "trained",
                        "layer": layer,
                        "seed": probe_seed,
                        **asdict(record),
                    }
                    for record in result.history
                )
                dev_embeddings = _probe_embeddings(
                    result.probe, dev_features, config["model"]["device"]
                )
                measurement, folds, identity_measurements = _crossfit_measurement(
                    dev_embeddings, dev_manifest, dev_folds, target_far
                )
                metric_rows.append(
                    {"layer": layer, "seed": probe_seed, **measurement}
                )
                fold_rows.extend(
                    {"layer": layer, "seed": probe_seed, **record}
                    for record in folds
                )
                dev_identity_rows.extend(
                    {"layer": layer, "seed": probe_seed, **record}
                    for record in identity_measurements
                )

                repeat = probe_config.get("verify_repeat", {})
                if (
                    bool(repeat.get("enabled", False))
                    and int(repeat["layer"]) == layer
                    and int(repeat["seed"]) == probe_seed
                ):
                    repeated_result = train_equal_capacity_probe(
                        train_features,
                        train_labels,
                        seed=probe_seed,
                        config=training,
                    )
                    repeated_embeddings = _probe_embeddings(
                        repeated_result.probe, dev_features, config["model"]["device"]
                    )
                    first_hash = _state_sha256(result.probe.state_dict())
                    second_hash = _state_sha256(repeated_result.probe.state_dict())
                    repeat_record = {
                        "layer": layer,
                        "seed": probe_seed,
                        "probe_state_exact": first_hash == second_hash,
                        "first_probe_state_sha256": first_hash,
                        "second_probe_state_sha256": second_hash,
                        "max_abs_embedding_difference": float(
                            np.abs(dev_embeddings - repeated_embeddings).max()
                        ),
                        "embeddings_exact": bool(
                            np.array_equal(dev_embeddings, repeated_embeddings)
                        ),
                    }
                    if not (
                        repeat_record["probe_state_exact"]
                        and repeat_record["embeddings_exact"]
                    ):
                        raise RuntimeError("Repeated probe training was not exact")

                if shuffled_labels is not None:
                    shuffled_result = train_equal_capacity_probe(
                        train_features,
                        shuffled_labels,
                        seed=probe_seed,
                        config=training,
                    )
                    shuffled_path = (
                        run_dir
                        / "checkpoints"
                        / "controls"
                        / f"shuffled_layer_{layer:02d}__seed_{probe_seed}.pt"
                    )
                    checkpoint_records.append(
                        _save_probe_checkpoint(
                            shuffled_path,
                            shuffled_result,
                            layer=layer,
                            control="condition_wise_sample_label_shuffle",
                        )
                    )
                    history_rows.extend(
                        {
                            "control": "condition_wise_sample_label_shuffle",
                            "layer": layer,
                            "seed": probe_seed,
                            **asdict(record),
                        }
                        for record in shuffled_result.history
                    )
                    shuffled_embeddings = _probe_embeddings(
                        shuffled_result.probe,
                        dev_features,
                        config["model"]["device"],
                    )
                    shuffled_measurement, _, _ = _crossfit_measurement(
                        shuffled_embeddings, dev_manifest, dev_folds, target_far
                    )
                    control_rows.append(
                        {
                            "control": "condition_wise_sample_label_shuffle",
                            "layer": layer,
                            "seed": probe_seed,
                            **shuffled_measurement,
                        }
                    )

                del result, dev_embeddings
                gc.collect()

        gaussian_config = controls.get("gaussian_features", {})
        if bool(gaussian_config.get("enabled", False)):
            for gaussian_seed in gaussian_config.get("seeds", []):
                gaussian_seed = int(gaussian_seed)
                rng = np.random.default_rng(gaussian_seed)
                gaussian = rng.standard_normal(
                    (len(development_manifest), input_dim), dtype=np.float32
                )
                gaussian_result = train_equal_capacity_probe(
                    gaussian[train_mask],
                    train_labels,
                    seed=gaussian_seed,
                    config=training,
                )
                gaussian_path = (
                    run_dir
                    / "checkpoints"
                    / "controls"
                    / f"gaussian__seed_{gaussian_seed}.pt"
                )
                checkpoint_records.append(
                    _save_probe_checkpoint(
                        gaussian_path,
                        gaussian_result,
                        layer="synthetic",
                        control="sample_independent_gaussian_features",
                    )
                )
                history_rows.extend(
                    {
                        "control": "sample_independent_gaussian_features",
                        "layer": 0,
                        "seed": gaussian_seed,
                        **asdict(record),
                    }
                    for record in gaussian_result.history
                )
                gaussian_embeddings = _probe_embeddings(
                    gaussian_result.probe,
                    gaussian[dev_mask],
                    config["model"]["device"],
                )
                gaussian_measurement, _, _ = _crossfit_measurement(
                    gaussian_embeddings, dev_manifest, dev_folds, target_far
                )
                control_rows.append(
                    {
                        "control": "sample_independent_gaussian_features",
                        "layer": 0,
                        "seed": gaussian_seed,
                        **gaussian_measurement,
                    }
                )

        training_elapsed = time.perf_counter() - training_started
        training_peak = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        )
        metrics_by_seed = pd.DataFrame(metric_rows).sort_values(["layer", "seed"])
        metrics_by_seed.to_csv(run_dir / "metrics_by_layer_seed.csv", index=False)
        pd.DataFrame(fold_rows).sort_values(["layer", "seed", "fold"]).to_csv(
            run_dir / "crossfit_operating_points.csv", index=False
        )
        pd.DataFrame(dev_identity_rows).sort_values(
            ["layer", "seed", "identity_id"]
        ).to_csv(run_dir / "dev_identity_tar.csv", index=False)
        pd.DataFrame(control_rows).sort_values(["control", "layer", "seed"]).to_csv(
            run_dir / "control_metrics.csv", index=False
        )
        pd.DataFrame(history_rows).sort_values(
            ["control", "layer", "seed", "epoch"]
        ).to_csv(run_dir / "training_history.csv", index=False)
        write_json(run_dir / "probe_checkpoints.json", checkpoint_records)
        write_json(run_dir / "probe_repeat.json", repeat_record)

        selection_result = aggregate_and_select_layer(
            metrics_by_seed,
            primary_metric="cross_fitted_tar",
            tiebreak_metrics=("template_d_prime", "template_roc_auc"),
            required_seed_count=len(seeds),
        )
        layer_summary = selection_result.summary.copy()
        for column in layer_summary.columns:
            if column.endswith("_std"):
                layer_summary[column] = layer_summary[column].fillna(0.0)
        layer_summary.to_csv(run_dir / "metrics_by_layer.csv", index=False)
        selected_layer = int(selection_result.selected_layer)
        selection = {
            "selected_layer": selected_layer,
            "final_layer": layer_count,
            "seed_winners": selection_result.seed_winners,
            "primary_metric": "mean cross-fitted template TAR@FAR=0.01",
            "tiebreak_metrics": ["mean template d-prime", "mean template ROC-AUC"],
            "last_tiebreak": "shallower layer",
            "selection_split": dev_split,
            "probe_seeds": seeds,
            "internal_features_accessed_before_selection": False,
        }
        write_json(run_dir / "selection.json", selection)
        plot_path = _plot_probe_curves(layer_summary, run_dir)
        logger.info("Dev-frozen selected layer: %d", selected_layer)

        # Only now may internal-evaluation images enter the frozen backbone path.
        internal_access_started = time.time()
        internal_manifest = usable[
            usable["split"].astype(str) == internal_split
        ].reset_index(drop=True)
        _public_manifest(internal_manifest).to_csv(
            run_dir / "internal_evaluation_manifest.csv", index=False
        )
        _, internal_loader = _loader(internal_manifest, config, seed)
        frozen_layers = sorted({selected_layer, layer_count})
        logger.info(
            "Extracting only frozen layers %s on internal-evaluation identities",
            frozen_layers,
        )
        internal_extraction = extract_layer_embeddings(
            model,
            internal_loader,
            config["model"]["device"],
            config["model"]["representations"],
            int(config["data"]["preprocessing"]["image_size"]),
            layers=frozen_layers,
        )
        write_json(run_dir / "extraction_internal_frozen.json", internal_extraction.metadata)
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        internal_metric_rows: list[dict[str, Any]] = []
        internal_identity_rows: list[dict[str, Any]] = []
        condition_rows: list[dict[str, Any]] = []
        frozen_embedding_arrays: dict[str, np.ndarray] = {}
        bootstrap_selected_rows: list[dict[str, Any]] = []
        bootstrap_final_rows: list[dict[str, Any]] = []
        for probe_seed in seeds:
            for layer, method in (
                (selected_layer, "equal_capacity_selected"),
                (layer_count, "equal_capacity_final"),
            ):
                probe = _load_probe(
                    checkpoint_by_layer_seed[(layer, probe_seed)],
                    input_dim,
                    int(probe_config["output_dim"]),
                    expected_layer=layer,
                    expected_seed=probe_seed,
                    expected_control="trained",
                    expected_file_sha256=checkpoint_record_by_layer_seed[
                        (layer, probe_seed)
                    ]["file_sha256"],
                    expected_probe_state_sha256=checkpoint_record_by_layer_seed[
                        (layer, probe_seed)
                    ]["probe_state_sha256"],
                )
                dev_features = development_extraction.embeddings[
                    (layer, input_representation)
                ][dev_mask]
                internal_features = internal_extraction.embeddings[
                    (layer, input_representation)
                ]
                dev_embeddings = _probe_embeddings(
                    probe, dev_features, config["model"]["device"]
                )
                internal_embeddings = _probe_embeddings(
                    probe, internal_features, config["model"]["device"]
                )
                metrics, identities, conditions = _frozen_evaluation(
                    method=method,
                    layer=layer,
                    seed=probe_seed,
                    calibration_embeddings=dev_embeddings,
                    calibration_manifest=dev_manifest,
                    evaluation_embeddings=internal_embeddings,
                    evaluation_manifest=internal_manifest,
                    full_manifest=full_manifest,
                    evaluation_split=internal_split,
                    far_targets=[float(value) for value in config["evaluation"]["far_targets"]],
                )
                internal_metric_rows.extend(metrics)
                internal_identity_rows.extend(identities)
                condition_rows.extend(conditions)
                frozen_embedding_arrays[
                    f"{method}__seed_{probe_seed}__dev"
                ] = dev_embeddings
                frozen_embedding_arrays[
                    f"{method}__seed_{probe_seed}__internal"
                ] = internal_embeddings
                primary_identities = [
                    record
                    for record in identities
                    if float(record["target_far"]) == target_far
                ]
                destination = (
                    bootstrap_selected_rows
                    if method == "equal_capacity_selected"
                    else bootstrap_final_rows
                )
                destination.extend(
                    {
                        "identity_id": record["identity_id"],
                        "seed": probe_seed,
                        "tar": record["tar"],
                    }
                    for record in primary_identities
                )

        for representation, method in (
            ("head_projected", "official_final_head"),
            ("token_mean", "pretrained_norm_token_mean_final"),
        ):
            dev_embeddings = development_extraction.embeddings[
                (layer_count, representation)
            ][dev_mask]
            internal_embeddings = internal_extraction.embeddings[
                (layer_count, representation)
            ]
            metrics, identities, conditions = _frozen_evaluation(
                method=method,
                layer=layer_count,
                seed=None,
                calibration_embeddings=dev_embeddings,
                calibration_manifest=dev_manifest,
                evaluation_embeddings=internal_embeddings,
                evaluation_manifest=internal_manifest,
                full_manifest=full_manifest,
                evaluation_split=internal_split,
                far_targets=[float(value) for value in config["evaluation"]["far_targets"]],
            )
            internal_metric_rows.extend(metrics)
            internal_identity_rows.extend(identities)
            condition_rows.extend(conditions)
            frozen_embedding_arrays[f"{method}__dev"] = dev_embeddings
            frozen_embedding_arrays[f"{method}__internal"] = internal_embeddings

        internal_metrics = pd.DataFrame(internal_metric_rows).sort_values(
            ["method", "seed"], na_position="last"
        )
        internal_metrics.to_csv(run_dir / "internal_frozen_metrics.csv", index=False)
        pd.DataFrame(internal_identity_rows).sort_values(
            ["method", "seed", "identity_id"], na_position="last"
        ).to_csv(run_dir / "internal_identity_tar.csv", index=False)
        pd.DataFrame(condition_rows).sort_values(
            ["method", "seed", "query_condition"], na_position="last"
        ).to_csv(run_dir / "metrics_by_condition.csv", index=False)

        bootstrap = bootstrap_selected_vs_final_identity_tar(
            pd.DataFrame(bootstrap_selected_rows),
            pd.DataFrame(bootstrap_final_rows),
            resamples=int(config["evaluation"].get("bootstrap_resamples", 1000)),
            bootstrap_seed=int(config["evaluation"].get("bootstrap_seed", seed)),
            required_seed_count=len(seeds),
        )
        bootstrap_record = asdict(bootstrap)
        write_json(run_dir / "selected_vs_final_bootstrap.json", bootstrap_record)

        seed_deltas = (
            pd.DataFrame(bootstrap_selected_rows)
            .groupby("seed")["tar"]
            .mean()
            .subtract(
                pd.DataFrame(bootstrap_final_rows).groupby("seed")["tar"].mean()
            )
        )
        positive_seed_count = int((seed_deltas > 0).sum())
        internal_support = bool(
            len(seeds) >= 5
            and experiment.get("protocol_stage") == "E2"
            and selected_layer < layer_count
            and bootstrap.ci_lower_95 > 0.0
            and positive_seed_count >= min(4, len(seeds))
        )

        diagnostics: list[dict[str, Any]] = []
        for representation in ("token_mean", "head_projected"):
            for layer in range(1, layer_count + 1):
                values = development_extraction.embeddings[(layer, representation)][dev_mask]
                diagnostics.append(
                    {
                        "representation": representation,
                        "layer": layer,
                        **template_query_metrics(values, dev_manifest, [target_far]),
                    }
                )
        pd.DataFrame(diagnostics).sort_values(["representation", "layer"]).to_csv(
            run_dir / "frozen_backbone_diagnostic_metrics.csv", index=False
        )

        if bool(experiment.get("save_embeddings", False)):
            _save_npz(
                run_dir / "probe_inputs_development.npz",
                {
                    f"layer_{layer:02d}": development_extraction.embeddings[
                        (layer, input_representation)
                    ]
                    for layer in range(1, layer_count + 1)
                },
            )
            _save_npz(
                run_dir / "probe_inputs_internal_frozen.npz",
                {
                    f"layer_{layer:02d}": internal_extraction.embeddings[
                        (layer, input_representation)
                    ]
                    for layer in frozen_layers
                },
            )
            _save_npz(run_dir / "frozen_embeddings.npz", frozen_embedding_arrays)

        environment = environment_metadata(root)
        write_json(run_dir / "environment.json", environment)
        (run_dir / "environment.lock.txt").write_text(
            python_package_lock(), encoding="utf-8"
        )
        performance = {
            "development_extraction": development_extraction.metadata,
            "probe_training_elapsed_seconds": training_elapsed,
            "probe_training_peak_cuda_bytes": training_peak,
            "internal_frozen_extraction": internal_extraction.metadata,
        }
        write_json(run_dir / "performance.json", performance)
        access_audit = {
            "selection_written_before_internal_feature_extraction": True,
            "selected_layer": selected_layer,
            "internal_extracted_layers": frozen_layers,
            "internal_feature_access_started_unix": internal_access_started,
        }
        write_json(run_dir / "access_audit.json", access_audit)
        summary = {
            "run_id": run_id,
            "protocol_stage": experiment.get("protocol_stage", "E2"),
            "dataset": {
                "name": config["data"]["name"],
                "source_samples": int(len(full_manifest)),
                "usable_samples": int(len(usable)),
                "coverage": float(len(usable) / len(full_manifest)),
                "source_identities": int(full_manifest["identity_id"].nunique()),
                "split_identity_counts": {
                    str(key): int(value)
                    for key, value in usable.groupby("split")["identity_id"]
                    .nunique()
                    .items()
                },
            },
            "model": {
                "name": config["model"]["name"],
                "parameters": development_extraction.metadata["parameter_count"],
                "layers": layer_count,
            },
            "probe": {
                "family": probe_config["name"],
                "seeds": seeds,
                "epochs": training.epochs,
                "selected_layer": selected_layer,
                "final_layer": layer_count,
                "repeat": repeat_record,
            },
            "selected_vs_final": bootstrap_record,
            "selected_vs_final_bootstrap_metric": "end_to_end identity TAR",
            "positive_seed_directions": positive_seed_count,
            "supports_internal_H2_gate": internal_support,
            "claim_scope": (
                "transferable accessibility under one supervised angular readout family "
                "on FEI internal development; not causal storage and not external "
                "confirmation"
            ),
            "selection": selection,
            "performance": performance,
            "plot": plot_path,
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "status.json", {"state": "complete", "run_id": run_id})
        logger.info("Equal-capacity probe run complete: %s", run_dir)
        return run_dir
    except Exception as error:
        logger.exception("Equal-capacity probe run failed")
        write_json(
            run_dir / "status.json",
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
