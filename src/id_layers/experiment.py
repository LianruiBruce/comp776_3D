from __future__ import annotations

import logging
import traceback
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
    bootstrap_selected_template_tar,
    evaluate_all_layers,
    evaluate_frozen_operating_points,
    evaluate_frozen_selected_layers_by_condition,
    evaluate_selected_layers,
    evaluate_selected_layers_by_condition,
    select_layers,
)
from .modeling import (
    check_determinism,
    compare_extraction_results,
    extract_layer_embeddings,
    load_lvface_model,
)
from .utils import (
    environment_metadata,
    python_package_lock,
    set_reproducibility,
    snapshot_research_sources,
    utc_run_id,
    write_json,
    write_yaml,
)

plt.switch_backend("Agg")


def _configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("id_layers")
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


def _plot_layer_curves(metrics, run_dir: Path) -> list[str]:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    separation_metric = (
        "template_d_prime" if "template_d_prime" in metrics else "d_prime"
    )
    retrieval_metric = "template_rank1" if "template_rank1" in metrics else "loo_top1"
    for split in metrics["split"].unique():
        subset = metrics[metrics["split"] == split]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for representation, group in subset.groupby("representation"):
            group = group.sort_values("layer")
            axes[0].plot(
                group["layer"], group[separation_metric], marker="o", label=representation
            )
            axes[1].plot(
                group["layer"], group[retrieval_metric], marker="o", label=representation
            )
        axes[0].set_title(f"{split}: identity separation")
        axes[0].set_xlabel("Transformer block")
        axes[0].set_ylabel(separation_metric.replace("_", " "))
        axes[1].set_title(f"{split}: leave-one-out retrieval")
        axes[1].set_xlabel("Transformer block")
        axes[1].set_ylabel(retrieval_metric.replace("_", " "))
        axes[1].set_ylim(0.0, 1.02)
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        path = plot_dir / f"layer_curves_{split}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path.relative_to(run_dir).as_posix())
    return paths


def _save_embeddings(
    run_dir: Path, embeddings: dict[tuple[int, str], np.ndarray]
) -> None:
    arrays = {
        f"layer_{layer:02d}__{representation}": value
        for (layer, representation), value in embeddings.items()
    }
    np.savez_compressed(run_dir / "embeddings.npz", **arrays)


def run_experiment(config: dict[str, Any], run_id: str | None = None) -> Path:
    root = repository_root()
    experiment_config = config["experiment"]
    output_root = Path(experiment_config["output_root"])
    run_id = run_id or utc_run_id(experiment_config["name"])
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = _configure_logging(run_dir)
    write_yaml(run_dir / "config.resolved.yaml", config)
    write_json(run_dir / "status.json", {"state": "running"})

    try:
        seed = int(experiment_config["seed"])
        set_reproducibility(seed, bool(experiment_config["deterministic"]))
        source_manifest = snapshot_research_sources(root, run_dir / "source_snapshot")
        write_json(run_dir / "source_snapshot_manifest.json", source_manifest)
        asset_lock = load_asset_lock(config)
        write_yaml(run_dir / "assets.lock.resolved.yaml", asset_lock)
        logger.info("Preparing dataset manifest and preprocessing coverage")
        full_manifest = prepare_dataset_manifest(config["data"], asset_lock, seed)
        private_columns = [
            column
            for column in ["absolute_path", "source_absolute_path"]
            if column in full_manifest
        ]
        full_manifest.drop(columns=private_columns).to_csv(
            run_dir / "dataset_manifest.csv", index=False
        )
        full_manifest.drop(columns=private_columns).to_csv(
            run_dir / "preprocess_coverage.csv", index=False
        )
        usable_mask = full_manifest["usable_for_model"].astype(bool).to_numpy()
        manifest = full_manifest.loc[usable_mask].reset_index(drop=True)
        if manifest.empty:
            raise RuntimeError("No samples passed detection/alignment preprocessing")

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

        logger.info("Loading pinned LVFace-T model")
        model = load_lvface_model(
            config["model"], asset_lock["models"][config["model"]["name"]]
        )
        first_sample = dataset[0]["image"].unsqueeze(0)
        determinism = check_determinism(model, first_sample, config["model"]["device"])
        logger.info("Extracting all Transformer block representations")
        extraction = extract_layer_embeddings(
            model=model,
            loader=loader,
            device_name=config["model"]["device"],
            representations=config["model"]["representations"],
            image_size=int(config["data"]["preprocessing"]["image_size"]),
        )
        extraction.metadata["determinism"] = determinism
        if bool(experiment_config.get("verify_full_extraction_repeat", False)):
            logger.info("Repeating the complete extraction for determinism verification")
            repeated = extract_layer_embeddings(
                model=model,
                loader=loader,
                device_name=config["model"]["device"],
                representations=config["model"]["representations"],
                image_size=int(config["data"]["preprocessing"]["image_size"]),
            )
            extraction.metadata["full_extraction_repeat"] = compare_extraction_results(
                extraction, repeated
            )
        write_json(run_dir / "extraction.json", extraction.metadata)

        logger.info("Computing identity verification and retrieval metrics")
        far_targets = [float(value) for value in config["evaluation"]["far_targets"]]
        profile_metrics = evaluate_all_layers(
            extraction.embeddings,
            manifest,
            list(config["evaluation"]["profiling_splits"]),
            far_targets,
        )
        selection_tiebreak_metrics = list(
            config["evaluation"].get("selection_tiebreak_metrics", [])
        )
        preliminary_selection = select_layers(
            profile_metrics,
            config["evaluation"]["selection_split"],
            config["evaluation"]["selection_metric"],
            selection_tiebreak_metrics,
        )
        final_metrics = evaluate_selected_layers(
            extraction.embeddings,
            manifest,
            preliminary_selection,
            config["evaluation"]["final_evaluation_split"],
            far_targets,
        )
        metrics = (
            pd.concat([profile_metrics, final_metrics], ignore_index=True)
            .sort_values(["representation", "split", "layer"])
            .reset_index(drop=True)
        )
        metrics.to_csv(run_dir / "metrics_by_layer.csv", index=False)
        selection = select_layers(
            metrics,
            config["evaluation"]["selection_split"],
            config["evaluation"]["selection_metric"],
            selection_tiebreak_metrics,
        )
        write_json(run_dir / "selection.json", selection)
        operating_points = evaluate_frozen_operating_points(
            extraction.embeddings,
            manifest,
            preliminary_selection,
            config["evaluation"]["threshold_calibration_split"],
            config["evaluation"]["final_evaluation_split"],
            far_targets,
        )
        write_json(run_dir / "frozen_operating_points.json", operating_points)
        condition_metrics_path = None
        bootstrap_path = None
        if "protocol_role" in manifest:
            condition_splits = list(
                config["evaluation"].get(
                    "condition_splits",
                    [
                        config["evaluation"]["selection_split"],
                        config["evaluation"]["final_evaluation_split"],
                    ],
                )
            )
            condition_metrics = evaluate_selected_layers_by_condition(
                extraction.embeddings,
                manifest,
                selection,
                condition_splits,
                far_targets,
            )
            coverage = (
                full_manifest[full_manifest["protocol_role"] == "query"]
                .groupby(["split", "condition"], as_index=False)
                .agg(
                    source_query_count=("sample_id", "size"),
                    usable_query_count=("usable_for_model", "sum"),
                )
                .rename(columns={"condition": "query_condition"})
            )
            coverage["query_coverage"] = (
                coverage["usable_query_count"] / coverage["source_query_count"]
            )
            condition_metrics = condition_metrics.merge(
                coverage, on=["split", "query_condition"], how="left", validate="many_to_one"
            )
            frozen_condition_metrics = evaluate_frozen_selected_layers_by_condition(
                extraction.embeddings,
                manifest,
                selection,
                config["evaluation"]["threshold_calibration_split"],
                config["evaluation"]["final_evaluation_split"],
                far_targets,
            )
            frozen_wide = frozen_condition_metrics.pivot(
                index=["split", "representation", "layer", "query_condition"],
                columns="target_far",
                values=["frozen_threshold", "frozen_observed_far", "frozen_tar"],
            )
            frozen_wide.columns = [
                f"{metric}_at_target_far_{target:g}" for metric, target in frozen_wide.columns
            ]
            frozen_wide = frozen_wide.reset_index()
            condition_metrics = condition_metrics.merge(
                frozen_wide,
                on=["split", "representation", "layer", "query_condition"],
                how="left",
                validate="one_to_one",
            )
            condition_metrics_path = "metrics_by_condition.csv"
            condition_metrics.to_csv(run_dir / condition_metrics_path, index=False)

            bootstrap = bootstrap_selected_template_tar(
                extraction.embeddings,
                manifest,
                selection,
                config["evaluation"]["threshold_calibration_split"],
                config["evaluation"]["final_evaluation_split"],
                far_targets,
                int(config["evaluation"].get("bootstrap_resamples", 1000)),
                int(config["evaluation"].get("bootstrap_seed", seed)),
            )
            bootstrap_path = "bootstrap_intervals.csv"
            bootstrap.to_csv(run_dir / bootstrap_path, index=False)
        plot_paths = _plot_layer_curves(metrics, run_dir)

        if bool(experiment_config["save_embeddings"]):
            _save_embeddings(run_dir, extraction.embeddings)

        environment = environment_metadata(root)
        write_json(run_dir / "environment.json", environment)
        (run_dir / "environment.lock.txt").write_text(
            python_package_lock(), encoding="utf-8"
        )
        summary = {
            "run_id": run_id,
            "dataset": {
                "name": config["data"]["name"],
                "source_samples": len(full_manifest),
                "usable_samples": len(manifest),
                "coverage": float(len(manifest) / len(full_manifest)),
                "source_identities": int(full_manifest["identity_id"].nunique()),
                "usable_identities": int(manifest["identity_id"].nunique()),
                "split_identity_counts": manifest.groupby("split")["identity_id"]
                .nunique()
                .to_dict(),
            },
            "model": {
                "name": config["model"]["name"],
                "parameters": extraction.metadata["parameter_count"],
                "layers": extraction.metadata["layer_count"],
            },
            "selection": selection,
            "frozen_operating_points": operating_points,
            "performance": extraction.metadata,
            "plots": plot_paths,
            "metrics_by_condition": condition_metrics_path,
            "bootstrap_intervals": bootstrap_path,
            "interpretation_scope": experiment_config.get(
                "interpretation_scope", "pipeline smoke; neutral-versus-smile only"
            ),
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "status.json", {"state": "complete", "run_id": run_id})
        logger.info("Run complete: %s", run_dir)
        return run_dir
    except Exception as error:
        logger.exception("Run failed")
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
