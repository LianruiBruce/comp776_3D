from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


E2_SPLITS = {"train", "dev", "internal_eval"}
E2_SELECTION_METRIC = "crossfit_template_tar_at_far_0.01_mean"
E2_SELECTION_TIEBREAK_METRICS = [
    "template_d_prime_mean",
    "template_roc_auc_mean",
]
E2_REQUIRED_REPRESENTATIONS = {
    "probe_ln0_mean",
    "token_mean",
    "head_projected",
}
E2_FORMAL_SEEDS = [776, 1776, 2776, 3776, 4776]
E2_FORMAL_OPTIMIZER = {
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "weight_decay": 0.1,
    "polynomial_power": 1.0,
    "final_learning_rate": 0.0,
}
E2_FORMAL_LOSS = {"scale": 64.0, "margin": 0.4}


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be a finite number")
    return result


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _require_fields(mapping: dict[str, Any], fields: set[str], field: str) -> None:
    missing = sorted(fields - mapping.keys())
    if missing:
        raise ConfigError(f"Missing {field} fields: {missing}")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}")
    return value


def resolve_experiment_config(path: str | Path) -> dict[str, Any]:
    root = repository_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    config = deepcopy(load_yaml(config_path))

    required_sections = {"experiment", "assets", "data", "model", "evaluation"}
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    experiment_kind = config["experiment"].get("kind", "layer_sweep")
    supported_kinds = {"layer_sweep", "equal_capacity_probes"}
    if experiment_kind not in supported_kinds:
        raise ConfigError(
            f"Unsupported experiment.kind {experiment_kind!r}; expected one of "
            f"{sorted(supported_kinds)}"
        )
    config["experiment"]["kind"] = experiment_kind

    path_fields = [
        ("experiment", "output_root"),
        ("assets", "lock_file"),
        ("data", "archive_root"),
        ("data", "processed_root"),
        ("data", "manifest_path"),
        ("model", "third_party_path"),
        ("model", "weights_path"),
    ]
    for section, key in path_fields:
        value = Path(config[section][key])
        config[section][key] = str((root / value).resolve() if not value.is_absolute() else value)

    for key in ["source_root"]:
        if key in config["data"]:
            value = Path(config["data"][key])
            config["data"][key] = str(
                (root / value).resolve() if not value.is_absolute() else value
            )
    preprocessing = config["data"].get("preprocessing", {})
    for key in [
        "detector_cache_root",
        "detector_model_path",
        "insightface_checkout_path",
    ]:
        if key in preprocessing:
            value = Path(preprocessing[key])
            preprocessing[key] = str(
                (root / value).resolve() if not value.is_absolute() else value
            )

    split_counts = config["data"]["split_identity_counts"]
    if sum(int(value) for value in split_counts.values()) != int(
        config["data"]["expected_identities"]
    ):
        raise ConfigError("Identity split counts must sum to expected_identities")
    if experiment_kind == "equal_capacity_probes":
        if set(split_counts) != E2_SPLITS:
            raise ConfigError(
                "equal_capacity_probes requires exactly the train, dev, and "
                "internal_eval identity splits"
            )
        if any(int(split_counts[split]) <= 0 for split in E2_SPLITS):
            raise ConfigError("Every equal_capacity_probes identity split must be non-empty")

    evaluation = config["evaluation"]
    profiling_splits = list(evaluation["profiling_splits"])
    selection_split = evaluation["selection_split"]
    calibration_split = evaluation["threshold_calibration_split"]
    final_split = evaluation["final_evaluation_split"]
    known_splits = set(split_counts)
    if not set(profiling_splits).issubset(known_splits):
        raise ConfigError("profiling_splits contains an unknown data split")
    if selection_split not in profiling_splits:
        raise ConfigError("selection_split must be one of profiling_splits")
    if calibration_split not in known_splits:
        raise ConfigError("threshold_calibration_split must be a known data split")
    if final_split not in known_splits or final_split in profiling_splits:
        raise ConfigError("final_evaluation_split must be known and excluded from profiling_splits")
    if calibration_split == final_split:
        raise ConfigError(
            "threshold_calibration_split and final_evaluation_split must be different"
        )

    if experiment_kind == "equal_capacity_probes":
        probe = config.get("probe")
        if not isinstance(probe, dict):
            raise ConfigError("equal_capacity_probes requires a probe section")
        required_probe_fields = {
            "augmentation",
            "input_representation",
            "input_dim",
            "output_dim",
            "layer_norm_eps",
            "seeds",
            "epochs",
            "batch_size",
            "controls",
            "early_stopping",
            "optimizer",
            "loss",
            "name",
            "dev_folds",
            "dev_fold_seed",
            "precision",
            "projection_bias",
            "save_checkpoints",
        }
        missing_probe = sorted(required_probe_fields - probe.keys())
        if missing_probe:
            raise ConfigError(f"Missing probe fields: {missing_probe}")

        protocol_stage = config["experiment"].get("protocol_stage")
        if protocol_stage not in {"E2", "E2-smoke"}:
            raise ConfigError(
                "equal_capacity_probes experiment.protocol_stage must be E2 or E2-smoke"
            )
        if profiling_splits != ["dev"]:
            raise ConfigError("equal_capacity_probes requires profiling_splits=[dev]")
        if selection_split != "dev":
            raise ConfigError("equal_capacity_probes requires selection_split=dev")
        if calibration_split != "dev":
            raise ConfigError(
                "equal_capacity_probes requires threshold_calibration_split=dev"
            )
        if final_split != "internal_eval":
            raise ConfigError(
                "equal_capacity_probes requires final_evaluation_split=internal_eval"
            )
        if evaluation.get("condition_splits") != ["internal_eval"]:
            raise ConfigError("equal_capacity_probes requires condition_splits=[internal_eval]")
        if evaluation.get("selection_metric") != E2_SELECTION_METRIC:
            raise ConfigError(
                f"equal_capacity_probes requires selection_metric={E2_SELECTION_METRIC}"
            )
        if evaluation.get("selection_tiebreak_metrics") != E2_SELECTION_TIEBREAK_METRICS:
            raise ConfigError(
                "equal_capacity_probes requires selection_tiebreak_metrics="
                f"{E2_SELECTION_TIEBREAK_METRICS}"
            )
        far_targets = evaluation.get("far_targets")
        if not isinstance(far_targets, list) or len(far_targets) != 1:
            raise ConfigError("equal_capacity_probes supports only far_targets=[0.01]")
        try:
            normalized_far_targets = [float(target) for target in far_targets]
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "equal_capacity_probes supports only far_targets=[0.01]"
            ) from error
        if normalized_far_targets != [0.01]:
            raise ConfigError("equal_capacity_probes supports only far_targets=[0.01]")
        evaluation["far_targets"] = normalized_far_targets

        if probe["projection_bias"] is not False:
            raise ConfigError("equal_capacity_probes requires probe.projection_bias=false")
        if probe["precision"] != "float32":
            raise ConfigError("equal_capacity_probes requires probe.precision=float32")
        if probe["augmentation"] != "none":
            raise ConfigError("equal_capacity_probes requires probe.augmentation=none")
        if probe["early_stopping"] is not False:
            raise ConfigError("equal_capacity_probes requires probe.early_stopping=false")
        if probe["save_checkpoints"] is not True:
            raise ConfigError("equal_capacity_probes requires probe.save_checkpoints=true")

        if not isinstance(probe["name"], str) or not probe["name"].strip():
            raise ConfigError("probe.name must be a non-empty string")
        representations = config["model"].get("representations")
        if not isinstance(representations, list):
            raise ConfigError("model.representations must be a list")
        missing_representations = sorted(E2_REQUIRED_REPRESENTATIONS - set(representations))
        if missing_representations:
            raise ConfigError(
                "equal_capacity_probes model.representations is missing: "
                f"{missing_representations}"
            )
        if probe["input_representation"] != "probe_ln0_mean":
            raise ConfigError(
                "equal_capacity_probes requires probe.input_representation=probe_ln0_mean"
            )
        raw_seeds = probe["seeds"]
        if (
            not isinstance(raw_seeds, list)
            or not raw_seeds
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_seeds)
        ):
            raise ConfigError("probe.seeds must be a non-empty list of unique integers")
        seeds = list(raw_seeds)
        if len(seeds) != len(set(seeds)):
            raise ConfigError("probe.seeds must be a non-empty list of unique integers")
        probe["seeds"] = seeds
        if protocol_stage == "E2" and seeds != E2_FORMAL_SEEDS:
            raise ConfigError(
                f"Formal protocol_stage=E2 requires probe.seeds={E2_FORMAL_SEEDS}"
            )
        for key in ["input_dim", "output_dim", "epochs", "batch_size", "dev_folds"]:
            probe[key] = _positive_integer(probe[key], f"probe.{key}")
        if probe["dev_folds"] < 2:
            raise ConfigError("probe.dev_folds must be at least 2")
        if probe["dev_folds"] > int(split_counts["dev"]):
            raise ConfigError(
                "probe.dev_folds cannot exceed the configured dev identity count"
            )
        layer_norm_eps = _finite_float(probe["layer_norm_eps"], "probe.layer_norm_eps")
        if layer_norm_eps != 1.0e-5:
            raise ConfigError("equal_capacity_probes requires probe.layer_norm_eps=1e-5")
        probe["layer_norm_eps"] = layer_norm_eps

        optimizer = probe["optimizer"]
        if not isinstance(optimizer, dict):
            raise ConfigError("probe.optimizer must be a mapping")
        _require_fields(
            optimizer,
            {
                "name",
                "learning_rate",
                "betas",
                "weight_decay",
                "polynomial_power",
                "final_learning_rate",
            },
            "probe.optimizer",
        )
        if optimizer.get("name") != "adamw_polynomial":
            raise ConfigError("E2 currently supports optimizer.name=adamw_polynomial only")
        learning_rate = _finite_float(
            optimizer["learning_rate"], "probe.optimizer.learning_rate"
        )
        if learning_rate <= 0:
            raise ConfigError("probe.optimizer.learning_rate must be positive")
        betas = optimizer["betas"]
        if not isinstance(betas, list) or len(betas) != 2:
            raise ConfigError("probe.optimizer.betas must contain exactly two values")
        normalized_betas = [
            _finite_float(value, f"probe.optimizer.betas[{index}]")
            for index, value in enumerate(betas)
        ]
        if any(not 0 <= value < 1 for value in normalized_betas):
            raise ConfigError("probe.optimizer.betas values must be in [0, 1)")
        weight_decay = _finite_float(
            optimizer["weight_decay"], "probe.optimizer.weight_decay"
        )
        if weight_decay < 0:
            raise ConfigError("probe.optimizer.weight_decay must be non-negative")
        polynomial_power = _finite_float(
            optimizer["polynomial_power"], "probe.optimizer.polynomial_power"
        )
        if polynomial_power <= 0:
            raise ConfigError("probe.optimizer.polynomial_power must be positive")
        final_learning_rate = _finite_float(
            optimizer["final_learning_rate"], "probe.optimizer.final_learning_rate"
        )
        if final_learning_rate != 0.0:
            raise ConfigError(
                "equal_capacity_probes requires optimizer.final_learning_rate=0"
            )
        optimizer.update(
            {
                "learning_rate": learning_rate,
                "betas": normalized_betas,
                "weight_decay": weight_decay,
                "polynomial_power": polynomial_power,
                "final_learning_rate": final_learning_rate,
            }
        )

        loss = probe["loss"]
        if not isinstance(loss, dict):
            raise ConfigError("probe.loss must be a mapping")
        _require_fields(loss, {"name", "scale", "margin"}, "probe.loss")
        if loss.get("name") != "cosface":
            raise ConfigError("E2 currently supports loss.name=cosface only")
        scale = _finite_float(loss["scale"], "probe.loss.scale")
        margin = _finite_float(loss["margin"], "probe.loss.margin")
        if scale <= 0 or not 0 <= margin < 1:
            raise ConfigError("CosFace scale must be positive and margin must be in [0, 1)")
        loss.update({"scale": scale, "margin": margin})

        controls = probe["controls"]
        if not isinstance(controls, dict):
            raise ConfigError("probe.controls must be a mapping")
        shuffled = controls.get("shuffled_labels")
        if not isinstance(shuffled, dict):
            raise ConfigError("probe.controls.shuffled_labels must be a mapping")
        enabled = shuffled.get("enabled")
        shuffled_seeds = shuffled.get("seeds")
        if not isinstance(enabled, bool):
            raise ConfigError("shuffled_labels.enabled must be a boolean")
        if not isinstance(shuffled_seeds, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in shuffled_seeds
        ):
            raise ConfigError("shuffled_labels.seeds must be a list of unique integers")
        if len(shuffled_seeds) != len(set(shuffled_seeds)):
            raise ConfigError("shuffled_labels.seeds must be a list of unique integers")
        if enabled != bool(shuffled_seeds):
            raise ConfigError(
                "shuffled_labels.enabled must be true exactly when shuffled-label seeds exist"
            )
        unexpected_shuffled_seeds = sorted(set(shuffled_seeds) - set(seeds))
        if unexpected_shuffled_seeds:
            raise ConfigError(
                "shuffled_labels.seeds must be drawn from probe.seeds; unexpected seeds: "
                f"{unexpected_shuffled_seeds}"
            )

        if protocol_stage == "E2":
            if split_counts != {"train": 100, "dev": 50, "internal_eval": 50}:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires identity split counts 100/50/50"
                )
            if config["data"].get("identity_split_seed") != 20260902:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires data.identity_split_seed=20260902"
                )
            if config["experiment"].get("verify_full_extraction_repeat") is not True:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires "
                    "experiment.verify_full_extraction_repeat=true"
                )
            if probe.get("verify_repeat") != {
                "enabled": True,
                "layer": 1,
                "seed": 776,
            }:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires probe.verify_repeat at layer 1, seed 776"
                )
            if evaluation.get("bootstrap_resamples") != 1000:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires evaluation.bootstrap_resamples=1000"
                )
            if evaluation.get("bootstrap_seed") != 20260902:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires evaluation.bootstrap_seed=20260902"
                )

            formal_probe_values = {
                "input_dim": 256,
                "output_dim": 512,
                "epochs": 120,
                "batch_size": 128,
                "dev_folds": 5,
            }
            for field, expected in formal_probe_values.items():
                if probe[field] != expected:
                    raise ConfigError(
                        f"Formal protocol_stage=E2 requires probe.{field}={expected}"
                    )
            if set(optimizer) != {"name", *E2_FORMAL_OPTIMIZER}:
                raise ConfigError(
                    "Formal protocol_stage=E2 optimizer must contain only the frozen fields"
                )
            for field, expected in E2_FORMAL_OPTIMIZER.items():
                if optimizer[field] != expected:
                    raise ConfigError(
                        "Formal protocol_stage=E2 requires "
                        f"probe.optimizer.{field}={expected}"
                    )
            if set(loss) != {"name", *E2_FORMAL_LOSS}:
                raise ConfigError(
                    "Formal protocol_stage=E2 loss must contain only the frozen fields"
                )
            for field, expected in E2_FORMAL_LOSS.items():
                if loss[field] != expected:
                    raise ConfigError(
                        f"Formal protocol_stage=E2 requires probe.loss.{field}={expected}"
                    )

            expected_control_names = {
                "untrained_projection",
                "shuffled_labels",
                "gaussian_features",
            }
            if set(controls) != expected_control_names:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires exactly the untrained_projection, "
                    "shuffled_labels, and gaussian_features controls"
                )
            if controls["untrained_projection"] is not True:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires controls.untrained_projection=true"
                )
            if shuffled != {"enabled": True, "seeds": [776]}:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires shuffled_labels enabled with seed 776"
                )
            gaussian = controls["gaussian_features"]
            if gaussian != {"enabled": True, "seeds": [776]}:
                raise ConfigError(
                    "Formal protocol_stage=E2 requires gaussian_features enabled with seed 776"
                )

    config["_meta"] = {
        "config_source": str(config_path),
        "repository_root": str(root),
    }
    return config


def load_asset_lock(config: dict[str, Any]) -> dict[str, Any]:
    return load_yaml(Path(config["assets"]["lock_file"]))
