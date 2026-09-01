from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


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

    split_counts = config["data"]["split_identity_counts"]
    if sum(int(value) for value in split_counts.values()) != int(
        config["data"]["expected_identities"]
    ):
        raise ConfigError("Identity split counts must sum to expected_identities")

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
    if calibration_split not in profiling_splits:
        raise ConfigError("threshold_calibration_split must be one of profiling_splits")
    if final_split not in known_splits or final_split in profiling_splits:
        raise ConfigError("final_evaluation_split must be known and excluded from profiling_splits")

    config["_meta"] = {
        "config_source": str(config_path),
        "repository_root": str(root),
    }
    return config


def load_asset_lock(config: dict[str, Any]) -> dict[str, Any]:
    return load_yaml(Path(config["assets"]["lock_file"]))
