from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from id_layers.config import ConfigError, resolve_experiment_config


def _probe_config(protocol_stage: str = "E2-smoke") -> dict[str, Any]:
    return {
        "experiment": {
            "name": "config_contract_test",
            "kind": "equal_capacity_probes",
            "protocol_stage": protocol_stage,
            "seed": 7,
            "deterministic": True,
            "output_root": "artifacts/runs",
        },
        "assets": {"lock_file": "configs/assets.lock.yaml"},
        "data": {
            "name": "synthetic",
            "archive_root": "data/raw/synthetic",
            "processed_root": "data/processed/synthetic",
            "manifest_path": "data/manifests/synthetic.csv",
            "expected_identities": 6,
            "split_identity_counts": {
                "train": 2,
                "dev": 2,
                "internal_eval": 2,
            },
        },
        "model": {
            "third_party_path": "third_party/LVFace",
            "weights_path": "artifacts/cache/model.pt",
            "representations": [
                "probe_ln0_mean",
                "token_mean",
                "head_projected",
            ],
        },
        "probe": {
            "name": "equal_capacity_supervised_metric_readout",
            "input_representation": "probe_ln0_mean",
            "input_dim": 256,
            "output_dim": 512,
            "layer_norm_eps": 1.0e-5,
            "projection_bias": False,
            "seeds": [776],
            "epochs": 2,
            "batch_size": 8,
            "precision": "float32",
            "augmentation": "none",
            "early_stopping": False,
            "optimizer": {
                "name": "adamw_polynomial",
                "learning_rate": 0.001,
                "betas": [0.9, 0.999],
                "weight_decay": 0.1,
                "polynomial_power": 1.0,
                "final_learning_rate": 0.0,
            },
            "loss": {"name": "cosface", "scale": 64.0, "margin": 0.4},
            "dev_folds": 2,
            "dev_fold_seed": 7,
            "controls": {
                "shuffled_labels": {"enabled": False, "seeds": []},
            },
            "save_checkpoints": True,
        },
        "evaluation": {
            "profiling_splits": ["dev"],
            "selection_split": "dev",
            "selection_metric": "crossfit_template_tar_at_far_0.01_mean",
            "selection_tiebreak_metrics": [
                "template_d_prime_mean",
                "template_roc_auc_mean",
            ],
            "threshold_calibration_split": "dev",
            "final_evaluation_split": "internal_eval",
            "far_targets": [0.01],
            "condition_splits": ["internal_eval"],
        },
    }


def _formal_probe_config() -> dict[str, Any]:
    config = _probe_config(protocol_stage="E2")
    config["experiment"]["verify_full_extraction_repeat"] = True
    config["data"]["identity_split_seed"] = 20260902
    config["data"]["expected_identities"] = 200
    config["data"]["split_identity_counts"] = {
        "train": 100,
        "dev": 50,
        "internal_eval": 50,
    }
    config["probe"].update(
        {
            "seeds": [776, 1776, 2776, 3776, 4776],
            "epochs": 120,
            "batch_size": 128,
            "dev_folds": 5,
            "verify_repeat": {"enabled": True, "layer": 1, "seed": 776},
            "controls": {
                "untrained_projection": True,
                "shuffled_labels": {"enabled": True, "seeds": [776]},
                "gaussian_features": {"enabled": True, "seeds": [776]},
            },
        }
    )
    config["evaluation"]["bootstrap_resamples"] = 1000
    config["evaluation"]["bootstrap_seed"] = 20260902
    return config


def _layer_sweep_config() -> dict[str, Any]:
    config = _probe_config()
    config["experiment"].pop("protocol_stage")
    config["experiment"]["kind"] = "layer_sweep"
    config.pop("probe")
    config["data"]["split_identity_counts"] = {"dev": 2, "val": 2, "test": 2}
    config["evaluation"].update(
        {
            "profiling_splits": ["dev"],
            "selection_split": "dev",
            "threshold_calibration_split": "val",
            "final_evaluation_split": "test",
        }
    )
    return config


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _resolve(tmp_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return resolve_experiment_config(path)


def test_equal_capacity_contract_accepts_smoke_and_five_seed_formal_config(
    tmp_path: Path,
) -> None:
    smoke = _resolve(tmp_path, _probe_config())
    assert smoke["probe"]["seeds"] == [776]

    resolved = _resolve(tmp_path, _formal_probe_config())
    assert len(resolved["probe"]["seeds"]) == 5
    assert resolved["evaluation"]["far_targets"] == [0.01]


def test_calibration_and_final_splits_must_differ_for_layer_sweeps(
    tmp_path: Path,
) -> None:
    config = _layer_sweep_config()
    config["evaluation"]["threshold_calibration_split"] = "test"
    with pytest.raises(ConfigError, match="must be different"):
        _resolve(tmp_path, config)


@pytest.mark.parametrize("missing", sorted(["probe_ln0_mean", "token_mean", "head_projected"]))
def test_equal_capacity_contract_requires_every_runner_representation(
    tmp_path: Path, missing: str
) -> None:
    config = _probe_config()
    config["model"]["representations"].remove(missing)
    with pytest.raises(ConfigError, match="model.representations is missing"):
        _resolve(tmp_path, config)


def test_equal_capacity_contract_requires_probe_name_input_and_epsilon(
    tmp_path: Path,
) -> None:
    missing_name = _probe_config()
    del missing_name["probe"]["name"]
    with pytest.raises(ConfigError, match="Missing probe fields.*name"):
        _resolve(tmp_path, missing_name)

    wrong_input = _probe_config()
    wrong_input["probe"]["input_representation"] = "token_mean"
    with pytest.raises(ConfigError, match="input_representation=probe_ln0_mean"):
        _resolve(tmp_path, wrong_input)

    wrong_epsilon = _probe_config()
    wrong_epsilon["probe"]["layer_norm_eps"] = 1.0e-6
    with pytest.raises(ConfigError, match="layer_norm_eps=1e-5"):
        _resolve(tmp_path, wrong_epsilon)


@pytest.mark.parametrize(
    ("mapping_name", "missing_field"),
    [
        ("optimizer", "name"),
        ("optimizer", "learning_rate"),
        ("optimizer", "betas"),
        ("optimizer", "weight_decay"),
        ("optimizer", "polynomial_power"),
        ("optimizer", "final_learning_rate"),
        ("loss", "name"),
        ("loss", "scale"),
        ("loss", "margin"),
    ],
)
def test_nested_training_fields_fail_as_config_errors(
    tmp_path: Path, mapping_name: str, missing_field: str
) -> None:
    config = _probe_config()
    del config["probe"][mapping_name][missing_field]
    with pytest.raises(ConfigError, match=f"Missing probe.{mapping_name} fields"):
        _resolve(tmp_path, config)


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        (1, "at least 2"),
        (3, "cannot exceed"),
    ],
)
def test_dev_fold_count_must_fit_configured_dev_identities(
    tmp_path: Path, folds: int, message: str
) -> None:
    config = _probe_config()
    config["probe"]["dev_folds"] = folds
    with pytest.raises(ConfigError, match=message):
        _resolve(tmp_path, config)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            [("data", "split_identity_counts")],
            "identity splits",
        ),
        (
            [
                ("evaluation", "profiling_splits"),
                ("evaluation", "selection_split"),
            ],
            "profiling_splits",
        ),
        (
            [("evaluation", "threshold_calibration_split")],
            "threshold_calibration_split",
        ),
        (
            [("evaluation", "final_evaluation_split")],
            "final_evaluation_split",
        ),
        (
            [("evaluation", "condition_splits")],
            "condition_splits",
        ),
        (
            [("evaluation", "selection_metric")],
            "selection_metric",
        ),
        (
            [("evaluation", "selection_tiebreak_metrics")],
            "selection_tiebreak_metrics",
        ),
        (
            [("evaluation", "far_targets")],
            "far_targets",
        ),
    ],
)
def test_equal_capacity_contract_rejects_unsupported_evaluation_fields(
    tmp_path: Path,
    changes: list[tuple[str, ...]],
    message: str,
) -> None:
    config = _probe_config()
    replacements: dict[tuple[str, ...], Any] = {
        ("data", "split_identity_counts"): {"train": 2, "dev": 2, "val": 2},
        ("evaluation", "profiling_splits"): ["train"],
        ("evaluation", "selection_split"): "train",
        ("evaluation", "threshold_calibration_split"): "train",
        ("evaluation", "final_evaluation_split"): "train",
        ("evaluation", "condition_splits"): ["dev", "internal_eval"],
        ("evaluation", "selection_metric"): "template_d_prime_mean",
        ("evaluation", "selection_tiebreak_metrics"): [
            "template_roc_auc_mean",
            "template_d_prime_mean",
        ],
        ("evaluation", "far_targets"): [0.001],
    }
    for path in changes:
        _set_nested(config, path, replacements[path])
    with pytest.raises(ConfigError, match=message):
        _resolve(tmp_path, config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("probe", "projection_bias"), True, "projection_bias"),
        (("probe", "precision"), "float16", "precision"),
        (("probe", "augmentation"), "horizontal_flip", "augmentation"),
        (("probe", "early_stopping"), True, "early_stopping"),
        (("probe", "save_checkpoints"), False, "save_checkpoints"),
        (
            ("probe", "optimizer", "final_learning_rate"),
            1.0e-5,
            "final_learning_rate",
        ),
    ],
)
def test_equal_capacity_contract_rejects_unsupported_probe_options(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    config = _probe_config()
    _set_nested(config, path, value)
    with pytest.raises(ConfigError, match=message):
        _resolve(tmp_path, config)


@pytest.mark.parametrize(
    ("enabled", "seeds", "message"),
    [
        (False, [776], "enabled"),
        (True, [], "enabled"),
        (True, [1776], "drawn from probe.seeds"),
        (True, [776, 776], "unique integers"),
        ("true", [776], "boolean"),
    ],
)
def test_shuffled_label_control_flag_and_seed_list_must_agree(
    tmp_path: Path,
    enabled: Any,
    seeds: list[int],
    message: str,
) -> None:
    config = _probe_config()
    config["probe"]["controls"]["shuffled_labels"] = {
        "enabled": enabled,
        "seeds": seeds,
    }
    with pytest.raises(ConfigError, match=message):
        _resolve(tmp_path, config)


def test_formal_e2_requires_exactly_five_unique_integer_seeds(tmp_path: Path) -> None:
    too_few = _formal_probe_config()
    too_few["probe"]["seeds"] = [776]
    with pytest.raises(ConfigError, match="requires probe.seeds"):
        _resolve(tmp_path, too_few)

    duplicates = _formal_probe_config()
    duplicates["probe"]["seeds"] = [776, 776, 1776, 2776, 3776]
    with pytest.raises(ConfigError, match="unique integers"):
        _resolve(tmp_path, duplicates)

    non_integer = _probe_config()
    non_integer["probe"]["seeds"] = [776.5]
    with pytest.raises(ConfigError, match="unique integers"):
        _resolve(tmp_path, non_integer)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("experiment", "verify_full_extraction_repeat"),
            False,
            "verify_full_extraction_repeat",
        ),
        (("data", "identity_split_seed"), 7, "identity_split_seed"),
        (
            ("data", "split_identity_counts"),
            {"train": 99, "dev": 51, "internal_eval": 50},
            "100/50/50",
        ),
        (
            ("probe", "seeds"),
            [776, 1776, 2776, 3776, 5776],
            "requires probe.seeds",
        ),
        (("probe", "input_dim"), 255, "input_dim=256"),
        (("probe", "output_dim"), 256, "output_dim=512"),
        (("probe", "epochs"), 119, "epochs=120"),
        (("probe", "batch_size"), 64, "batch_size=128"),
        (("probe", "dev_folds"), 4, "dev_folds=5"),
        (
            ("probe", "verify_repeat"),
            {"enabled": True, "layer": 1, "seed": 1776},
            "verify_repeat",
        ),
        (
            ("probe", "optimizer", "learning_rate"),
            0.0005,
            "optimizer.learning_rate=0.001",
        ),
        (
            ("probe", "optimizer", "betas"),
            [0.8, 0.999],
            r"optimizer.betas=\[0.9, 0.999\]",
        ),
        (
            ("probe", "optimizer", "weight_decay"),
            0.0,
            "optimizer.weight_decay=0.1",
        ),
        (
            ("probe", "optimizer", "polynomial_power"),
            2.0,
            "optimizer.polynomial_power=1.0",
        ),
        (("probe", "loss", "scale"), 32.0, "loss.scale=64.0"),
        (("probe", "loss", "margin"), 0.3, "loss.margin=0.4"),
        (("probe", "controls", "untrained_projection"), False, "untrained_projection"),
        (
            ("probe", "controls", "shuffled_labels"),
            {"enabled": True, "seeds": [1776]},
            "shuffled_labels enabled with seed 776",
        ),
        (
            ("probe", "controls", "gaussian_features"),
            {"enabled": True, "seeds": [1776]},
            "gaussian_features enabled with seed 776",
        ),
        (("evaluation", "bootstrap_resamples"), 500, "bootstrap_resamples=1000"),
        (("evaluation", "bootstrap_seed"), 7, "bootstrap_seed=20260902"),
    ],
)
def test_formal_e2_recipe_is_frozen(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    config = _formal_probe_config()
    _set_nested(config, path, value)
    with pytest.raises(ConfigError, match=message):
        _resolve(tmp_path, config)


@pytest.mark.parametrize(
    ("config_name", "expected_kind"),
    [
        ("smoke_fei_lvface_t.yaml", "layer_sweep"),
        ("smoke_fei_full_lvface_t.yaml", "layer_sweep"),
        ("fei_full_lvface_layers.yaml", "layer_sweep"),
        ("smoke_fei_full_lvface_probes.yaml", "equal_capacity_probes"),
        ("fei_full_lvface_probes.yaml", "equal_capacity_probes"),
    ],
)
def test_checked_in_configs_remain_compatible(
    config_name: str, expected_kind: str
) -> None:
    repository = Path(__file__).resolve().parents[1]
    resolved = resolve_experiment_config(repository / "configs" / config_name)
    assert resolved["experiment"]["kind"] == expected_kind


def test_config_mutations_do_not_leak_between_resolutions(tmp_path: Path) -> None:
    config = _probe_config()
    original = deepcopy(config)
    _resolve(tmp_path, config)
    assert config == original
