from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from id_layers.config import repository_root  # noqa: E402
from id_layers.generation import (  # noqa: E402
    PhotoMakerV2Backend,
    build_generation_plan,
    generation_environment,
    local_model_metadata,
    run_generation_plan,
    verify_reference_files,
)
from id_layers.utils import sha256_file, utc_run_id  # noqa: E402

FROZEN_IDENTITIES = [
    "fei_0058",
    "fei_0075",
    "fei_0092",
    "fei_0100",
    "fei_0115",
    "fei_0162",
    "fei_0168",
    "fei_0193",
]
FROZEN_CONDITIONS = [
    "k1_full",
    "k4_diverse_full",
    "k4_repeat_full",
    "global_target_patch_donor",
    "global_donor_patch_target",
    "text_only",
]
FROZEN_NUM_INFERENCE_STEPS = 30


def _path_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen local-only PhotoMaker V2 generation pilot"
    )
    parser.add_argument("--config", default="configs/generation_identity_pilot.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an explicit --run-id after validating plan and output hashes",
    )
    parser.add_argument(
        "--fallback-768",
        action="store_true",
        help="Use the preregistered 768x768 OOM fallback in a new run",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=FROZEN_CONDITIONS,
        default=None,
        help="Engineering-smoke filter; may be repeated. Default runs every condition.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Deterministic engineering-smoke prefix after --condition filtering",
    )
    parser.add_argument(
        "--face-provider",
        action="append",
        default=None,
        help="ONNX Runtime provider in priority order; may be repeated",
    )
    parser.add_argument("--photomaker-source", default="third_party/PhotoMaker")
    parser.add_argument(
        "--insightface-source", default="third_party/InsightFace/python-package"
    )
    parser.add_argument("--insightface-root", default="artifacts/cache/models/insightface")
    parser.add_argument(
        "--base-model", default="artifacts/cache/models/PhotoMaker/RealVisXL_V4.0"
    )
    parser.add_argument(
        "--adapter",
        default="artifacts/cache/models/PhotoMaker/PhotoMaker-V2/photomaker-v2.bin",
    )
    return parser


def _validate_frozen_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    generation = config["generation"]
    if data["eligible_split"] != "internal_eval":
        raise ValueError("Frozen pilot eligible_split must be internal_eval")
    if list(data["identities"]) != FROZEN_IDENTITIES:
        raise ValueError("Frozen pilot identity list or order changed")
    expected_donors = {
        identity: FROZEN_IDENTITIES[(index + 1) % len(FROZEN_IDENTITIES)]
        for index, identity in enumerate(FROZEN_IDENTITIES)
    }
    if dict(data["donor_mapping"]) != expected_donors:
        raise ValueError("Frozen cyclic donor mapping changed")
    if [item["name"] for item in generation["conditions"]] != FROZEN_CONDITIONS:
        raise ValueError("Frozen generation condition names or order changed")
    condition_by_name = {item["name"]: item for item in generation["conditions"]}
    conflict_expectations = {
        "global_target_patch_donor": ("target", "donor"),
        "global_donor_patch_target": ("donor", "target"),
    }
    for name, (global_identity, pixel_identity) in conflict_expectations.items():
        item = condition_by_name[name]
        if (
            item.get("global_identity") != global_identity
            or item.get("pixel_identity") != pixel_identity
            or item.get("global_reference_conditions") != "k1"
            or item.get("pixel_reference_conditions") != "k1"
        ):
            raise ValueError(f"Frozen K1 conflict definition changed for {name}")
    if condition_by_name["text_only"].get("start_merge_step") != int(
        model["num_inference_steps"]
    ):
        raise ValueError("text_only start_merge_step must equal num_inference_steps")
    if int(model["num_inference_steps"]) != FROZEN_NUM_INFERENCE_STEPS:
        raise ValueError(
            f"Frozen generation uses {FROZEN_NUM_INFERENCE_STEPS} inference steps"
        )
    if model["dtype"] != "float16" or model["device"] != "cuda":
        raise ValueError("Frozen generation requires CUDA float16")
    if model["batch_size"] != 1 or generation["num_images_per_prompt"] != 1:
        raise ValueError("Frozen generation requires batch size and images-per-prompt of one")
    if model["enable_model_cpu_offload"] is not True:
        raise ValueError("Frozen generation requires model CPU offload")
    if model["fuse_photomaker_lora"] is not True:
        raise ValueError("Frozen generation requires fused PhotoMaker LoRA")
    if generation["generator_device"] != "cuda":
        raise ValueError("Frozen actual latents must be generated on CUDA")
    expected = (
        len(data["identities"])
        * len(generation["conditions"])
        * len(generation["prompts"])
        * len(generation["seeds"])
    )
    if expected != int(experiment["expected_generated_images"]):
        raise ValueError("Frozen factorial size does not match expected_generated_images")


def _pilot_input_subset(manifest: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    data = config["data"]
    required_conditions = {
        *data["reference_conditions"]["k4_diverse"],
        *data["heldout_query_conditions"],
    }
    subset = manifest[
        manifest["identity_id"].astype(str).isin(data["identities"])
        & (manifest["split"].astype(str) == data["eligible_split"])
        & manifest["condition"].astype(str).isin(required_conditions)
    ].copy()
    expected = int(data["expected_required_manifest_rows"])
    if len(subset) != expected:
        raise ValueError(f"Expected {expected} frozen pilot input rows; observed {len(subset)}")
    counts = subset.groupby(["identity_id", "condition"]).size()
    if bool((counts != 1).any()) or len(counts) != expected:
        raise ValueError("Pilot input requires one row per frozen identity/condition")
    private = [
        column
        for column in ("absolute_path", "source_absolute_path")
        if column in subset
    ]
    subset = subset.drop(columns=private).sort_values(["identity_id", "acquisition_slot"])
    return json.loads(subset.to_json(orient="records"))


def _validate_locked_assets(
    model_metadata: dict[str, Any], lock: dict[str, Any], config: dict[str, Any]
) -> None:
    assets = config["assets"]
    photomaker_lock = lock["models"][assets["photomaker_asset_name"]]
    base_lock = lock["models"][assets["base_model_asset_name"]]
    insightface_lock = lock["models"][assets["conditioner_recognizer_asset_name"]]
    photomaker = model_metadata["photomaker_source"]
    insightface = model_metadata["insightface_source"]
    adapter = model_metadata["photomaker_adapter"]
    if (
        photomaker.get("revision") != photomaker_lock["code_revision"]
        or photomaker.get("dirty") is not False
    ):
        raise RuntimeError("PhotoMaker source does not match the clean locked revision")
    if (
        insightface.get("revision") != insightface_lock["code_revision"]
        or insightface.get("dirty") is not False
    ):
        raise RuntimeError("InsightFace source does not match the clean locked revision")
    if adapter["sha256"] != photomaker_lock["weight_sha256"]:
        raise RuntimeError("PhotoMaker adapter does not match its locked SHA256")
    base_files = {
        item["path"]: item["sha256"] for item in model_metadata["base_model"]["files"]
    }
    for relative, expected in base_lock["weight_sha256"].items():
        if base_files.get(relative) != expected:
            raise RuntimeError(f"RealVisXL file does not match lock: {relative}")
    insightface_files = {
        item["path"]: item["sha256"]
        for item in model_metadata["insightface_models"]["files"]
    }
    locked_face_files = {
        insightface_lock["detector_filename"]: insightface_lock["detector_sha256"],
        insightface_lock["recognition_filename"]: insightface_lock["recognition_sha256"],
    }
    for filename, expected in locked_face_files.items():
        if insightface_files.get(filename) != expected:
            raise RuntimeError(f"InsightFace file does not match lock: {filename}")


def _filter_engineering_smoke(
    tasks: list[Any], conditions: list[str] | None, max_tasks: int | None
) -> tuple[list[Any], dict[str, Any]]:
    selected = tasks
    if conditions:
        allowed = set(conditions)
        selected = [task for task in selected if task.condition in allowed]
    if max_tasks is not None:
        if max_tasks <= 0:
            raise ValueError("--max-tasks must be positive")
        selected = selected[:max_tasks]
    if not selected:
        raise ValueError("Engineering-smoke filters selected no tasks")
    return selected, {
        "filtered": len(selected) != len(tasks),
        "condition_filter": conditions,
        "max_tasks": max_tasks,
        "full_task_count": len(tasks),
        "executed_task_count": len(selected),
    }


def main() -> None:
    arguments = build_parser().parse_args()
    root = repository_root()
    config_path = _path_from_root(root, arguments.config)
    config = _load_yaml(config_path)
    _validate_frozen_config(config)
    if arguments.resume and arguments.run_id is None:
        raise SystemExit("--resume requires an explicit --run-id")

    data_config = config["data"]
    model_config = config["model"]
    generation_config = config["generation"]
    assets_config = config["assets"]
    manifest_path = _path_from_root(root, data_config["manifest_path"])
    data_root = _path_from_root(root, data_config["path_root"])
    output_root = _path_from_root(root, config["experiment"]["output_root"])
    lock_path = _path_from_root(root, assets_config["lock_file"])
    photomaker_source = _path_from_root(root, arguments.photomaker_source)
    insightface_source = _path_from_root(root, arguments.insightface_source)
    insightface_root = _path_from_root(root, arguments.insightface_root)
    base_model = _path_from_root(root, arguments.base_model)
    adapter_path = _path_from_root(root, arguments.adapter)

    manifest = pd.read_csv(manifest_path)
    identity_pairs = [
        (identity, data_config["donor_mapping"][identity])
        for identity in data_config["identities"]
    ]
    if arguments.fallback_768:
        fallback = config["fallbacks"]["cuda_out_of_memory"]["fallback_resolution"]
        width, height = map(int, fallback)
    else:
        width, height = int(model_config["width"]), int(model_config["height"])
    full_tasks = build_generation_plan(
        manifest,
        data_root,
        identity_pairs,
        dict(generation_config["prompts"]),
        str(model_config["negative_prompt"]),
        list(generation_config["seeds"]),
        eligible_split=str(data_config["eligible_split"]),
        k1_conditions=tuple(data_config["reference_conditions"]["k1"]),
        k4_diverse_conditions=tuple(
            data_config["reference_conditions"]["k4_diverse"]
        ),
        height=height,
        width=width,
        num_inference_steps=int(model_config["num_inference_steps"]),
        guidance_scale=float(model_config["guidance_scale"]),
        start_merge_step=int(generation_config["conditions"][0]["start_merge_step"]),
    )
    expected_tasks = int(config["experiment"]["expected_generated_images"])
    if len(full_tasks) != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} generation tasks; built {len(full_tasks)}")
    tasks, smoke_filter = _filter_engineering_smoke(
        full_tasks, arguments.condition, arguments.max_tasks
    )
    verified_references = verify_reference_files(tasks)
    models = local_model_metadata(
        photomaker_source=photomaker_source,
        insightface_source=insightface_source,
        insightface_root=insightface_root,
        base_model=base_model,
        adapter_path=adapter_path,
    )
    asset_lock = _load_yaml(lock_path)
    _validate_locked_assets(models, asset_lock, config)

    run_metadata = {
        "experiment": "photomaker_v2_generation_identity_pilot",
        "protocol_stage": (
            "filtered_engineering_smoke"
            if smoke_filter["filtered"]
            else "exploratory_internal_generation_diagnostic"
        ),
        "engineering_smoke_filter": smoke_filter,
        "resolution_fallback_768": bool(arguments.fallback_768),
        "resolved_config": config,
        "asset_lock": asset_lock,
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "asset_lock_path": lock_path.relative_to(root).as_posix(),
        "asset_lock_sha256": sha256_file(lock_path),
        "dataset_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
            "input_subset": _pilot_input_subset(manifest, config),
        },
        "verified_generation_references": verified_references,
        "models": models,
        "runner": {
            "generation_module_sha256": sha256_file(root / "src/id_layers/generation.py"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    providers = arguments.face_provider or [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    run_id = arguments.run_id or utc_run_id(
        "smoke_photomaker_v2_generation"
        if smoke_filter["filtered"]
        else "photomaker_v2_generation_pilot"
    )
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be one path-safe directory name")
    run_dir = output_root / run_id
    summary = run_generation_plan(
        tasks,
        run_dir,
        lambda: PhotoMakerV2Backend(
            tasks,
            photomaker_source=photomaker_source,
            insightface_source=insightface_source,
            insightface_root=insightface_root,
            base_model=base_model,
            adapter_path=adapter_path,
            providers=providers,
            dtype=str(model_config["dtype"]),
            cpu_offload=bool(model_config["enable_model_cpu_offload"]),
            fuse_lora=bool(model_config["fuse_photomaker_lora"]),
        ),
        run_metadata=run_metadata,
        environment=generation_environment(),
        resume=arguments.resume,
        latent_device=str(generation_config["generator_device"]),
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
