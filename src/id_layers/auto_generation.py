"""Frozen FEI development matrix and append-only automatic generation execution.

This is deliberately separate from the historical pilot runner.  In particular,
failed cells are retained on resume and no completed run is rewritten.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .generation import (
    BackendGeneration,
    GenerationBackend,
    GenerationTask,
    ReferenceRecord,
    _cuda_stats_finish,
    _cuda_stats_start,
    _generation_manifest_row,
    _safe_name,
    append_jsonl,
    make_fixed_latents,
    read_jsonl,
    tensor_sha256,
    verify_reference_files,
    write_jsonl,
)
from .utils import (
    python_package_lock,
    sha256_file,
    snapshot_research_sources,
    write_json,
    write_yaml,
)

SELECTION_SALT = "autoresearch20260905"
OLD_PILOT_IDENTITIES = (
    "fei_0058",
    "fei_0075",
    "fei_0092",
    "fei_0100",
    "fei_0115",
    "fei_0162",
    "fei_0168",
    "fei_0193",
)
GALLERY_CONDITIONS = (
    "frontal_smile",
    "left_three_quarter",
    "right_three_quarter_expression",
)
PROMPTS = {
    "neutral": (
        "a studio head-and-shoulders photograph of a person img, neutral expression, "
        "front view, even soft lighting, plain gray background"
    ),
    "smile": (
        "a studio head-and-shoulders photograph of a person img, smiling expression, "
        "front view, even soft lighting, plain gray background"
    ),
}
NEGATIVE_PROMPT = (
    "multiple people, duplicate person, duplicate face, cropped face, malformed face, "
    "deformed eyes, text, watermark, low resolution"
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _inside(child: Path, parent: Path) -> bool:
    return child.resolve().is_relative_to(parent.resolve())


def select_dev_identities(
    manifest: pd.DataFrame,
    *,
    count: int = 12,
    excluded_identities: Sequence[str] = OLD_PILOT_IDENTITIES,
) -> dict[str, Any]:
    """Rank IDs before reading images; selection never depends on usability/effects."""
    required = {"sample_id", "identity_id", "split"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if manifest["sample_id"].isna().any() or manifest["sample_id"].duplicated().any():
        raise ValueError("Missing or duplicate sample ID")
    if manifest["identity_id"].isna().any():
        raise ValueError("Missing identity ID")
    if (manifest.groupby("identity_id")["split"].nunique() != 1).any():
        raise ValueError("Identity crosses dataset splits")
    if count < 1:
        raise ValueError("Selection count must be positive")
    eligible = set(manifest.loc[manifest["split"] == "dev", "identity_id"].astype(str))
    eligible.difference_update(map(str, excluded_identities))
    ranked = sorted(
        (
            {
                "identity_id": item,
                "selection_sha256": hashlib.sha256(f"{SELECTION_SALT}:{item}".encode()).hexdigest(),
            }
            for item in eligible
        ),
        key=lambda item: (item["selection_sha256"], item["identity_id"]),
    )
    if len(ranked) < count:
        raise ValueError(f"Need {count} dev identities; found {len(ranked)}")
    return {
        "schema_version": 1,
        "evidence_scope": "FEI_development_engineering_only",
        "selection_formula": (
            "SHA256(UTF8('autoresearch20260905:' + identity_id)); "
            "ascending hexadecimal, identity_id tie-break"
        ),
        "selection_salt": SELECTION_SALT,
        "excluded_identities": sorted(set(excluded_identities)),
        "eligible_identity_count": len(ranked),
        "ranked_eligible_identities": ranked,
        "selected_identities": [item["identity_id"] for item in ranked[:count]],
        "replacement_policy": "No replacement of failed identities or cells",
    }


def prepare_generation_suite(
    manifest: pd.DataFrame,
    data_root: Path,
    *,
    excluded_identities: Sequence[str] = OLD_PILOT_IDENTITIES,
    identity_count: int = 12,
) -> tuple[list[GenerationTask], dict[str, Any], list[dict[str, Any]]]:
    """Return 4 cells per fixed ID, a selection manifest, and independent gallery."""
    selection = select_dev_identities(
        manifest, count=identity_count, excluded_identities=excluded_identities
    )
    required = {
        "condition",
        "protocol_role",
        "source_relative_path",
        "source_sha256",
        "acquisition_slot",
    }
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    tasks: list[GenerationTask] = []
    gallery: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    used_roles: dict[str, str] = {}
    used_hashes: dict[str, str] = {}
    for ordinal, identity in enumerate(selection["selected_identities"]):
        identity_rows = manifest.loc[manifest["identity_id"] == identity]
        records: dict[str, ReferenceRecord] = {}
        for condition in ("frontal_neutral", *GALLERY_CONDITIONS):
            matches = identity_rows.loc[identity_rows["condition"] == condition]
            if len(matches) != 1:
                raise ValueError(
                    f"Frozen selected identity {identity}: expected one {condition} row; "
                    "no replacement allowed"
                )
            row = matches.iloc[0]
            expected_role = "reference" if condition == "frontal_neutral" else "query"
            if row["protocol_role"] != expected_role or row["split"] != "dev":
                raise ValueError(f"Invalid photo role or split: {row['sample_id']}")
            source = (data_root / str(row["source_relative_path"])).resolve()
            if not _inside(source, data_root):
                raise ValueError("Source escapes data root")
            role = "generator_input" if expected_role == "reference" else "gallery"
            sample_id = str(row["sample_id"])
            source_hash = str(row["source_sha256"]).lower()
            if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
                raise ValueError(f"Invalid source SHA256 for {sample_id}")
            if sample_id in used_roles or source_hash in used_hashes:
                raise ValueError("A source ID or source hash is reused across photo roles")
            used_roles[sample_id] = role
            used_hashes[source_hash] = role
            record = ReferenceRecord(
                sample_id=sample_id,
                identity_id=identity,
                acquisition_slot=int(row["acquisition_slot"]),
                condition=condition,
                source_relative_path=str(row["source_relative_path"]),
                source_sha256=source_hash,
                source_path=source,
            )
            records[condition] = record
            # Native manifest fields (including detection failure) stay in the snapshot.
            snapshot = json.loads(matches.to_json(orient="records"))[0]
            snapshot["auto_research_photo_role"] = role
            inputs.append(snapshot)
            if role == "gallery":
                gallery.append(snapshot)
        reference = records["frontal_neutral"]
        for prompt_id, prompt in PROMPTS.items():
            for base_seed in (776, 1776):
                pair_id = f"{identity}__{prompt_id}__s{base_seed}"
                tasks.append(
                    GenerationTask(
                        task_id=f"{pair_id}__k1_full",
                        pair_id=pair_id,
                        target_identity_id=identity,
                        donor_identity_id=identity,
                        condition="k1_full",
                        prompt_id=prompt_id,
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        base_seed=base_seed,
                        effective_seed=base_seed + ordinal * 100_000,
                        height=1024,
                        width=1024,
                        num_inference_steps=30,
                        guidance_scale=5.0,
                        start_merge_step=10,
                        patch_references=(reference,),
                        global_references=(reference,),
                    )
                )
    selection.update(
        {
            "expected_cells": len(tasks),
            "input_subset": inputs,
            "generator_input_count": identity_count,
            "gallery_image_count": len(gallery),
            "base_seeds": [776, 1776],
            "effective_seed_rule": "base_seed + selected_identity_ordinal * 100000",
            "prompts": PROMPTS,
            "single_factor": "neutral expression -> smiling expression",
            "human_ratings": "none; self-rated D/E cannot be calculated from these outputs",
        }
    )
    return tasks, selection, gallery


def smoke_tasks(tasks: Sequence[GenerationTask], selection: dict[str, Any]) -> list[GenerationTask]:
    """Fixed first-four-identity PuLID smoke; full matrix must use a separate run."""
    identities = set(selection["selected_identities"][:4])
    subset = [task for task in tasks if task.target_identity_id in identities]
    if len(subset) != 16:
        raise ValueError("PuLID smoke requires exactly four selected identities / 16 cells")
    return subset


def _checked_existing_rows(
    path: Path, tasks: dict[str, GenerationTask], run_dir: Path
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id not in tasks or sample_id in by_id:
            raise ValueError("Existing generation manifest has unknown or duplicate ID")
        task = tasks[sample_id]
        for field, expected in (
            ("identity_id", task.target_identity_id),
            ("prompt_id", task.prompt_id),
            ("condition", task.condition),
            ("base_seed", task.base_seed),
            ("effective_seed", task.effective_seed),
        ):
            if row.get(field) != expected:
                raise ValueError(f"Existing cell pairing mismatch: {sample_id}/{field}")
        if row["status"] == "complete":
            output = (run_dir / str(row["output_relative_path"])).resolve()
            if not _inside(output, run_dir) or not output.is_file():
                raise ValueError("Completed output missing or escapes run directory")
            if sha256_file(output) != row["output_sha256"]:
                raise ValueError(f"Completed output hash mismatch: {sample_id}")
        elif row["status"] not in {"failed", "budget_stopped", "blocked_external", "not_started"}:
            raise ValueError(f"Unknown cell status: {row['status']}")
        by_id[sample_id] = row
    return by_id


def run_generation_cells(
    tasks: Sequence[GenerationTask],
    run_dir: Path,
    backend_factory: Callable[[], GenerationBackend],
    *,
    run_metadata: dict[str, Any],
    forbidden_source: Path,
    resume: bool = False,
    latent_device: str = "cpu",
    can_start: Callable[[float], bool] | None = None,
    record_usage: Callable[[dict[str, Any]], None] | None = None,
    estimated_cell_seconds: float = 30.0,
    estimated_load_seconds: float = 120.0,
) -> dict[str, Any]:
    """Execute once per cell; resume only cells that have never been attempted.

    ``can_start`` receives a conservative next-operation time estimate and the
    root budget ledger decides whether to start it. ``record_usage`` receives
    actual serial wall time and peak CUDA memory after load and each cell.
    Terminal completed runs are verified and returned without writing any file.
    """
    run_dir = run_dir.resolve()
    if _inside(run_dir, forbidden_source):
        raise ValueError("Output path is inside immutable source run")
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Empty plan or duplicate task ID")
    for task in tasks:
        _safe_name(task.task_id, "auto-generation task ID")
    if min(estimated_cell_seconds, estimated_load_seconds) <= 0:
        raise ValueError("Budget estimates must be positive")
    # Canonicalize fingerprints/execution independently of caller row order.
    ordered = sorted(tasks, key=lambda task: task.task_id)
    by_id = {task.task_id: task for task in ordered}
    fingerprint = hashlib.sha256(
        _json_bytes(
            {
                "schema_version": 1,
                "tasks": [t.provenance() for t in ordered],
                "run_metadata": {k: v for k, v in run_metadata.items() if k != "environment"},
                "latent_device": latent_device,
            }
        )
    ).hexdigest()
    status_path = run_dir / "status.json"
    manifest_path = run_dir / "generation_manifest.jsonl"
    events_path = run_dir / "run_events.jsonl"
    if resume:
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous["plan_sha256"] != fingerprint:
            raise ValueError("Resume plan hash mismatch")
        rows = _checked_existing_rows(manifest_path, by_id, run_dir)
        if previous["state"] in {"complete", "complete_with_failures"}:
            if len(rows) != len(tasks) or any(
                r["status"] not in {"complete", "failed"} for r in rows.values()
            ):
                raise ValueError("Terminal status does not match cell ledger")
            return previous
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        rows = {}
        write_json(
            run_dir / "plan.json",
            {
                "plan_sha256": fingerprint,
                "latent_device": latent_device,
                "run_metadata": run_metadata,
                "tasks": [t.provenance() for t in ordered],
            },
        )
        for key, filename in (
            ("resolved_config", "config.resolved.yaml"),
            ("asset_lock", "assets.lock.resolved.yaml"),
        ):
            if isinstance(run_metadata.get(key), dict):
                write_yaml(run_dir / filename, run_metadata[key])
        if isinstance(run_metadata.get("input_subset"), list):
            pd.DataFrame(run_metadata["input_subset"]).to_csv(
                run_dir / "dataset_manifest.csv", index=False
            )
        if "environment" in run_metadata:
            write_json(run_dir / "environment.json", run_metadata["environment"])
        if run_metadata.get("snapshot_sources", True):
            repository = Path(__file__).resolve().parents[2]
            source_manifest = snapshot_research_sources(repository, run_dir / "source_snapshot")
            write_json(run_dir / "source_snapshot_manifest.json", source_manifest)
            (run_dir / "environment.lock.txt").write_text(python_package_lock(), encoding="utf-8")
        append_jsonl(
            events_path,
            {"event": "run_started", "timestamp_utc": _now(), "plan_sha256": fingerprint},
        )

    def cell_stub(task: GenerationTask, state: str, reason: str) -> dict[str, Any]:
        return {
            "sample_id": task.task_id,
            "identity_id": task.target_identity_id,
            "prompt_id": task.prompt_id,
            "condition": task.condition,
            "base_seed": task.base_seed,
            "effective_seed": task.effective_seed,
            "status": state,
            "reason": reason,
            "output_sha256": None,
            "output_relative_path": None,
            "attempt": 0,
        }

    def persist(state: str, reason: str | None = None) -> dict[str, Any]:
        for task in ordered:
            if task.task_id not in rows:
                rows[task.task_id] = cell_stub(task, "not_started", "Awaiting execution")
        write_jsonl(manifest_path, [rows[task.task_id] for task in ordered])
        counts = {
            key: sum(row["status"] == key for row in rows.values())
            for key in ("complete", "failed", "budget_stopped", "blocked_external", "not_started")
        }
        result = {
            "state": state,
            "plan_sha256": fingerprint,
            "planned": len(tasks),
            **counts,
            "updated_at_utc": _now(),
            "reason": reason,
        }
        write_json(status_path, result)
        return result

    pending = [
        t for t in ordered if rows.get(t.task_id, {}).get("status") not in {"complete", "failed"}
    ]
    persist("running")
    if not pending:
        return persist(
            "complete_with_failures"
            if any(r["status"] == "failed" for r in rows.values())
            else "complete"
        )
    if can_start and not can_start(estimated_load_seconds + estimated_cell_seconds):
        for task in pending:
            rows[task.task_id] = cell_stub(
                task, "budget_stopped", "Insufficient budget for backend and next cell"
            )
        return persist("budget_stopped")
    started = time.perf_counter()
    load_stats = _cuda_stats_start()
    backend = None
    try:
        verify_reference_files(pending)
        backend = backend_factory()
        load_stats = _cuda_stats_finish(load_stats)
        load_usage = {
            "unit": "backend_load",
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_memory": load_stats,
            "run_dir": str(run_dir),
        }
        append_jsonl(
            events_path, {"event": "backend_ready", "backend": backend.provenance(), **load_usage}
        )
        if record_usage:
            record_usage(load_usage)
    except Exception as error:
        load_usage = {
            "unit": "backend_load_failed",
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_memory": _cuda_stats_finish(load_stats),
            "run_dir": str(run_dir),
        }
        if record_usage:
            record_usage(load_usage)
        append_jsonl(
            events_path,
            {
                "event": "backend_failed",
                **load_usage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        for task in pending:
            rows[task.task_id] = cell_stub(
                task, "blocked_external", f"{type(error).__name__}: {error}"
            )
        return persist("blocked_external", str(error))

    try:
        for task in pending:
            if can_start and not can_start(estimated_cell_seconds):
                for waiting in pending:
                    if rows[waiting.task_id]["status"] not in {"complete", "failed"}:
                        rows[waiting.task_id] = cell_stub(
                            waiting, "budget_stopped", "Insufficient budget for next fixed cell"
                        )
                return persist("budget_stopped")
            started = time.perf_counter()
            stats = _cuda_stats_start()
            latent_hash = ""
            try:
                latents = make_fixed_latents(
                    task.effective_seed, task.height, task.width, device=latent_device
                )
                latent_hash = tensor_sha256(latents)
                append_jsonl(
                    events_path,
                    {
                        "event": "cell_started",
                        "sample_id": task.task_id,
                        "latent_sha256": latent_hash,
                        "timestamp_utc": _now(),
                    },
                )
                with torch.inference_mode():
                    output = backend.generate(task, latents.clone())
                if not isinstance(output, BackendGeneration):
                    raise TypeError("Backend did not return BackendGeneration")
                image = output.image.convert("RGB")
                if image.size != (task.width, task.height):
                    raise ValueError("Generated dimensions do not match frozen task")
                if not np.any(np.ptp(np.asarray(image).reshape(-1, 3), axis=0)):
                    raise ValueError(
                        "Generated image is exactly constant; possible decoding failure"
                    )
                relative = Path("images") / f"{task.task_id}.png"
                destination = run_dir / relative
                destination.parent.mkdir(exist_ok=True)
                if destination.exists():
                    raise FileExistsError("Unaccounted output exists; refusing overwrite")
                temporary = destination.with_suffix(".png.tmp")
                image.save(temporary, format="PNG")
                temporary.replace(destination)
                stats = _cuda_stats_finish(stats)
                row = _generation_manifest_row(
                    task,
                    attempt=1,
                    status="complete",
                    elapsed_seconds=time.perf_counter() - started,
                    cuda_memory=stats,
                    latent_sha256=latent_hash,
                    output_relative_path=relative.as_posix(),
                    output_sha256=sha256_file(destination),
                    backend_metadata=output.metadata,
                )
            except Exception as error:
                stats = _cuda_stats_finish(stats)
                row = _generation_manifest_row(
                    task,
                    attempt=1,
                    status="failed",
                    elapsed_seconds=time.perf_counter() - started,
                    cuda_memory=stats,
                    latent_sha256=latent_hash,
                    output_relative_path=None,
                    output_sha256=None,
                    backend_metadata=None,
                    error=error,
                )
                row["traceback"] = traceback.format_exc()
            rows[task.task_id] = row
            append_jsonl(events_path, {"event": "cell_finished", **row})
            # Persist the irreversible attempt before calling external budget accounting.
            persist("running")
            if record_usage:
                record_usage(
                    {
                        "unit": "cell",
                        "sample_id": task.task_id,
                        "elapsed_seconds": row["elapsed_seconds"],
                        "cuda_memory": stats,
                        "run_dir": str(run_dir),
                        "status": row["status"],
                    }
                )
    finally:
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return persist(
        "complete_with_failures"
        if any(r["status"] == "failed" for r in rows.values())
        else "complete"
    )


class PuLIDV11Backend:
    """Adapter around an already initialized official SDXL v1.1 pipeline.

    Initialization/downloads belong to the root budget-aware environment stage.
    The official inference API samples CPU noise internally; we verify exact
    equality with the audited supplied latent and never claim unused noise was
    used. Its DPM++ schedule and ID-specific settings remain explicit differences.
    """

    def __init__(
        self,
        pipeline: Any,
        tasks: Sequence[GenerationTask],
        *,
        source_revision: str,
        source_path: Path,
        id_scale: float = 0.8,
        steps: int = 25,
        guidance_scale: float = 7.0,
        num_zero: int = 20,
        ortho_v2: bool = True,
    ) -> None:
        if len(source_revision) != 40:
            raise ValueError("PuLID source must be pinned to a full commit")
        self.pipeline = pipeline
        self.settings = {
            "id_scale": id_scale,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "num_zero": num_zero,
            "ortho_v2": ortho_v2,
            "source_revision": source_revision,
            "source_path": str(source_path.resolve()),
        }
        from pulid import attention_processor
        from pulid.utils import resize_numpy_image_long

        attention_processor.NUM_ZERO = num_zero
        attention_processor.ORTHO = False
        attention_processor.ORTHO_v2 = ortho_v2
        self.embeddings: dict[str, Any] = {}
        for task in tasks:
            if len(task.patch_references) != 1 or task.patch_references != task.global_references:
                raise ValueError("PuLID C adapter accepts only the frozen K1 common-source path")
            reference = task.patch_references[0]
            if reference.sample_id in self.embeddings:
                continue
            image = np.asarray(Image.open(reference.source_path).convert("RGB"))
            with torch.inference_mode():
                self.embeddings[reference.sample_id] = pipeline.get_id_embedding(
                    [resize_numpy_image_long(image, 1024)]
                )
            pipeline.debug_img_list = []

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "official_PuLID_v1.1_SDXL",
            **self.settings,
            "sampler": self.pipeline.sampler.__name__,
            "dtype": str(self.pipeline.pipe.unet.dtype),
            "vae_dtype": str(self.pipeline.pipe.vae.dtype),
            "inactive_reference_encoders_released_after_caching": getattr(
                self, "inactive_encoders_released", False
            ),
            "reference_face_analysis_modules": getattr(
                self, "face_analysis_modules", "upstream_default"
            ),
            "vae_decode_residency": getattr(self, "vae_decode_residency", "sdxl_models_resident"),
            "latent_rng": "CPU torch.Generator; bitwise-verified against supplied noise",
            "prompt_difference": "Remove PhotoMaker-only ' img' trigger",
            "reference_preprocessing": (
                "Official resize_numpy_image_long(...,1024), antelopev2, facexlib, EVA-CLIP"
            ),
            "base_compatibility": (
                "RealVisXL is an engineering test, not an officially validated base in app_v1_1.py"
            ),
        }

    def generate(self, task: GenerationTask, latents: torch.Tensor) -> BackendGeneration:
        expected = make_fixed_latents(task.effective_seed, task.height, task.width, device="cpu")
        if not torch.equal(latents.detach().cpu(), expected):
            raise ValueError(
                "PuLID official inference internally uses CPU noise; supplied latent does not match"
            )
        uncond, cond = self.embeddings[task.patch_references[0].sample_id]
        prompt = task.prompt.replace("person img", "person")
        with torch.inference_mode():
            images = self.pipeline.inference(
                prompt=prompt,
                size=(1, task.height, task.width),
                prompt_n=task.negative_prompt,
                id_embedding=cond,
                uncond_id_embedding=uncond,
                id_scale=self.settings["id_scale"],
                guidance_scale=self.settings["guidance_scale"],
                steps=self.settings["steps"],
                seed=task.effective_seed,
            )
        if len(images) != 1:
            raise ValueError("PuLID did not return exactly one image")
        return BackendGeneration(
            images[0],
            {
                **self.provenance(),
                "actual_prompt": prompt,
                "actual_steps": self.settings["steps"],
                "actual_guidance_scale": self.settings["guidance_scale"],
                "latent_sha256": tensor_sha256(expected),
            },
        )


def pulid_dependency_inventory() -> dict[str, Any]:
    """Read-only package probe; no downloads, imports with model side effects, or GPU."""
    modules = (
        "diffusers",
        "transformers",
        "timm",
        "einops",
        "ftfy",
        "facexlib",
        "basicsr",
        "insightface",
        "onnxruntime",
        "safetensors",
        "torchsde",
    )
    available = {}
    for module in modules:
        try:
            available[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available[module] = False
    return {
        "model": "PuLID-v1.1-SDXL",
        "available_modules": available,
        "missing_modules": [k for k, v in available.items() if not v],
        "official_source": "https://github.com/ToTheBeginning/PuLID",
        "weight_url": "https://huggingface.co/guozinan/PuLID/blob/main/pulid_v1.1.safetensors",
        "weight_sha256": "4cb8ceec1078e0165399b88332ab3c5971619111b8e1730e6bae64144aabae41",
        "required_other_assets": [
            "EVA02_CLIP_L_336_psz14_s6B.pt",
            "antelopev2/glintr100.onnx",
            "antelopev2 detector",
            "facexlib retinaface_resnet50",
            "facexlib bisenet",
        ],
        "official_requirements_warning": (
            "Do not install both onnxruntime and onnxruntime-gpu; "
            "upstream pins are not the research environment lock"
        ),
        "compatibility_time_limit_seconds": 3600,
        "realvis_status": (
            "The official demo comments out RealVisXL_V4.0; "
            "runtime compatibility needs smoke verification"
        ),
    }
