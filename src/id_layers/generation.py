from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .utils import (
    git_revision,
    python_package_lock,
    sha256_bytes,
    sha256_file,
    snapshot_research_sources,
    write_json,
    write_yaml,
)

GENERATION_SCHEMA_VERSION = 1
DEFAULT_K1_CONDITIONS = ("frontal_neutral",)
DEFAULT_K4_DIVERSE_CONDITIONS = (
    "frontal_neutral",
    "left_profile",
    "right_profile",
    "frontal_scale",
)
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TRIGGER_WORD_PATTERN = re.compile(r"(?<![A-Za-z0-9_])img(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class ReferenceRecord:
    sample_id: str
    identity_id: str
    acquisition_slot: int
    condition: str
    source_relative_path: str
    source_sha256: str
    source_path: Path

    def provenance(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "identity_id": self.identity_id,
            "acquisition_slot": self.acquisition_slot,
            "condition": self.condition,
            "source_relative_path": self.source_relative_path,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class GenerationTask:
    task_id: str
    pair_id: str
    target_identity_id: str
    donor_identity_id: str
    condition: str
    prompt_id: str
    prompt: str
    negative_prompt: str
    base_seed: int
    effective_seed: int
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    start_merge_step: int
    patch_references: tuple[ReferenceRecord, ...]
    global_references: tuple[ReferenceRecord, ...]
    text_only: bool = False

    def provenance(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pair_id": self.pair_id,
            "target_identity_id": self.target_identity_id,
            "donor_identity_id": self.donor_identity_id,
            "condition": self.condition,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "base_seed": self.base_seed,
            "effective_seed": self.effective_seed,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "start_merge_step": self.start_merge_step,
            "patch_references": [item.provenance() for item in self.patch_references],
            "global_references": [item.provenance() for item in self.global_references],
            "text_only": self.text_only,
        }


@dataclass(frozen=True)
class BackendGeneration:
    image: Image.Image
    metadata: dict[str, Any]


class GenerationBackend(Protocol):
    def generate(self, task: GenerationTask, latents: torch.Tensor) -> BackendGeneration:
        ...

    def provenance(self) -> dict[str, Any]:
        ...


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_name(value: str, field: str) -> str:
    if not SAFE_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must match {SAFE_NAME_PATTERN.pattern!r}; received {value!r}"
        )
    return value


def _manifest_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Invalid boolean value in {field}: {value!r}")


def _resolve_source(data_root: Path, relative: str) -> Path:
    root = data_root.resolve()
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Source path escapes data root: {relative!r}") from error
    return candidate


def _reference_rows(
    manifest: pd.DataFrame,
    data_root: Path,
    identity_id: str,
    conditions: Sequence[str],
    eligible_split: str,
) -> tuple[ReferenceRecord, ...]:
    required_columns = {
        "sample_id",
        "identity_id",
        "acquisition_slot",
        "condition",
        "protocol_role",
        "source_relative_path",
        "source_sha256",
        "split",
        "usable_for_model",
    }
    missing = sorted(required_columns.difference(manifest.columns))
    if missing:
        raise ValueError(f"Generation manifest is missing columns: {missing}")
    if bool(manifest["sample_id"].duplicated().any()):
        duplicates = sorted(manifest.loc[manifest["sample_id"].duplicated(), "sample_id"].unique())
        raise ValueError(f"Generation manifest has duplicate sample_id values: {duplicates[:5]}")

    identity_rows = manifest[
        (manifest["identity_id"].astype(str) == identity_id)
        & (manifest["split"].astype(str) == eligible_split)
        & (manifest["protocol_role"].astype(str) == "reference")
    ].copy()
    if identity_rows.empty:
        raise ValueError(
            f"Identity {identity_id!r} has no {eligible_split}/reference rows"
        )
    identity_rows["_usable"] = identity_rows["usable_for_model"].map(
        lambda value: _manifest_bool(value, "usable_for_model")
    )

    records: list[ReferenceRecord] = []
    for condition in conditions:
        rows = identity_rows[identity_rows["condition"].astype(str) == str(condition)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one {eligible_split}/reference row for "
                f"identity={identity_id!r}, condition={condition!r}; observed {len(rows)}"
            )
        row = rows.iloc[0]
        if not bool(row["_usable"]):
            raise ValueError(
                f"Reference is not usable: identity={identity_id!r}, condition={condition!r}"
            )
        relative = str(row["source_relative_path"])
        records.append(
            ReferenceRecord(
                sample_id=str(row["sample_id"]),
                identity_id=identity_id,
                acquisition_slot=int(row["acquisition_slot"]),
                condition=str(condition),
                source_relative_path=relative,
                source_sha256=str(row["source_sha256"]).lower(),
                source_path=_resolve_source(data_root, relative),
            )
        )
    return tuple(records)


def _task_id(
    target: str,
    donor: str,
    prompt_id: str,
    seed: int,
    condition: str,
) -> str:
    values = [target, donor, prompt_id, f"s{seed}", condition]
    for value in values:
        _safe_name(value, "task identifier component")
    return "__".join(values)


def build_generation_plan(
    manifest: pd.DataFrame,
    data_root: Path,
    identity_pairs: Sequence[tuple[str, str]],
    prompts: dict[str, str],
    negative_prompt: str,
    base_seeds: Sequence[int],
    *,
    eligible_split: str = "internal_eval",
    k1_conditions: Sequence[str] = DEFAULT_K1_CONDITIONS,
    k4_diverse_conditions: Sequence[str] = DEFAULT_K4_DIVERSE_CONDITIONS,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    start_merge_step: int = 10,
) -> list[GenerationTask]:
    """Build the fixed six-condition PhotoMaker pilot without touching the models."""
    if not identity_pairs:
        raise ValueError("At least one target/donor identity pair is required")
    if not prompts:
        raise ValueError("At least one prompt is required")
    if not base_seeds:
        raise ValueError("At least one base seed is required")
    if not eligible_split:
        raise ValueError("eligible_split must be non-empty")
    if tuple(k1_conditions) != ("frontal_neutral",):
        raise ValueError("The frozen K1 condition must be exactly frontal_neutral")
    if len(k4_diverse_conditions) != 4 or len(set(k4_diverse_conditions)) != 4:
        raise ValueError("The frozen K4 diverse set must contain four unique conditions")
    if height <= 0 or width <= 0 or height % 8 or width % 8:
        raise ValueError("Generation height and width must be positive multiples of 8")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if start_merge_step < 0 or start_merge_step >= num_inference_steps:
        raise ValueError("start_merge_step must be in [0, num_inference_steps)")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")

    for prompt_id, prompt in prompts.items():
        _safe_name(str(prompt_id), "prompt_id")
        prompt_text = str(prompt)
        matches = list(TRIGGER_WORD_PATTERN.finditer(prompt_text))
        if len(matches) != 1:
            raise ValueError(
                f"Prompt {prompt_id!r} must contain the PhotoMaker trigger word 'img' exactly once"
            )
        prefix_words = re.findall(r"[A-Za-z0-9_-]+", prompt_text[: matches[0].start()])
        if not prefix_words:
            raise ValueError(f"Prompt {prompt_id!r} must put a class word before 'img'")

    tasks: list[GenerationTask] = []
    seen_pairs: set[tuple[str, str]] = set()
    for identity_ordinal, (target, donor) in enumerate(identity_pairs):
        target = _safe_name(str(target), "target identity")
        donor = _safe_name(str(donor), "donor identity")
        if target == donor:
            raise ValueError("Target and donor identities must differ")
        if (target, donor) in seen_pairs:
            raise ValueError(f"Duplicate target/donor identity pair: {(target, donor)!r}")
        seen_pairs.add((target, donor))

        target_k1 = _reference_rows(
            manifest, data_root, target, k1_conditions, eligible_split
        )
        target_k4 = _reference_rows(
            manifest, data_root, target, k4_diverse_conditions, eligible_split
        )
        donor_k1 = _reference_rows(
            manifest, data_root, donor, k1_conditions, eligible_split
        )
        target_repeat = target_k1 * 4

        conditions = [
            ("k1_full", target_k1, target_k1, False, start_merge_step),
            ("k4_diverse_full", target_k4, target_k4, False, start_merge_step),
            ("k4_repeat_full", target_repeat, target_repeat, False, start_merge_step),
            (
                "global_target_patch_donor",
                donor_k1,
                target_k1,
                False,
                start_merge_step,
            ),
            (
                "global_donor_patch_target",
                target_k1,
                donor_k1,
                False,
                start_merge_step,
            ),
            ("text_only", target_k1, target_k1, True, num_inference_steps),
        ]
        for prompt_id, prompt in prompts.items():
            for seed_value in base_seeds:
                base_seed = int(seed_value)
                if base_seed < 0:
                    raise ValueError("Base seeds must be non-negative")
                effective_seed = base_seed + identity_ordinal * 100_000
                pair_id = "__".join(
                    (target, donor, str(prompt_id), f"s{base_seed}", f"e{effective_seed}")
                )
                for (
                    condition,
                    patch_refs,
                    global_refs,
                    text_only,
                    task_start_merge_step,
                ) in conditions:
                    tasks.append(
                        GenerationTask(
                            task_id=_task_id(
                                target, donor, str(prompt_id), base_seed, condition
                            ),
                            pair_id=pair_id,
                            target_identity_id=target,
                            donor_identity_id=donor,
                            condition=condition,
                            prompt_id=str(prompt_id),
                            prompt=str(prompt),
                            negative_prompt=str(negative_prompt),
                            base_seed=base_seed,
                            effective_seed=effective_seed,
                            height=int(height),
                            width=int(width),
                            num_inference_steps=int(num_inference_steps),
                            guidance_scale=float(guidance_scale),
                            start_merge_step=int(task_start_merge_step),
                            patch_references=tuple(patch_refs),
                            global_references=tuple(global_refs),
                            text_only=bool(text_only),
                        )
                    )

    identifiers = [task.task_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Generation plan contains duplicate task IDs")
    return tasks


def unique_references(tasks: Sequence[GenerationTask]) -> tuple[ReferenceRecord, ...]:
    records: dict[str, ReferenceRecord] = {}
    for task in tasks:
        for record in (*task.patch_references, *task.global_references):
            previous = records.setdefault(record.sample_id, record)
            if previous != record:
                raise ValueError(f"Conflicting provenance for sample {record.sample_id!r}")
    return tuple(records[key] for key in sorted(records))


def verify_reference_files(tasks: Sequence[GenerationTask]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for record in unique_references(tasks):
        if not record.source_path.is_file():
            raise FileNotFoundError(f"Missing reference source: {record.source_path}")
        observed = sha256_file(record.source_path)
        if observed.lower() != record.source_sha256.lower():
            raise RuntimeError(
                f"Reference SHA256 mismatch for {record.sample_id}: "
                f"{observed} != {record.source_sha256}"
            )
        verified.append({**record.provenance(), "observed_sha256": observed})
    return verified


def make_fixed_latents(
    seed: int,
    height: int,
    width: int,
    *,
    batch_size: int = 1,
    latent_channels: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if seed < 0:
        raise ValueError("Latent seed must be non-negative")
    if height <= 0 or width <= 0 or height % 8 or width % 8:
        raise ValueError("Latent height and width must be positive multiples of 8")
    if batch_size <= 0 or latent_channels <= 0:
        raise ValueError("batch_size and latent_channels must be positive")
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(int(seed))
    return torch.randn(
        (batch_size, latent_channels, height // 8, width // 8),
        generator=generator,
        dtype=torch.float32,
        device=target_device,
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(record).decode("utf-8") + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical_json(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def upsert_generation_manifest(path: Path, row: dict[str, Any]) -> None:
    sample_id = str(row.get("sample_id", ""))
    if not sample_id:
        raise ValueError("Generation manifest row requires sample_id")
    records = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        existing_id = str(record.get("sample_id", ""))
        if not existing_id or existing_id in by_id:
            raise RuntimeError("Existing generation manifest has missing or duplicate sample_id")
        by_id[existing_id] = record
        order.append(existing_id)
    if sample_id not in by_id:
        order.append(sample_id)
    by_id[sample_id] = row
    write_jsonl(path, [by_id[item] for item in order])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"Blank JSONL record at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed JSONL record at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"Non-object JSONL record at {path}:{line_number}")
            records.append(value)
    return records


def plan_sha256(tasks: Sequence[GenerationTask], run_metadata: dict[str, Any]) -> str:
    value = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "run_metadata": run_metadata,
        "tasks": [task.provenance() for task in tasks],
    }
    return sha256_bytes(_canonical_json(value))


def _cuda_stats_start() -> dict[str, int | bool]:
    available = torch.cuda.is_available()
    if not available:
        return {"cuda_available": False}
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    return {
        "cuda_available": True,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
    }


def _cuda_stats_finish(start: dict[str, int | bool]) -> dict[str, int | bool]:
    if not bool(start["cuda_available"]):
        return start
    torch.cuda.synchronize()
    return {
        **start,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _output_relative_path(task: GenerationTask) -> Path:
    return Path("images") / task.pair_id / f"{task.condition}.png"


def _save_image_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _completed_tasks(
    manifest_rows: Sequence[dict[str, Any]],
    tasks_by_id: dict[str, GenerationTask],
    run_dir: Path,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        if row.get("status") != "complete":
            continue
        task_id = str(row.get("sample_id", ""))
        if task_id not in tasks_by_id:
            raise RuntimeError(f"Provenance contains an unknown completed task: {task_id!r}")
        if task_id in completed:
            raise RuntimeError(f"Provenance contains duplicate completion for task {task_id!r}")
        relative = Path(str(row.get("output_relative_path", "")))
        output_path = (run_dir / relative).resolve()
        try:
            output_path.relative_to(run_dir.resolve())
        except ValueError as error:
            raise RuntimeError(f"Completed output escapes run directory: {relative}") from error
        if not output_path.is_file():
            raise RuntimeError(f"Completed output is missing for task {task_id!r}: {relative}")
        observed = sha256_file(output_path)
        expected = str(row.get("output_sha256", ""))
        if observed != expected:
            raise RuntimeError(
                f"Completed output SHA256 mismatch for task {task_id!r}: {observed} != {expected}"
            )
        completed[task_id] = row
    return completed


def _attempt_number(events: Sequence[dict[str, Any]], task_id: str) -> int:
    return 1 + sum(
        event.get("event") == "task_started" and event.get("task_id") == task_id
        for event in events
    )


def _reference_identity(task: GenerationTask, *, global_embedding: bool) -> str | None:
    if task.text_only:
        return None
    references = task.global_references if global_embedding else task.patch_references
    identities = {item.identity_id for item in references}
    if len(identities) != 1:
        kind = "global embedding" if global_embedding else "patch image"
        raise RuntimeError(f"Task {task.task_id} has multiple {kind} identities: {identities}")
    return next(iter(identities))


def _generation_manifest_row(
    task: GenerationTask,
    *,
    attempt: int,
    status: str,
    elapsed_seconds: float,
    cuda_memory: dict[str, Any],
    latent_sha256: str,
    output_relative_path: str | None,
    output_sha256: str | None,
    backend_metadata: dict[str, Any] | None,
    error: Exception | None = None,
) -> dict[str, Any]:
    patch_ids = [item.sample_id for item in task.patch_references]
    global_ids = [item.sample_id for item in task.global_references]
    reference_ids = patch_ids if patch_ids == global_ids else [*global_ids, *patch_ids]
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "sample_id": task.task_id,
        "identity_id": task.target_identity_id,
        "donor_identity_id": task.donor_identity_id,
        "condition": task.condition,
        "prompt_id": task.prompt_id,
        "prompt": task.prompt,
        "negative_prompt": task.negative_prompt,
        "base_seed": task.base_seed,
        "effective_seed": task.effective_seed,
        "latent_sha256": latent_sha256,
        "output_relative_path": output_relative_path,
        "output_sha256": output_sha256,
        "status": status,
        "attempt": attempt,
        "reference_sample_ids": reference_ids,
        "patch_reference_sample_ids": patch_ids,
        "global_reference_sample_ids": global_ids,
        "global_embedding_identity_id": _reference_identity(task, global_embedding=True),
        "patch_image_identity_id": _reference_identity(task, global_embedding=False),
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_bytes": cuda_memory.get("peak_allocated_bytes"),
        "peak_cuda_reserved_bytes": cuda_memory.get("peak_reserved_bytes"),
        "cuda_memory": cuda_memory,
        "height": task.height,
        "width": task.width,
        "num_inference_steps": task.num_inference_steps,
        "guidance_scale": task.guidance_scale,
        "start_merge_step": task.start_merge_step,
        "backend_metadata": backend_metadata or {},
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "timestamp_utc": _utc_now(),
    }


def run_generation_plan(
    tasks: Sequence[GenerationTask],
    run_dir: Path,
    backend_factory: Callable[[], GenerationBackend],
    *,
    run_metadata: dict[str, Any],
    environment: dict[str, Any] | None = None,
    resume: bool = False,
    latent_device: str = "cpu",
) -> dict[str, Any]:
    """Execute a plan transactionally and retain an append-only, resumable JSONL log."""
    if not tasks:
        raise ValueError("Generation plan is empty")
    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("Generation plan has duplicate task IDs")
    event_log_path = run_dir / "run_events.jsonl"
    generation_manifest_path = run_dir / "generation_manifest.jsonl"
    status_path = run_dir / "status.json"
    fingerprint = plan_sha256(tasks, run_metadata)

    if resume:
        if not run_dir.is_dir() or not event_log_path.is_file() or not status_path.is_file():
            raise FileNotFoundError(f"Cannot resume missing generation run: {run_dir}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        resumable_states = {
            "generation_running",
            "generation_failed",
            "generation_complete_evaluation_pending",
        }
        if status.get("state") not in resumable_states:
            raise RuntimeError(
                f"Generation run state is not resumable: {status.get('state')!r}"
            )
        events = read_jsonl(event_log_path)
        starts = [event for event in events if event.get("event") == "run_started"]
        if len(starts) != 1:
            raise RuntimeError("Resume requires exactly one run_started provenance event")
        observed = starts[0].get("plan_sha256")
        if observed != fingerprint:
            raise RuntimeError(
                f"Resume plan SHA256 mismatch: {observed!r} != {fingerprint!r}"
            )
        append_jsonl(
            event_log_path,
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "event": "run_resumed",
                "timestamp_utc": _utc_now(),
                "plan_sha256": fingerprint,
                "environment": environment or {},
            },
        )
        events = read_jsonl(event_log_path)
        write_json(
            status_path,
            {
                "state": "generation_running",
                "plan_sha256": fingerprint,
                "resumed_at_utc": _utc_now(),
            },
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        repository = Path(__file__).resolve().parents[2]
        resolved_config = run_metadata.get("resolved_config")
        resolved_asset_lock = run_metadata.get("asset_lock")
        formal_artifact_snapshot = isinstance(resolved_config, dict) and isinstance(
            resolved_asset_lock, dict
        )
        if isinstance(resolved_config, dict):
            write_yaml(run_dir / "config.resolved.yaml", resolved_config)
        if isinstance(resolved_asset_lock, dict):
            write_yaml(run_dir / "assets.lock.resolved.yaml", resolved_asset_lock)
        dataset_metadata = run_metadata.get("dataset_manifest")
        if isinstance(dataset_metadata, dict) and isinstance(
            dataset_metadata.get("input_subset"), list
        ):
            pd.DataFrame(dataset_metadata["input_subset"]).to_csv(
                run_dir / "dataset_manifest.csv", index=False
            )
        if formal_artifact_snapshot:
            source_manifest = snapshot_research_sources(
                repository, run_dir / "source_snapshot"
            )
            write_json(run_dir / "source_snapshot_manifest.json", source_manifest)
            write_json(run_dir / "environment.json", environment or {})
            (run_dir / "environment.lock.txt").write_text(
                python_package_lock(), encoding="utf-8"
            )
        append_jsonl(
            event_log_path,
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "event": "run_started",
                "timestamp_utc": _utc_now(),
                "plan_sha256": fingerprint,
                "run_metadata": run_metadata,
                "environment": environment or {},
                "tasks": [task.provenance() for task in tasks],
            },
        )
        events = read_jsonl(event_log_path)
        write_json(
            status_path,
            {
                "state": "generation_running",
                "plan_sha256": fingerprint,
                "started_at_utc": _utc_now(),
            },
        )

    manifest_rows = read_jsonl(generation_manifest_path)
    completed = _completed_tasks(manifest_rows, tasks_by_id, run_dir)
    pending = [task for task in tasks if task.task_id not in completed]
    if not pending:
        summary = {
            "planned": len(tasks),
            "generated": 0,
            "skipped_completed": len(completed),
            "plan_sha256": fingerprint,
        }
        append_jsonl(
            event_log_path,
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "event": "run_completed",
                "timestamp_utc": _utc_now(),
                **summary,
            },
        )
        write_json(
            status_path,
            {
                "state": "generation_complete_evaluation_pending",
                "plan_sha256": fingerprint,
                **summary,
                "completed_at_utc": _utc_now(),
            },
        )
        return summary

    load_stats = _cuda_stats_start()
    load_start = time.perf_counter()
    try:
        backend = backend_factory()
        load_elapsed = time.perf_counter() - load_start
        load_stats = _cuda_stats_finish(load_stats)
        append_jsonl(
            event_log_path,
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "event": "backend_ready",
                "timestamp_utc": _utc_now(),
                "elapsed_seconds": load_elapsed,
                "cuda_memory": load_stats,
                "backend": backend.provenance(),
            },
        )
    except Exception as error:
        append_jsonl(
            event_log_path,
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "event": "run_failed",
                "timestamp_utc": _utc_now(),
                "stage": "backend_initialization",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        write_json(
            status_path,
            {
                "state": "generation_failed",
                "plan_sha256": fingerprint,
                "stage": "backend_initialization",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": _utc_now(),
            },
        )
        raise

    generated = 0
    for task in pending:
        events = read_jsonl(event_log_path)
        attempt = _attempt_number(events, task.task_id)
        latents = make_fixed_latents(
            task.effective_seed,
            task.height,
            task.width,
            device=latent_device,
        )
        latent_hash = tensor_sha256(latents)
        append_jsonl(
            event_log_path,
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "event": "task_started",
                "timestamp_utc": _utc_now(),
                "task_id": task.task_id,
                "attempt": attempt,
                "latent_sha256": latent_hash,
                "task": task.provenance(),
            },
        )
        stats = _cuda_stats_start()
        started = time.perf_counter()
        try:
            result = backend.generate(task, latents.clone())
            if not isinstance(result, BackendGeneration):
                raise TypeError("Generation backend must return BackendGeneration")
            image = result.image.convert("RGB")
            if image.size != (task.width, task.height):
                raise RuntimeError(
                    f"Generated image size mismatch for {task.task_id}: "
                    f"{image.size} != {(task.width, task.height)}"
                )
            relative = _output_relative_path(task)
            output_path = run_dir / relative
            _save_image_atomic(image, output_path)
            elapsed = time.perf_counter() - started
            stats = _cuda_stats_finish(stats)
            output_sha256 = sha256_file(output_path)
            manifest_row = _generation_manifest_row(
                task,
                attempt=attempt,
                status="complete",
                elapsed_seconds=elapsed,
                cuda_memory=stats,
                latent_sha256=latent_hash,
                output_relative_path=relative.as_posix(),
                output_sha256=output_sha256,
                backend_metadata=result.metadata,
            )
            upsert_generation_manifest(generation_manifest_path, manifest_row)
            append_jsonl(
                event_log_path,
                {
                    "schema_version": GENERATION_SCHEMA_VERSION,
                    "event": "task_completed",
                    "timestamp_utc": _utc_now(),
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "pair_id": task.pair_id,
                    "condition": task.condition,
                    "base_seed": task.base_seed,
                    "effective_seed": task.effective_seed,
                    "latent_sha256": latent_hash,
                    "elapsed_seconds": elapsed,
                    "cuda_memory": stats,
                    "backend_metadata": result.metadata,
                    "output": {
                        "relative_path": relative.as_posix(),
                        "sha256": output_sha256,
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                    },
                    "task": task.provenance(),
                },
            )
            generated += 1
        except Exception as error:
            elapsed = time.perf_counter() - started
            try:
                stats = _cuda_stats_finish(stats)
            except Exception:
                stats = {**stats, "finish_error": traceback.format_exc()}
            failure = {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "event": "task_failed",
                "timestamp_utc": _utc_now(),
                "task_id": task.task_id,
                "attempt": attempt,
                "latent_sha256": latent_hash,
                "elapsed_seconds": elapsed,
                "cuda_memory": stats,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "task": task.provenance(),
            }
            upsert_generation_manifest(
                generation_manifest_path,
                _generation_manifest_row(
                    task,
                    attempt=attempt,
                    status="failed",
                    elapsed_seconds=elapsed,
                    cuda_memory=stats,
                    latent_sha256=latent_hash,
                    output_relative_path=None,
                    output_sha256=None,
                    backend_metadata=None,
                    error=error,
                ),
            )
            append_jsonl(event_log_path, failure)
            append_jsonl(
                event_log_path,
                {
                    "schema_version": GENERATION_SCHEMA_VERSION,
                    "event": "run_failed",
                    "timestamp_utc": _utc_now(),
                    "stage": "task",
                    "task_id": task.task_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            write_json(
                status_path,
                {
                    "state": "generation_failed",
                    "plan_sha256": fingerprint,
                    "stage": "task",
                    "task_id": task.task_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at_utc": _utc_now(),
                },
            )
            raise

    summary = {
        "planned": len(tasks),
        "generated": generated,
        "skipped_completed": len(completed),
        "plan_sha256": fingerprint,
    }
    append_jsonl(
        event_log_path,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "event": "run_completed",
            "timestamp_utc": _utc_now(),
            **summary,
        },
    )
    write_json(
        status_path,
        {
            "state": "generation_complete_evaluation_pending",
            "plan_sha256": fingerprint,
            **summary,
            "completed_at_utc": _utc_now(),
        },
    )
    return summary


def directory_provenance(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing local model directory: {path}")
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or ".cache" in item.parts:
            continue
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    if not files:
        raise RuntimeError(f"Local model directory contains no files: {path}")
    return {
        "path": path.resolve().as_posix(),
        "file_count": len(files),
        "files": files,
        "inventory_sha256": sha256_bytes(_canonical_json(files)),
    }


def generation_environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "accelerate",
        "diffusers",
        "huggingface-hub",
        "numpy",
        "onnxruntime-gpu",
        "opencv-python",
        "peft",
        "pillow",
        "safetensors",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    gpu: dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(properties.total_memory),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return {
        "timestamp_utc": _utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "gpu": gpu,
    }


class PhotoMakerV2Backend:
    """Local-only PhotoMaker V2 backend with explicit global/patch routing."""

    def __init__(
        self,
        tasks: Sequence[GenerationTask],
        *,
        photomaker_source: Path,
        insightface_source: Path,
        insightface_root: Path,
        base_model: Path,
        adapter_path: Path,
        providers: Sequence[str],
        dtype: str = "auto",
        cpu_offload: bool = True,
        fuse_lora: bool = True,
    ) -> None:
        self._validate_assets(
            photomaker_source,
            insightface_source,
            insightface_root,
            base_model,
            adapter_path,
        )
        if not providers:
            raise ValueError("At least one ONNX Runtime provider is required")
        if not torch.cuda.is_available():
            raise RuntimeError("PhotoMaker pilot requires CUDA")

        sys.path.insert(0, str(insightface_source.resolve()))
        sys.path.insert(0, str(photomaker_source.resolve()))
        from diffusers import EulerDiscreteScheduler
        from photomaker import (
            FaceAnalysis2,
            PhotoMakerStableDiffusionXLPipeline,
            analyze_faces,
        )

        self._providers = tuple(str(provider) for provider in providers)
        self._image_cache: dict[str, Image.Image] = {}
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._face_metadata: dict[str, dict[str, Any]] = {}

        references = unique_references(tasks)
        analyzer = FaceAnalysis2(
            name="buffalo_l",
            root=str(insightface_root.resolve()),
            providers=list(self._providers),
            allowed_modules=["detection", "recognition"],
        )
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
        try:
            for reference in references:
                image = Image.open(reference.source_path).convert("RGB")
                self._image_cache[reference.sample_id] = image.copy()
                bgr = np.asarray(image, dtype=np.uint8)[:, :, ::-1]
                faces = analyze_faces(analyzer, bgr)
                if len(faces) != 1:
                    raise RuntimeError(
                        f"Expected exactly one detected face for {reference.sample_id}; "
                        f"observed {len(faces)}"
                    )
                face = faces[0]
                embedding = np.asarray(face["embedding"], dtype=np.float32)
                if embedding.shape != (512,) or not np.isfinite(embedding).all():
                    raise RuntimeError(
                        f"Invalid 512-D face embedding for {reference.sample_id}: {embedding.shape}"
                    )
                self._embedding_cache[reference.sample_id] = embedding.copy()
                self._face_metadata[reference.sample_id] = {
                    "sample_id": reference.sample_id,
                    "detected_face_count": 1,
                    "bbox": np.asarray(face["bbox"], dtype=float).tolist(),
                    "detection_score": float(face["det_score"]),
                    "embedding_dim": int(embedding.size),
                    "embedding_l2_norm": float(np.linalg.norm(embedding)),
                }
        finally:
            del analyzer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if dtype == "auto":
            self._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif dtype == "bfloat16":
            self._dtype = torch.bfloat16
        elif dtype == "float16":
            self._dtype = torch.float16
        else:
            raise ValueError("dtype must be one of: auto, bfloat16, float16")

        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            str(base_model.resolve()),
            torch_dtype=self._dtype,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        self.pipe.load_photomaker_adapter(
            str(adapter_path.resolve().parent),
            subfolder="",
            weight_name=adapter_path.name,
            trigger_word="img",
            local_files_only=True,
        )
        self._fuse_lora = bool(fuse_lora)
        if self._fuse_lora:
            self.pipe.fuse_lora()
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_vae_tiling()
        self._cpu_offload = bool(cpu_offload)
        if self._cpu_offload:
            self.pipe.enable_model_cpu_offload()
            # PhotoMaker registers id_encoder after the base SDXL modules and it is not
            # part of Diffusers' SDXL model_cpu_offload_seq. Keep this component on the
            # execution device explicitly; otherwise CUDA pixels meet CPU CLIP weights.
            self.pipe.id_encoder.to(device="cuda", dtype=self._dtype)
        else:
            self.pipe.to("cuda")

    @staticmethod
    def _validate_assets(
        photomaker_source: Path,
        insightface_source: Path,
        insightface_root: Path,
        base_model: Path,
        adapter_path: Path,
    ) -> None:
        required = [
            photomaker_source / "photomaker" / "pipeline.py",
            insightface_source / "insightface" / "app" / "face_analysis.py",
            insightface_root / "models" / "buffalo_l" / "det_10g.onnx",
            insightface_root / "models" / "buffalo_l" / "w600k_r50.onnx",
            base_model / "model_index.json",
            base_model / "unet" / "diffusion_pytorch_model.fp16.safetensors",
            base_model / "vae" / "diffusion_pytorch_model.fp16.safetensors",
            adapter_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing local PhotoMaker assets: {missing}")

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "PhotoMakerV2Backend",
            "dtype": str(self._dtype),
            "cpu_offload": self._cpu_offload,
            "id_encoder_device": str(next(self.pipe.id_encoder.parameters()).device),
            "photomaker_lora_fused": self._fuse_lora,
            "onnxruntime_providers": list(self._providers),
            "face_embeddings": [
                self._face_metadata[key] for key in sorted(self._face_metadata)
            ],
        }

    def _patch_images(self, task: GenerationTask) -> list[Image.Image]:
        return [self._image_cache[record.sample_id].copy() for record in task.patch_references]

    def _global_embeddings(self, task: GenerationTask) -> torch.Tensor:
        values = [self._embedding_cache[record.sample_id] for record in task.global_references]
        return torch.from_numpy(np.stack(values, axis=0))

    def generate(self, task: GenerationTask, latents: torch.Tensor) -> BackendGeneration:
        if int(self.pipe.unet.config.in_channels) != int(latents.shape[1]):
            raise RuntimeError("Latent channel count does not match the local UNet")
        common = {
            "negative_prompt": task.negative_prompt,
            "height": task.height,
            "width": task.width,
            "num_inference_steps": task.num_inference_steps,
            "guidance_scale": task.guidance_scale,
            "num_images_per_prompt": 1,
            "latents": latents.to(dtype=self._dtype),
            "generator": torch.Generator(device=latents.device).manual_seed(task.effective_seed),
            "output_type": "pil",
        }
        if len(task.patch_references) != len(task.global_references):
            raise RuntimeError("Patch/global reference counts must match")
        if not task.patch_references:
            raise RuntimeError("PhotoMaker requires at least one patch/global reference pair")
        output = self.pipe(
            prompt=task.prompt,
            input_id_images=self._patch_images(task),
            id_embeds=self._global_embeddings(task),
            start_merge_step=task.start_merge_step,
            **common,
        )
        metadata = {
            "conditioning": "photomaker_v2_text_only_schedule"
            if task.text_only
            else "photomaker_v2",
            "identity_adapter_loaded": True,
            "identity_fused_denoising_iterations": 0 if task.text_only else None,
            "patch_sample_ids": [item.sample_id for item in task.patch_references],
            "global_sample_ids": [item.sample_id for item in task.global_references],
        }
        if len(output.images) != 1:
            raise RuntimeError(f"Expected one generated image, observed {len(output.images)}")
        return BackendGeneration(image=output.images[0], metadata=metadata)


def local_model_metadata(
    *,
    photomaker_source: Path,
    insightface_source: Path,
    insightface_root: Path,
    base_model: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Missing PhotoMaker adapter: {adapter_path}")
    buffalo = insightface_root / "models" / "buffalo_l"
    return {
        "photomaker_source": {
            "path": photomaker_source.resolve().as_posix(),
            **git_revision(photomaker_source.resolve()),
        },
        "insightface_source": {
            "path": insightface_source.resolve().as_posix(),
            **git_revision(insightface_source.resolve().parent),
        },
        "base_model": directory_provenance(base_model),
        "photomaker_adapter": {
            "path": adapter_path.resolve().as_posix(),
            "size_bytes": adapter_path.stat().st_size,
            "sha256": sha256_file(adapter_path),
        },
        "insightface_models": directory_provenance(buffalo),
    }
