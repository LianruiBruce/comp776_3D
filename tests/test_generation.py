from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from id_layers.generation import (
    BackendGeneration,
    build_generation_plan,
    make_fixed_latents,
    read_jsonl,
    run_generation_plan,
    tensor_sha256,
    verify_reference_files,
)
from id_layers.utils import sha256_file

FROZEN_REFERENCE_CONDITIONS = [
    "frontal_neutral",
    "left_profile",
    "right_profile",
    "frontal_scale",
]


def _manifest(tmp_path: Path) -> pd.DataFrame:
    rows = []
    slots = {
        "frontal_neutral": 11,
        "left_profile": 2,
        "right_profile": 9,
        "frontal_scale": 13,
    }
    for identity in ("fei_0001", "fei_0002"):
        for condition in FROZEN_REFERENCE_CONDITIONS:
            sample_id = f"{identity}_{slots[condition]:02d}"
            relative = Path("processed") / f"{sample_id}.jpg"
            source = tmp_path / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{sample_id}:{condition}".encode())
            rows.append(
                {
                    "sample_id": sample_id,
                    "identity_id": identity,
                    "acquisition_slot": slots[condition],
                    "condition": condition,
                    "protocol_role": "reference",
                    "source_relative_path": relative.as_posix(),
                    "source_sha256": sha256_file(source),
                    "split": "internal_eval",
                    "usable_for_model": True,
                }
            )
    return pd.DataFrame(rows)


def _plan(tmp_path: Path, *, prompts: dict[str, str] | None = None):
    return build_generation_plan(
        _manifest(tmp_path),
        tmp_path,
        [("fei_0001", "fei_0002"), ("fei_0002", "fei_0001")],
        prompts or {"p0": "portrait of a person img"},
        "bad image",
        [776],
        height=16,
        width=16,
        num_inference_steps=30,
        start_merge_step=10,
    )


def test_generation_plan_matches_frozen_conditions_and_channel_routes(
    tmp_path: Path,
) -> None:
    tasks = _plan(tmp_path)
    first = {task.condition: task for task in tasks[:6]}

    assert list(first) == [
        "k1_full",
        "k4_diverse_full",
        "k4_repeat_full",
        "global_target_patch_donor",
        "global_donor_patch_target",
        "text_only",
    ]
    assert len(first["k1_full"].patch_references) == 1
    assert [item.condition for item in first["k4_diverse_full"].patch_references] == (
        FROZEN_REFERENCE_CONDITIONS
    )
    repeated = first["k4_repeat_full"]
    assert len(repeated.patch_references) == 4
    assert len({item.sample_id for item in repeated.patch_references}) == 1

    target_global = first["global_target_patch_donor"]
    assert len(target_global.patch_references) == 1
    assert {item.identity_id for item in target_global.global_references} == {"fei_0001"}
    assert {item.identity_id for item in target_global.patch_references} == {"fei_0002"}
    donor_global = first["global_donor_patch_target"]
    assert {item.identity_id for item in donor_global.global_references} == {"fei_0002"}
    assert {item.identity_id for item in donor_global.patch_references} == {"fei_0001"}

    text_only = first["text_only"]
    assert text_only.text_only is True
    assert text_only.prompt.endswith("person img")
    assert len(text_only.patch_references) == len(text_only.global_references) == 1
    assert text_only.start_merge_step == text_only.num_inference_steps == 30

    first_identity = tasks[:6]
    second_identity = tasks[6:12]
    assert {task.base_seed for task in first_identity} == {776}
    assert {task.effective_seed for task in first_identity} == {776}
    assert {task.effective_seed for task in second_identity} == {100_776}
    assert len(verify_reference_files(tasks)) == 8


def test_effective_seed_reuses_actual_latent_across_conditions_and_prompts(
    tmp_path: Path,
) -> None:
    tasks = _plan(
        tmp_path,
        prompts={"p0": "portrait of a person img", "p1": "photo of a person img"},
    )
    hashes_by_identity: dict[str, set[str]] = {}
    for task in tasks:
        latent = make_fixed_latents(task.effective_seed, task.height, task.width)
        hashes_by_identity.setdefault(task.target_identity_id, set()).add(
            tensor_sha256(latent)
        )
    assert {identity: len(hashes) for identity, hashes in hashes_by_identity.items()} == {
        "fei_0001": 1,
        "fei_0002": 1,
    }
    assert hashes_by_identity["fei_0001"] != hashes_by_identity["fei_0002"]


def test_prompt_validation_accepts_punctuation_after_special_trigger(tmp_path: Path) -> None:
    tasks = _plan(tmp_path, prompts={"p0": "portrait of a person img, neutral expression"})
    assert len(tasks) == 12


def test_plan_rejects_wrong_split_and_reference_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["identity_id"] == "fei_0001", "split"] = "train"
    with pytest.raises(ValueError, match="internal_eval/reference"):
        build_generation_plan(
            manifest,
            tmp_path,
            [("fei_0001", "fei_0002")],
            {"p0": "portrait of a person img"},
            "bad",
            [776],
            height=16,
            width=16,
            num_inference_steps=30,
        )

    tasks = _plan(tmp_path)
    tasks[0].patch_references[0].source_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_reference_files(tasks)


class _FakeBackend:
    def __init__(self, *, fail_task_id: str | None = None) -> None:
        self.fail_task_id = fail_task_id
        self.calls: list[tuple[str, str]] = []

    def provenance(self) -> dict[str, object]:
        return {"backend": "fake"}

    def generate(self, task, latents) -> BackendGeneration:
        latent_hash = tensor_sha256(latents)
        self.calls.append((task.task_id, latent_hash))
        if task.task_id == self.fail_task_id:
            raise RuntimeError("synthetic failure")
        return BackendGeneration(
            image=Image.new("RGB", (task.width, task.height), color=(20, 30, 40)),
            metadata={"fake": True},
        )


def test_run_manifest_contract_and_completed_resume(tmp_path: Path) -> None:
    tasks = _plan(tmp_path)[:6]
    run_dir = tmp_path / "run"
    backend = _FakeBackend()
    summary = run_generation_plan(
        tasks,
        run_dir,
        lambda: backend,
        run_metadata={"protocol": "test"},
    )

    assert summary["generated"] == 6
    assert len({latent_hash for _, latent_hash in backend.calls}) == 1
    rows = read_jsonl(run_dir / "generation_manifest.jsonl")
    assert len(rows) == 6
    assert len({row["sample_id"] for row in rows}) == 6
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
        "reference_sample_ids",
        "global_embedding_identity_id",
        "patch_image_identity_id",
        "elapsed_seconds",
        "peak_cuda_bytes",
    }
    assert all(required <= row.keys() for row in rows)
    assert {row["status"] for row in rows} == {"complete"}
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "generation_complete_evaluation_pending"

    def should_not_load():
        raise AssertionError("A fully completed resume must not load the backend")

    resumed = run_generation_plan(
        tasks,
        run_dir,
        should_not_load,
        run_metadata={"protocol": "test"},
        resume=True,
    )
    assert resumed["generated"] == 0
    assert resumed["skipped_completed"] == 6


def test_fail_fast_then_resume_replaces_failed_manifest_row(tmp_path: Path) -> None:
    tasks = _plan(tmp_path)[:3]
    run_dir = tmp_path / "failed_run"
    first_backend = _FakeBackend(fail_task_id=tasks[1].task_id)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_generation_plan(
            tasks,
            run_dir,
            lambda: first_backend,
            run_metadata={"protocol": "test"},
        )
    assert [task_id for task_id, _ in first_backend.calls] == [
        tasks[0].task_id,
        tasks[1].task_id,
    ]
    failed_rows = read_jsonl(run_dir / "generation_manifest.jsonl")
    assert [row["status"] for row in failed_rows] == ["complete", "failed"]
    assert json.loads((run_dir / "status.json").read_text())["state"] == (
        "generation_failed"
    )

    second_backend = _FakeBackend()
    resumed = run_generation_plan(
        tasks,
        run_dir,
        lambda: second_backend,
        run_metadata={"protocol": "test"},
        resume=True,
    )
    assert resumed["skipped_completed"] == 1
    assert [task_id for task_id, _ in second_backend.calls] == [
        tasks[1].task_id,
        tasks[2].task_id,
    ]
    rows = read_jsonl(run_dir / "generation_manifest.jsonl")
    assert len(rows) == 3
    assert len({row["sample_id"] for row in rows}) == 3
    assert {row["status"] for row in rows} == {"complete"}
    retried = next(row for row in rows if row["sample_id"] == tasks[1].task_id)
    assert retried["attempt"] == 2
    events = read_jsonl(run_dir / "run_events.jsonl")
    assert any(event["event"] == "task_failed" for event in events)


def test_resume_rejects_output_hash_mismatch(tmp_path: Path) -> None:
    task = _plan(tmp_path)[:1]
    run_dir = tmp_path / "corrupt_run"
    run_generation_plan(
        task,
        run_dir,
        _FakeBackend,
        run_metadata={"protocol": "test"},
    )
    row = read_jsonl(run_dir / "generation_manifest.jsonl")[0]
    (run_dir / row["output_relative_path"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="output SHA256 mismatch"):
        run_generation_plan(
            task,
            run_dir,
            _FakeBackend,
            run_metadata={"protocol": "test"},
            resume=True,
        )
