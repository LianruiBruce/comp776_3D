from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from id_layers.auto_generation import (
    GALLERY_CONDITIONS,
    PROMPTS,
    prepare_generation_suite,
    run_generation_cells,
    select_dev_identities,
    smoke_tasks,
)
from id_layers.generation import BackendGeneration, read_jsonl, tensor_sha256
from id_layers.utils import sha256_file


@pytest.fixture(autouse=True)
def no_gpu_during_cpu_tests(monkeypatch):
    monkeypatch.setattr("id_layers.auto_generation.torch.cuda.is_available", lambda: False)


def manifest(tmp_path: Path, count: int = 15) -> pd.DataFrame:
    rows = []
    for identity_index in range(count):
        identity = f"fei_{identity_index:04d}"
        for slot, condition in enumerate(("frontal_neutral", *GALLERY_CONDITIONS)):
            source = tmp_path / "source" / f"{identity}_{slot}.png"
            source.parent.mkdir(exist_ok=True)
            Image.new("RGB", (16, 16), color=(identity_index, slot, 123)).save(source)
            rows.append(
                {
                    "sample_id": f"{identity}_{slot}",
                    "identity_id": identity,
                    "split": "dev",
                    "condition": condition,
                    "protocol_role": "reference" if slot == 0 else "query",
                    "source_relative_path": source.relative_to(tmp_path).as_posix(),
                    "source_sha256": sha256_file(source),
                    "acquisition_slot": slot,
                    "usable_for_model": True,
                }
            )
    return pd.DataFrame(rows)


def test_selection_and_cells_are_independent_of_manifest_order(tmp_path: Path) -> None:
    frame = manifest(tmp_path)
    first, selection, gallery = prepare_generation_suite(frame, tmp_path)
    second, shuffled_selection, shuffled_gallery = prepare_generation_suite(
        frame.sample(frac=1, random_state=92), tmp_path
    )
    assert [t.provenance() for t in first] == [t.provenance() for t in second]
    assert selection == shuffled_selection
    assert gallery == shuffled_gallery
    assert len(first) == 48
    assert len(gallery) == 36
    assert len(smoke_tasks(first, selection)) == 16
    assert (
        PROMPTS["neutral"].replace("neutral expression", "smiling expression") == PROMPTS["smile"]
    )
    reference_ids = {r.sample_id for t in first for r in t.patch_references}
    assert not reference_ids.intersection(g["sample_id"] for g in gallery)
    for identity in selection["selected_identities"]:
        subset = [t for t in first if t.target_identity_id == identity]
        assert len(subset) == 4
        for seed in (776, 1776):
            assert len({t.effective_seed for t in subset if t.base_seed == seed}) == 1


def test_selection_excludes_old_and_does_not_replace_failed_identity(tmp_path: Path) -> None:
    frame = manifest(tmp_path)
    selection = select_dev_identities(frame, count=12, excluded_identities=["fei_0000"])
    assert "fei_0000" not in selection["selected_identities"]
    selected = selection["selected_identities"][0]
    frame.loc[frame.identity_id == selected, "usable_for_model"] = False
    assert select_dev_identities(frame, count=12, excluded_identities=["fei_0000"]) == selection


def test_manifest_rejects_duplicate_roles_and_split_crossing(tmp_path: Path) -> None:
    frame = manifest(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        select_dev_identities(pd.concat([frame, frame.iloc[:1]]))
    crossing = frame.copy()
    crossing.loc[0, "split"] = "train"
    with pytest.raises(ValueError, match="crosses"):
        select_dev_identities(crossing)
    chosen = select_dev_identities(frame)["selected_identities"][0]
    frame.loc[
        (frame.identity_id == chosen) & (frame.condition == "frontal_smile"), "protocol_role"
    ] = "reference"
    with pytest.raises(ValueError, match="photo role"):
        prepare_generation_suite(frame, tmp_path)


class FakeBackend:
    def __init__(self, fail: str | None = None):
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def provenance(self):
        return {"backend": "synthetic_test_only"}

    def generate(self, task, latents):
        self.calls.append((task.task_id, tensor_sha256(latents)))
        if task.task_id == self.fail:
            raise RuntimeError("Fixed synthetic generation failure")
        image = Image.new("RGB", (16, 16), (50, 70, 90))
        image.putpixel((0, 0), (51, 71, 91))
        return BackendGeneration(image, {})


def small_tasks(tmp_path: Path):
    tasks, _, _ = prepare_generation_suite(manifest(tmp_path), tmp_path)
    return [replace(task, height=16, width=16) for task in tasks[:4]]


def run_kwargs(tmp_path: Path):
    return {
        "run_metadata": {"snapshot_sources": False, "test_only": True},
        "forbidden_source": tmp_path / "frozen",
        "latent_device": "cpu",
    }


def test_failures_retained_and_complete_resume_is_read_only(tmp_path: Path) -> None:
    tasks = small_tasks(tmp_path)
    backend = FakeBackend(fail=tasks[0].task_id)
    run_dir = tmp_path / "new_run"
    status = run_generation_cells(tasks, run_dir, lambda: backend, **run_kwargs(tmp_path))
    assert status["state"] == "complete_with_failures"
    assert status["complete"] == 3 and status["failed"] == 1
    before = {
        p.relative_to(run_dir).as_posix(): sha256_file(p) for p in run_dir.rglob("*") if p.is_file()
    }
    replacement = FakeBackend()
    resumed = run_generation_cells(
        list(reversed(tasks)), run_dir, lambda: replacement, resume=True, **run_kwargs(tmp_path)
    )
    assert resumed == status
    assert replacement.calls == []
    assert before == {
        p.relative_to(run_dir).as_posix(): sha256_file(p) for p in run_dir.rglob("*") if p.is_file()
    }
    rows = read_jsonl(run_dir / "generation_manifest.jsonl")
    assert all(row["attempt"] == 1 for row in rows)


def test_budget_stop_resumes_only_unattempted_cells(tmp_path: Path) -> None:
    tasks = small_tasks(tmp_path)
    backend = FakeBackend()
    permitted = iter((True, True, False))  # load; first cell; budget stops second
    run_dir = tmp_path / "new_run"
    status = run_generation_cells(
        tasks, run_dir, lambda: backend, can_start=lambda _: next(permitted), **run_kwargs(tmp_path)
    )
    assert status["state"] == "budget_stopped"
    assert status["complete"] == 1 and status["budget_stopped"] == 3
    finished = backend.calls[0][0]
    resumed_backend = FakeBackend()
    resumed = run_generation_cells(
        tasks, run_dir, lambda: resumed_backend, resume=True, **run_kwargs(tmp_path)
    )
    assert resumed["state"] == "complete"
    assert len(resumed_backend.calls) == 3
    assert finished not in {call[0] for call in resumed_backend.calls}


def test_refuses_source_output_and_wrong_pair_or_hash(tmp_path: Path) -> None:
    tasks = small_tasks(tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        run_generation_cells(
            tasks, tmp_path / "frozen" / "extra", FakeBackend, **run_kwargs(tmp_path)
        )
    run_dir = tmp_path / "new_run"
    run_generation_cells(tasks, run_dir, FakeBackend, **run_kwargs(tmp_path))
    manifest_path = run_dir / "generation_manifest.jsonl"
    original = manifest_path.read_text(encoding="utf-8")
    rows = read_jsonl(manifest_path)
    rows[0]["base_seed"] = 0
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pairing mismatch"):
        run_generation_cells(tasks, run_dir, FakeBackend, resume=True, **run_kwargs(tmp_path))
    manifest_path.write_text(original, encoding="utf-8")
    image_path = run_dir / read_jsonl(manifest_path)[0]["output_relative_path"]
    image_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_generation_cells(tasks, run_dir, FakeBackend, resume=True, **run_kwargs(tmp_path))


def test_reference_hash_failure_prevents_backend_initialization(tmp_path: Path) -> None:
    tasks = small_tasks(tmp_path)
    tasks[0].patch_references[0].source_path.write_bytes(b"tampered")
    calls = []

    def factory():
        calls.append(True)
        return FakeBackend()

    status = run_generation_cells(tasks, tmp_path / "new_run", factory, **run_kwargs(tmp_path))
    assert status["state"] == "blocked_external"
    assert status["blocked_external"] == 4
    assert calls == []
