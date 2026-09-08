from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_auto_research_under_test", ROOT / "scripts/finalize_auto_research.py"
)
closure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closure)


def test_closure_rejects_running_tasks_active_gpu_and_missing_tasks() -> None:
    config = {"tasks": ["audit", "summary"]}
    with pytest.raises(RuntimeError, match="active"):
        closure.require_terminal_tasks(
            config, {"audit": {"state": "complete"}}, {"active_gpu": {"name": "work"}}, {}
        )
    with pytest.raises(RuntimeError, match="running"):
        closure.require_terminal_tasks(config, {"audit": {"state": "running"}}, {}, {})
    with pytest.raises(RuntimeError, match="missing task"):
        closure.require_terminal_tasks(config, {}, {}, {})


def test_closure_preserves_failed_tasks_with_traceable_reasons() -> None:
    config = {"tasks": ["audit"]}
    closure.require_terminal_tasks(
        config, {"audit": {"state": "failed", "reason": "recorded exception"}}, {}, {}
    )
    with pytest.raises(RuntimeError, match="traceable reason"):
        closure.require_terminal_tasks(config, {"audit": {"state": "failed"}}, {}, {})


def test_manifest_excludes_itself_and_locks_but_includes_environment_locks(tmp_path: Path) -> None:
    target = tmp_path / "provenance/final_test"
    target.mkdir(parents=True)
    (tmp_path / "resource_ledger.lock").write_bytes(b"mutable")
    (tmp_path / "environment.lock.txt").write_bytes(b"torch==locked")
    (tmp_path / "artifact.json").write_bytes(b"evidence")
    before = closure.inventory_run(tmp_path, target)
    (target / "manifest.json").write_text(json.dumps(before))
    assert before == closure.inventory_run(tmp_path, target)
    assert set(before) == {"environment.lock.txt", "artifact.json"}


def test_source_integrity_rejects_added_removed_and_changed_bytes() -> None:
    before = {"image": {"sha256": "a", "bytes": 1}}
    assert closure.unchanged_source(before, before)["unchanged"]
    for after in (
        {},
        {"image": {"sha256": "b", "bytes": 1}},
        {**before, "new": {"sha256": "c", "bytes": 1}},
    ):
        with pytest.raises(RuntimeError, match="source changed"):
            closure.unchanged_source(before, after)


def test_memory_peak_uses_cell_max_and_wall_time_is_not_double_counted(tmp_path: Path) -> None:
    event_file = tmp_path / "run_events.jsonl"
    event_file.write_text(
        json.dumps(
            {
                "event": "cell_finished",
                "elapsed_seconds": 2,
                "cuda_memory": {"peak_allocated_bytes": 500, "peak_reserved_bytes": 900},
            }
        )
        + "\n"
    )
    ledger = {
        "download_bytes": 10,
        "downloads": [{"bytes": 10}],
        "gpu_seconds": 3,
        "gpu_tasks": [{"name": "generation", "elapsed_seconds": 3, "peak_allocated_bytes": 100}],
    }
    supplement = {"download_bytes": 5, "downloads": [{"bytes": 5}], "gpu_seconds": 0}
    config = {"budget": {"download_bytes": 20, "gpu_seconds": 10}}
    result = closure.summarize_resources(config, ledger, supplement, [event_file], tmp_path)
    assert result["total_download_bytes"] == 15
    assert result["gpu_worker_wall_seconds"] == 3
    assert result["max_pytorch_peak_allocated_bytes"] == 500
    assert result["max_pytorch_peak_reserved_bytes"] == 900
    assert result["generation_inner_timings"][0]["finished_cell_wall_seconds"] == 2


def test_budget_receipt_mismatch_and_overrun_are_rejected(tmp_path: Path) -> None:
    config = {"budget": {"download_bytes": 20, "gpu_seconds": 10}}
    ledger = {"download_bytes": 10, "downloads": [{"bytes": 9}], "gpu_seconds": 0, "gpu_tasks": []}
    with pytest.raises(RuntimeError, match="receipts do not reconcile"):
        closure.summarize_resources(config, ledger, {}, [], tmp_path)
    ledger["download_bytes"] = 21
    with pytest.raises(RuntimeError, match="exceeds"):
        closure.summarize_resources(config, ledger, {}, [], tmp_path)
