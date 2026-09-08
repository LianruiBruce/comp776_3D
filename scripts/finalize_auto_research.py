"""CPU-only evidence closure after all automatic-research tasks are terminal.

Writes exclusively into a new timestamped provenance/final_* directory. It never
updates task statuses, budgets, completed attempts, or historical experiments.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from id_layers.auto_diagnostics import (  # noqa: E402
    build_seed_pairs,
    load_generation_manifest,
    resolve_inside,
    validate_generation_records,
)
from id_layers.auto_research import digest, dump, outside_source, tree_hashes  # noqa: E402

TERMINAL_STATES = {
    "complete",
    "complete_with_limitations",
    "complete_with_failures",
    "failed",
    "budget_stopped",
    "blocked_external",
    "blocked",
}
COMPLETE_STATES = {"complete", "complete_with_limitations", "complete_with_failures"}
GENERATION_EXPECTATIONS = {"photomaker": (48, 24), "pulid_smoke": (16, 8), "pulid_full": (48, 24)}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_terminal_tasks(config: dict, tasks: dict, ledger: dict, supplement: dict) -> None:
    if ledger.get("active_gpu") or supplement.get("active_gpu"):
        raise RuntimeError("Cannot close evidence while a GPU worker/reservation is active")
    expected = set(config["tasks"]) - {"summary"}
    if missing := expected - tasks.keys():
        raise RuntimeError(f"Cannot close evidence with missing task states: {sorted(missing)}")
    bad = {
        name: state.get("state")
        for name, state in tasks.items()
        if state.get("state") not in TERMINAL_STATES
    }
    if bad:
        raise RuntimeError(f"Cannot close evidence with nonterminal/running task states: {bad}")
    for name, state in tasks.items():
        if state["state"] not in COMPLETE_STATES and not (
            state.get("reason") or state.get("error")
        ):
            raise RuntimeError(f"Terminal stopped/failed task lacks a traceable reason: {name}")
    for name, data in (("main", ledger), ("supplemental", supplement)):
        if any(row.get("status") == "running" for row in data.get("downloads", [])):
            raise RuntimeError(f"Cannot close evidence while {name} download is running")


def unchanged_source(before: dict, after: dict) -> dict:
    changed = sorted(
        set(before) ^ set(after)
        | {key for key in set(before) & set(after) if before[key] != after[key]}
    )
    if changed:
        raise RuntimeError(f"Immutable original source changed: {changed[:10]}")
    return {
        "unchanged": True,
        "file_count": len(before),
        "bytes": sum(row["bytes"] for row in before.values()),
        "changed_files": [],
    }


def inventory_run(run: Path, closure: Path) -> dict[str, dict]:
    run, closure = run.resolve(), closure.resolve()
    if run not in closure.parents:
        raise ValueError("Closure must be a new child directory of the automatic run")
    records = {}
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.is_relative_to(closure) or path.suffix.lower() == ".lock":
            continue
        records[path.relative_to(run).as_posix()] = {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    return records


def numeric_peak_records(value: Any, source: str, pointer: str = "") -> list[dict]:
    """Harvest maxima from nested cell/load events; never add memory peaks."""
    records = []
    if isinstance(value, dict):
        names = {
            "peak_allocated_bytes": "allocated",
            "peak_cuda_bytes": "allocated",
            "peak_reserved_bytes": "reserved",
            "peak_cuda_reserved_bytes": "reserved",
        }
        for key, item in value.items():
            location = pointer + "/" + str(key)
            if (
                key in names
                and isinstance(item, (int, float))
                and math.isfinite(item)
                and item >= 0
            ):
                records.append(
                    {"kind": names[key], "bytes": int(item), "source": source, "pointer": location}
                )
            else:
                records.extend(numeric_peak_records(item, source, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            records.extend(numeric_peak_records(item, source, pointer + "/" + str(index)))
    return records


def summarize_resources(
    config: dict, ledger: dict, supplement: dict, generation_files: list[Path], run: Path
) -> dict:
    downloaded = int(ledger["download_bytes"]) + int(supplement.get("download_bytes", 0))
    gpu_seconds = float(ledger["gpu_seconds"]) + float(supplement.get("gpu_seconds", 0))
    if (
        downloaded > config["budget"]["download_bytes"]
        or gpu_seconds > config["budget"]["gpu_seconds"]
    ):
        raise RuntimeError("Final combined resource usage exceeds the authorized budget")
    workers = list(ledger.get("gpu_tasks", [])) + list(supplement.get("gpu_tasks", []))
    worker_sum = sum(float(row["elapsed_seconds"]) for row in workers)
    if not math.isclose(worker_sum, gpu_seconds, abs_tol=1e-5, rel_tol=1e-8):
        raise RuntimeError(
            f"GPU worker accounting does not reconcile: records={worker_sum}, total={gpu_seconds}"
        )
    download_sum = sum(
        int(row.get("bytes", 0))
        for data in (ledger, supplement)
        for row in data.get("downloads", [])
    )
    if download_sum != downloaded:
        raise RuntimeError(
            f"Download byte receipts do not reconcile: receipts={download_sum}, total={downloaded}"
        )
    by_task = defaultdict(
        lambda: {"worker_count": 0, "wall_seconds": 0.0, "interrupted_or_timeout_count": 0}
    )
    for row in workers:
        entry = by_task[row["name"]]
        entry["worker_count"] += 1
        entry["wall_seconds"] += float(row["elapsed_seconds"])
        entry["interrupted_or_timeout_count"] += int(
            row.get("status") in {"interrupted_engineering", "hard_timeout"}
        )
    peaks = numeric_peak_records(ledger, "resource_ledger.json") + numeric_peak_records(
        supplement, "pulid_download_ledger.json"
    )
    generation_inner = []
    for path in generation_files:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        relative = path.relative_to(run).as_posix()
        peaks.extend(numeric_peak_records(rows, relative))
        if path.name == "run_events.jsonl":
            loads = [row for row in rows if row.get("event") in {"backend_ready", "backend_failed"}]
            finished = [row for row in rows if row.get("event") == "cell_finished"]
            generation_inner.append(
                {
                    "event_log": relative,
                    "backend_load_attempt_count": len(loads),
                    "backend_load_wall_seconds": sum(
                        float(row["elapsed_seconds"]) for row in loads
                    ),
                    "cell_finished_count": len(finished),
                    "finished_cell_wall_seconds": sum(
                        float(row["elapsed_seconds"]) for row in finished
                    ),
                    "cell_started_count": sum(row.get("event") == "cell_started" for row in rows),
                    "accounting": (
                        "nested within GPU-worker wall time; NOT added again to global budget"
                    ),
                }
            )
    maxima = {}
    for kind in ("allocated", "reserved"):
        relevant = [row for row in peaks if row["kind"] == kind]
        maxima["max_pytorch_peak_" + kind + "_bytes"] = max(
            (row["bytes"] for row in relevant), default=None
        )
        maxima["max_pytorch_peak_" + kind + "_sources"] = [
            row for row in relevant if row["bytes"] == maxima["max_pytorch_peak_" + kind + "_bytes"]
        ]
    return {
        "total_download_bytes": downloaded,
        "download_receipt_bytes": download_sum,
        "main_download_bytes": int(ledger["download_bytes"]),
        "pulid_download_bytes": int(supplement.get("download_bytes", 0)),
        "gpu_worker_wall_seconds": gpu_seconds,
        "gpu_worker_record_seconds": worker_sum,
        "download_limit_bytes": config["budget"]["download_bytes"],
        "gpu_limit_seconds": config["budget"]["gpu_seconds"],
        "within_budget": True,
        "gpu_workers_by_task": dict(sorted(by_task.items())),
        "all_gpu_worker_attempts": workers,
        "generation_inner_timings": generation_inner,
        **maxima,
        "memory_scope": (
            "PyTorch allocator only; maxima across worker/load/cell records compensate for "
            "per-cell resets; not total process/ORT/driver VRAM"
        ),
    }


def verify_snapshot(wrapper: Path, root: Path) -> dict:
    manifest_path = wrapper / "source_snapshot_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["file_count"] != len(manifest["files"]):
        raise ValueError("Wrapper source snapshot manifest count mismatch")
    for record in manifest["files"]:
        path = resolve_inside(wrapper / "source_snapshot", record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"Successful wrapper source snapshot changed: {path}")
    return {
        "wrapper": wrapper.relative_to(root).as_posix(),
        "source_snapshot": (wrapper / "source_snapshot").relative_to(root).as_posix(),
        "source_snapshot_manifest_sha256": digest(manifest_path),
        "source_file_count": manifest["file_count"],
        "environment_json_sha256": digest(wrapper / "environment.json"),
        "environment_lock_sha256": digest(wrapper / "environment.lock.txt"),
    }


def verify_generation_tasks(root: Path, run: Path, tasks: dict) -> tuple[list[dict], list[dict]]:
    selection = read_json(
        resolve_inside(root, tasks["audit"]["output"]) / "generation_selection.json"
    )
    checks, execution = [], []
    for name, (expected_count, expected_pairs) in GENERATION_EXPECTATIONS.items():
        state = tasks[name]
        wrapper = resolve_inside(root, state["output"])
        if not wrapper.is_relative_to(run):
            raise ValueError("Task output escaped automatic run")
        if state["state"] not in COMPLETE_STATES:
            checks.append(
                {
                    "task": name,
                    "status": "not_completed",
                    "terminal_state": state["state"],
                    "reason": state.get("reason") or state.get("error"),
                    "expected_cells": expected_count,
                    "expected_seed_pairs": expected_pairs,
                }
            )
            continue
        result = read_json(wrapper / "result.json")
        generation = resolve_inside(root, result["generation_path"])
        if not generation.is_relative_to(run):
            raise ValueError("Generation path escaped automatic run")
        rows = load_generation_manifest(generation / "generation_manifest.jsonl")
        expected_ids = (
            selection["selected_identities"][:4]
            if name == "pulid_smoke"
            else selection["selected_identities"]
        )
        check = validate_generation_records(
            rows,
            generation,
            expected_count=expected_count,
            expected_seeds=selection["base_seeds"],
            expected_identities=expected_ids,
            expected_conditions=["k1_full"],
            expected_prompts=list(selection["prompts"]),
        )
        pairs = build_seed_pairs(
            rows, expected_seeds=selection["base_seeds"], expected_pair_count=expected_pairs
        )
        if check["failed_cell_count"]:
            raise ValueError(f"Task {name} claims completed generation but has failed image cells")
        checks.append(
            {
                "task": name,
                "generation_path": generation.relative_to(root).as_posix(),
                "generation_manifest_sha256": digest(generation / "generation_manifest.jsonl"),
                "verified_seed_pairs": len(pairs),
                **check,
            }
        )
        execution.append(
            {
                "task": name,
                "generation_path": generation.relative_to(root).as_posix(),
                "successful_wrapper_provenance": verify_snapshot(wrapper, root),
                "initial_generation_snapshot_manifest_sha256": digest(
                    generation / "source_snapshot_manifest.json"
                ),
                "interpretation": (
                    "Initial generation-directory snapshot describes initial engineering state; "
                    "successful wrapper snapshot records code at the completed execution. "
                    "Interrupted attempts are retained; residency differences are "
                    "per-image backend metadata."
                ),
                "per_image_backend_metadata": [
                    {
                        "sample_id": row["sample_id"],
                        "status": row["status"],
                        "backend_metadata": row.get("backend_metadata"),
                    }
                    for row in rows
                ],
            }
        )
    return checks, execution


def snapshot_final_sources(root: Path, output: Path) -> dict:
    roots = [
        root / name
        for name in (
            "pyproject.toml",
            "configs",
            "requirements",
            "scripts",
            "src",
            "tests",
            "docs",
            "reports",
            "README.md",
            "AGENTS.md",
        )
    ]
    paths = []
    for path in roots:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts
            )
    records = []
    for path in sorted(paths):
        relative = path.relative_to(root)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records.append(
            {"path": relative.as_posix(), "sha256": digest(path), "bytes": path.stat().st_size}
        )
    result = {
        "file_count": len(records),
        "files": records,
        "scope": (
            "Final executable source plus curated documentation; successful worker snapshots "
            "remain separately authoritative for executed experiments"
        ),
    }
    dump(output / "source_snapshot_manifest.json", result)
    return result


def cpu_environment(root: Path, output: Path) -> dict:
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "onnxruntime-gpu",
        "diffusers",
        "numpy",
        "lpips",
        "mediapipe",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    locks = []
    for environment in (".venv-auto", ".venv-pulid"):
        executable = root / environment / "Scripts/python.exe"
        if not executable.is_file():
            continue
        package_lock = subprocess.check_output(
            [str(executable), "-m", "pip", "freeze", "--all"], text=True, stderr=subprocess.STDOUT
        )
        target = output / (environment.removeprefix(".") + ".environment.lock.txt")
        target.write_text(package_lock, encoding="utf-8")
        locks.append({"environment": environment, "path": target.name, "sha256": digest(target)})
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages_in_closure_interpreter": versions,
        "environment_locks": locks,
        "gpu_runtime_queried": False,
        "gpu_computation_performed": False,
    }
    dump(output / "environment.json", result)
    return result


def finalize(root: Path, config: dict) -> Path:
    run, source = (
        resolve_inside(root, config["run_dir"]),
        resolve_inside(root, config["source_run"]),
    )
    outside_source(run, source)
    tasks_path, ledger_path = run / "tasks.json", run / "resource_ledger.json"
    supplement_path = run / "pulid_download_ledger.json"
    tasks, ledger = read_json(tasks_path), read_json(ledger_path)
    supplement = (
        read_json(supplement_path)
        if supplement_path.exists()
        else {"download_bytes": 0, "gpu_seconds": 0, "downloads": [], "gpu_tasks": []}
    )
    require_terminal_tasks(config, tasks, ledger, supplement)
    source_integrity = unchanged_source(
        read_json(run / "source_hashes_before.json"), tree_hashes(source)
    )
    # No output directory is created until the terminal-state and original-source checks pass.
    output = run / "provenance" / ("final_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    try:
        before = inventory_run(run, output)
        generation_checks, execution = verify_generation_tasks(root, run, tasks)
        generation_files = sorted(
            set(run.rglob("generation_manifest.jsonl")) | set(run.rglob("run_events.jsonl"))
        )
        generation_files = [path for path in generation_files if not path.is_relative_to(output)]
        resources = summarize_resources(config, ledger, supplement, generation_files, run)
        resources["latest_wrapper_task_wall_seconds"] = {
            name: state.get("elapsed_seconds") for name, state in tasks.items()
        }
        source_snapshot = snapshot_final_sources(root, output)
        environment = cpu_environment(root, output)
        dump(output / "config.resolved.json", config)
        dump(output / "tasks.snapshot.json", tasks)
        dump(output / "resource_ledger.snapshot.json", ledger)
        dump(output / "pulid_download_ledger.snapshot.json", supplement)
        dump(output / "source_integrity.json", source_integrity)
        dump(output / "generation_verification.json", generation_checks)
        dump(output / "execution_provenance.json", execution)
        dump(output / "resources.json", resources)
        dump(
            output / "new_run_file_manifest.json",
            {
                "file_count": len(before),
                "files": before,
                "root": config["run_dir"],
                "excluded": [output.relative_to(run).as_posix() + "/**", "**/*.lock"],
                "interpretation": (
                    "Hashes all pre-existing automatic-run files, including interrupted attempts; "
                    "excludes this new closure and mutable lock files"
                ),
            },
        )
        # Detect accidental mutation or a worker starting while closure was being prepared.
        unchanged_source(before, inventory_run(run, output))
        require_terminal_tasks(
            config,
            read_json(tasks_path),
            read_json(ledger_path),
            read_json(supplement_path) if supplement_path.exists() else supplement,
        )
        summary = {
            "status": "complete",
            "scope": config["scope"],
            "gpu_computation_performed": False,
            "source_unchanged": True,
            "source_file_count": source_integrity["file_count"],
            "automatic_run_file_count": len(before),
            "final_source_snapshot_file_count": source_snapshot["file_count"],
            "generation_checks": generation_checks,
            "resources": resources,
            "task_terminal_states": {name: record["state"] for name, record in tasks.items()},
            "environment_lock_count": len(environment["environment_locks"]),
            "preserved_failures": (
                "Earlier failed/interrupted engineering attempts remain in the hashed inventory "
                "and resource ledger; completion does not mean every attempt succeeded"
            ),
        }
        dump(output / "closure.json", summary)
        return output
    except Exception as error:
        dump(
            output / "failure.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/auto_research.yaml")
    args = parser.parse_args()
    output = finalize(ROOT, yaml.safe_load(args.config.read_text(encoding="utf-8")))
    print(json.dumps({"status": "complete", "closure": str(output)}, indent=2))


if __name__ == "__main__":
    main()
