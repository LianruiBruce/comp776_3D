from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from id_layers.auto_research import dump
from id_layers.process_supervision import (
    ProcessIdentity,
    SupervisedTimeout,
    TerminationUnverified,
    _live_process,
    run_supervised,
)


def test_real_cpu_child_tree_is_terminated_on_timeout(tmp_path):
    output = tmp_path / "owned_cpu_processes.txt"
    code = (
        "import os,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(os.getpid(),p.pid,flush=True); time.sleep(60)"
    )
    with output.open("w") as handle, pytest.raises(SupervisedTimeout) as captured:
        run_supervised([sys.executable, "-c", code], cwd=tmp_path, stdout=handle, timeout=2.0)
    report = captured.value.termination_report
    assert report["termination_verified"] is True
    observed_pids = {int(value) for value in output.read_text().split()}
    recorded = {item["pid"]: item["creation_time"] for item in report["owned_processes"]}
    assert len(observed_pids) == 2
    assert observed_pids <= recorded.keys()
    assert all(_live_process(ProcessIdentity(pid, recorded[pid])) is None for pid in observed_pids)
    assert captured.value.elapsed_seconds >= 2.0


def test_pid_reuse_is_not_treated_as_owned_process(monkeypatch):
    class DifferentProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 99.0

    monkeypatch.setattr(psutil, "Process", DifferentProcess)
    assert _live_process(ProcessIdentity(123, 10.0)) is None


def test_successful_cpu_command_and_nonzero_exit(tmp_path):
    with (tmp_path / "output.txt").open("w") as handle:
        result = run_supervised(
            [sys.executable, "-c", "print('ok')"], cwd=tmp_path, stdout=handle, timeout=10.0
        )
        assert result.returncode == 0
        with pytest.raises(subprocess.CalledProcessError):
            run_supervised(
                [sys.executable, "-c", "raise SystemExit(7)"],
                cwd=tmp_path,
                stdout=handle,
                timeout=10.0,
            )


def runner(tmp_path, monkeypatch, *, gpu_seconds=0.0):
    source = Path(__file__).resolve().parents[1] / "scripts/run_auto_research.py"
    spec = importlib.util.spec_from_file_location("auto_research_supervision_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "frozen").mkdir()
    run = tmp_path / "run"
    run.mkdir()
    ledger = {
        "download_limit_bytes": 30_000_000_000,
        "gpu_limit_seconds": 14400,
        "download_bytes": 0,
        "gpu_seconds": gpu_seconds,
        "downloads": [],
        "gpu_tasks": [],
    }
    dump(run / "resource_ledger.json", ledger)
    config = tmp_path / "config.yaml"
    config.write_text("run_dir: run\nsource_run: frozen\ntasks: [photomaker]\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", [str(source), "run", "--task", "photomaker", "--config", str(config)]
    )
    return module, run


def test_insufficient_budget_launches_no_worker(tmp_path, monkeypatch):
    module, run = runner(tmp_path, monkeypatch, gpu_seconds=14300)
    launched = []
    monkeypatch.setattr(module, "run_supervised", lambda *a, **k: launched.append(True))
    module.main()
    assert not launched
    assert json.loads((run / "tasks.json").read_text())["photomaker"]["worker_launched"] is False


def test_unverified_termination_keeps_reservation(tmp_path, monkeypatch):
    module, run = runner(tmp_path, monkeypatch)

    def unverified(*args, **kwargs):
        raise TerminationUnverified(
            "synthetic access failure",
            elapsed_seconds=1802,
            report={"termination_verified": False, "remaining_pids": [123]},
        )

    monkeypatch.setattr(module, "run_supervised", unverified)
    with pytest.raises(module.BudgetStop, match="reservation retained"):
        module.main()
    ledger = json.loads((run / "resource_ledger.json").read_text())
    assert ledger["active_gpu"]["termination_unverified"] is True
    assert ledger["active_gpu"]["reservation_seconds"] >= 1802
    assert ledger["gpu_seconds"] == 0


def test_verified_timeout_does_not_double_charge_completed_context(tmp_path, monkeypatch):
    module, run = runner(tmp_path, monkeypatch, gpu_seconds=100)

    def timeout(command, **kwargs):
        ledger = json.loads((run / "resource_ledger.json").read_text())
        ledger["gpu_seconds"] += 1700
        dump(run / "resource_ledger.json", ledger)
        raise SupervisedTimeout(
            command, 1800, elapsed_seconds=1803, termination_report={"termination_verified": True}
        )

    monkeypatch.setattr(module, "run_supervised", timeout)
    with pytest.raises(module.BudgetStop, match="tree terminated"):
        module.main()
    ledger = json.loads((run / "resource_ledger.json").read_text())
    assert ledger["gpu_seconds"] == 1903
    assert ledger["gpu_tasks"][-1]["elapsed_seconds"] == 103
    assert "active_gpu" not in ledger
