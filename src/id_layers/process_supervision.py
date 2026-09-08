"""Supervise owned subprocess trees, including Windows venv launcher children."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    creation_time: float


class TerminationUnverified(RuntimeError):
    """The caller must retain its resource reservation and forbid another worker."""

    def __init__(self, message: str, *, elapsed_seconds: float, report: dict[str, Any]):
        super().__init__(message)
        self.elapsed_seconds = elapsed_seconds
        self.report = report


class SupervisedTimeout(subprocess.TimeoutExpired):
    def __init__(
        self, command, timeout: float, *, elapsed_seconds: float, termination_report: dict[str, Any]
    ):
        super().__init__(command, timeout)
        self.elapsed_seconds = elapsed_seconds
        self.termination_report = termination_report


def _identity(process: psutil.Process) -> ProcessIdentity:
    return ProcessIdentity(process.pid, process.create_time())


def _live_process(identity: ProcessIdentity) -> psutil.Process | None:
    """Never signal a recycled PID that belongs to a different process."""
    try:
        process = psutil.Process(identity.pid)
        if process.create_time() != identity.creation_time:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except psutil.NoSuchProcess:
        return None


def _discover(known: dict[int, ProcessIdentity]) -> None:
    # Retaining descendants while the launcher is alive lets us clean up known
    # orphans even if the launcher exits before the timeout handler runs.
    for identity in list(known.values()):
        parent = _live_process(identity)
        if parent is None:
            continue
        try:
            for child in parent.children(recursive=True):
                known[child.pid] = _identity(child)
        except psutil.NoSuchProcess:
            continue


def terminate_owned_tree(
    known: dict[int, ProcessIdentity],
    *,
    verification_seconds: float = 10.0,
) -> dict[str, Any]:
    """Suspend the owned tree, kill descendants first, and verify original PIDs died.

    Suspending first closes the ordinary race in which a worker spawns another
    child between enumeration and termination. Access errors are explicit; they
    never justify releasing a GPU reservation.
    """
    report: dict[str, Any] = {"termination_verified": False, "suspended": [], "signalled": []}
    suspended: set[ProcessIdentity] = set()
    try:
        for _ in range(16):
            _discover(known)
            before = set(known.values())
            for identity in sorted(before, key=lambda item: (item.creation_time, item.pid)):
                process = _live_process(identity)
                if process is not None and identity not in suspended:
                    try:
                        process.suspend()
                        suspended.add(identity)
                        report["suspended"].append(identity.pid)
                    except psutil.NoSuchProcess:
                        pass
            _discover(known)
            if set(known.values()) == before:
                break
        else:
            raise RuntimeError("Owned descendant set did not stabilize while suspending")
        # A descendant cannot be older than its parent. No new children can be
        # created after all observed living parents have been suspended.
        for identity in sorted(
            known.values(), key=lambda item: (item.creation_time, item.pid), reverse=True
        ):
            process = _live_process(identity)
            if process is not None:
                try:
                    process.kill()
                    report["signalled"].append(identity.pid)
                except psutil.NoSuchProcess:
                    pass
        deadline = time.monotonic() + verification_seconds
        while True:
            remaining = [
                identity.pid for identity in known.values() if _live_process(identity) is not None
            ]
            if not remaining:
                report["termination_verified"] = True
                break
            if time.monotonic() >= deadline:
                report["remaining_pids"] = remaining
                break
            time.sleep(0.05)
    except (psutil.Error, OSError, RuntimeError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["owned_processes"] = [
        {"pid": item.pid, "creation_time": item.creation_time}
        for item in sorted(known.values(), key=lambda item: item.pid)
    ]
    return report


def run_supervised(
    command: list[str],
    *,
    cwd,
    stdout,
    stderr=subprocess.STDOUT,
    timeout: float | None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Popen-compatible synchronous execution with verified tree-wide timeout.

    This executes commands directly, without a shell. The caller should charge
    ``elapsed_seconds`` on SupervisedTimeout, including termination verification.
    TerminationUnverified must leave any active GPU reservation in place.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("Timeout must be positive before a worker is launched")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr)
    known: dict[int, ProcessIdentity] = {}
    try:
        owned = psutil.Process(process.pid)
        known[process.pid] = _identity(owned)
    except psutil.NoSuchProcess:
        # A very short-lived command may exit before psutil opens its handle.
        pass
    try:
        while True:
            _discover(known)
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                returncode = process.wait(
                    timeout=min(0.25, remaining) if remaining is not None else 0.25
                )
                break
            except subprocess.TimeoutExpired:
                continue
    except subprocess.TimeoutExpired as error:
        report = terminate_owned_tree(known)
        elapsed = time.monotonic() - started
        if not report["termination_verified"]:
            raise TerminationUnverified(
                "Timed-out worker tree could not be proven terminated; keep resource reservation",
                elapsed_seconds=elapsed,
                report=report,
            ) from error
        process.wait(timeout=1)
        raise SupervisedTimeout(
            command, timeout, elapsed_seconds=elapsed, termination_report=report
        ) from error
    except (psutil.Error, OSError) as error:
        report = terminate_owned_tree(known)
        raise TerminationUnverified(
            f"Worker process-tree supervision failed: {error}",
            elapsed_seconds=time.monotonic() - started,
            report=report,
        ) from error
    # A failed launcher must not silently leave its previously observed child.
    living = [item for item in known.values() if _live_process(item) is not None]
    if living:
        report = terminate_owned_tree(known)
        if not report["termination_verified"]:
            raise TerminationUnverified(
                "Exited launcher left an unverified child; keep resource reservation",
                elapsed_seconds=time.monotonic() - started,
                report=report,
            )
        raise subprocess.CalledProcessError(returncode or 1, command)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)
