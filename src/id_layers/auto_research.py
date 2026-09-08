"""Immutable-task bookkeeping and conservative resource accounting for auto research."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    tmp.replace(path)


def outside_source(output: Path, source: Path) -> Path:
    output, source = output.resolve(), source.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must not be inside the immutable source run")
    return output


def tree_hashes(root: Path) -> dict[str, dict]:
    return {
        p.relative_to(root).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class BudgetStop(RuntimeError):
    pass


class Budget:
    """Persist each streamed download and every GPU task, including unsuccessful tasks.

    GPU time is conservative elapsed wall time of the serial GPU worker (includes IO).
    An interrupted active worker reserves its entire declared limit until explicitly
    reconciled from its process log; it cannot silently give back unrecorded GPU time.
    """

    def __init__(self, path: Path, download_limit=30_000_000_000, gpu_limit=14400):
        self.path = path
        self.kind = None
        self.state = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "download_limit_bytes": download_limit,
                "gpu_limit_seconds": gpu_limit,
                "download_bytes": 0,
                "gpu_seconds": 0.0,
                "downloads": [],
                "gpu_tasks": [],
                "accounting": "payload bytes; GPU worker wall time including IO; serial only",
            }
        )

    @contextmanager
    def _lock(self):
        """OS-backed lock releases on process exit; no stale lock-file deadlock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+b") as f:
            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                f.seek(0)
                if os.name == "nt":
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def save(self):
        with self._lock():
            current = json.loads(self.path.read_text()) if self.path.exists() else {}
            foreign = (
                ["gpu_seconds", "gpu_tasks", "active_gpu"]
                if self.kind == "download"
                else ["download_bytes", "downloads"]
                if self.kind == "gpu"
                else []
            )
            for key in foreign:
                if key in current:
                    self.state[key] = current[key]
                elif key in self.state and key == "active_gpu":
                    self.state.pop(key)
            dump(self.path, self.state)

    def can_gpu(self, seconds):
        active = self.state.get("active_gpu", {}).get("reservation_seconds", 0)
        return self.state["gpu_seconds"] + active + seconds <= self.state["gpu_limit_seconds"]

    @contextmanager
    def gpu(self, name: str, estimate: float):
        import torch

        self.kind = "gpu"
        with self._lock():
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
            if self.state.get("active_gpu"):
                raise BudgetStop("An unreconciled GPU worker exists; concurrent GPU work forbidden")
            if not self.can_gpu(estimate):
                raise BudgetStop(f"Insufficient GPU budget for {name}: estimate {estimate}s")
            self.state["active_gpu"] = {
                "name": name,
                "reservation_seconds": estimate,
                "started_utc": datetime.now(UTC).isoformat(),
            }
            dump(self.path, self.state)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            record = {
                **self.state.pop("active_gpu"),
                "elapsed_seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            self.state["gpu_seconds"] += record["elapsed_seconds"]
            self.state["gpu_tasks"].append(record)
            self.save()

    def download(self, url: str, target: Path, *, max_bytes: int, expected_sha: str | None = None):
        """Stream with a hard byte limit; retries are counted, partial files never reused."""
        import requests

        self.kind = "download"
        if target.exists():
            observed = digest(target)
            if expected_sha and observed != expected_sha:
                raise ValueError(f"Cached asset hash mismatch: {target}")
            return {
                "path": str(target),
                "sha256": observed,
                "bytes": target.stat().st_size,
                "url": url,
                "reused": True,
            }
        if self.state["download_bytes"] + max_bytes > self.state["download_limit_bytes"]:
            raise BudgetStop(f"Download reservation exceeds remaining budget: {target.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".partial")
        record = {"url": url, "path": str(target), "bytes": 0, "status": "running"}
        self.state["downloads"].append(record)
        self.save()
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                record["resolved_url"] = r.url
                size = int(r.headers.get("content-length", 0))
                if size > max_bytes:
                    raise BudgetStop("Content-Length exceeds asset reservation")
                with part.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        record["bytes"] += len(chunk)
                        self.state["download_bytes"] += len(chunk)
                        if record["bytes"] > max_bytes:
                            raise BudgetStop("Download exceeds declared reservation")
                        f.write(chunk)
                        self.save()
            record["sha256"] = digest(part)
            if expected_sha and record["sha256"] != expected_sha:
                raise ValueError("Downloaded asset SHA256 mismatch")
            part.replace(target)
            record["status"] = "complete"
            return dict(record)
        except Exception as e:
            record["status"] = "failed"
            record["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.save()
