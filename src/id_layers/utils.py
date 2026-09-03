from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def set_reproducibility(seed: int, deterministic: bool) -> None:
    if deterministic:
        # Required by CUDA >= 10.2 for deterministic cuBLAS matrix multiplications.
        # Set before the first CUDA operation in the experiment process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def utc_run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{name}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
    os.replace(temporary, path)


def snapshot_research_sources(repository: Path, destination: Path) -> dict[str, Any]:
    """Copy the exact executable research state needed to reconstruct a dirty run."""
    roots = [
        repository / "pyproject.toml",
        repository / "configs",
        repository / "requirements",
        repository / "scripts",
        repository / "src",
        repository / "tests",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
        else:
            raise FileNotFoundError(f"Missing source snapshot input: {root}")

    records = []
    for source in sorted(files):
        relative = source.relative_to(repository)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    return {"file_count": len(records), "files": records}


def python_package_lock() -> str:
    return subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
        stderr=subprocess.STDOUT,
    )


def git_revision(repository: Path) -> dict[str, Any]:
    safe_directory = repository.as_posix()
    base = ["git", "-c", f"safe.directory={safe_directory}", "-C", str(repository)]
    try:
        revision = subprocess.check_output(
            [*base, "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        status = subprocess.check_output(
            [*base, "status", "--porcelain"], text=True, stderr=subprocess.STDOUT
        ).strip()
        return {"revision": revision, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"revision": None, "dirty": None, "error": str(error)}


def environment_metadata(repository: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for module_name in [
        "cv2",
        "insightface",
        "numpy",
        "onnxruntime",
        "pandas",
        "sklearn",
        "timm",
        "torch",
        "torchvision",
    ]:
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[module_name] = "not-installed"

    gpu: dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": properties.total_memory,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
        try:
            gpu["driver_version"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            gpu["driver_version_error"] = str(error)

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "gpu": gpu,
        "repository": git_revision(repository),
    }
