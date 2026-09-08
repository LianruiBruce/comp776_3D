"""Bootstrap exact diagnostic assets from the versioned lock, never a live branch.

Run in the independent .venv-auto. All package wheels are SHA256-verified before
offline installation. ``--verify-only`` performs no downloads or installations.
PuLID's separately managed source and models are also recorded in that lock.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from id_layers.auto_research import Budget, digest, dump  # noqa: E402


def locked_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    base = root.resolve()
    target = (base / path).resolve()
    if path.is_absolute() or target == base or base not in target.parents:
        raise ValueError(f"Locked path must remain relative to repository: {relative}")
    return target


def load_frozen_lock(path: Path, root: Path = ROOT) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Versioned asset lock required; no mutable-source fallback: {path}"
        )
    frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != 1:
        raise ValueError("Unsupported automatic-research asset lock schema")
    revision = frozen["diagnostic_source_revisions"]["adaface"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("AdaFace revision must be an immutable 40-character commit")
    for group in ("diagnostic_packages", "diagnostic_assets"):
        for name, record in frozen[group].items():
            locked_path(root, record["path"])
            sha = record["sha256"]
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"Invalid frozen SHA256 for {name}")
            if not isinstance(record["bytes"], int) or record["bytes"] < 1:
                raise ValueError(f"Invalid frozen byte count for {name}")
            if not record["url"].startswith("https://"):
                raise ValueError(f"Expected HTTPS source for {name}")
            if name in {
                "adaface_net.py",
                "adaface_inference.py",
                "adaface_README.md",
                "adaface_LICENSE",
            }:
                expected = f"https://raw.githubusercontent.com/mk-minchul/AdaFace/{revision}/"
                if not record["url"].startswith(expected):
                    raise ValueError(f"AdaFace source URL is not pinned to locked revision: {name}")
    return frozen


def verify_cached_asset(root: Path, name: str, record: dict) -> dict:
    path = locked_path(root, record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Frozen cached asset missing: {name}: {path}")
    observed = digest(path)
    if observed != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Frozen cached asset mismatch: {name}: {path}")
    return {"asset": name, "path": record["path"], "sha256": observed, "bytes": path.stat().st_size}


def verify_packaged_resources(frozen: dict) -> list[dict]:
    resources = []
    for name, record in frozen["packaged_resources"].items():
        distribution = importlib.metadata.distribution(record["package"])
        if distribution.version != record["package_version"]:
            raise ValueError(f"Installed package version mismatch for {name}")
        path = Path(distribution.locate_file(record["wheel_member"]))
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Installed packaged resource hash mismatch for {name}")
        resources.append(
            {
                "asset": name,
                "package": record["package"],
                "version": distribution.version,
                "wheel_member": record["wheel_member"],
                "sha256": record["sha256"],
            }
        )
    return resources


def acquire_locked_asset(budget: Budget, root: Path, name: str, record: dict) -> dict:
    target = locked_path(root, record["path"])
    # A different cached file is rejected. Do not replace it with a new version
    # merely because a remote URL or local metadata changed.
    if target.exists():
        verify_cached_asset(root, name, record)
    receipt = budget.download(
        record["url"], target, max_bytes=record["bytes"], expected_sha=record["sha256"]
    )
    verify_cached_asset(root, name, record)
    return {
        **receipt,
        "url": record["url"],
        "path": str(target),
        "sha256": record["sha256"],
        "bytes": record["bytes"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", action="store_true")
    parser.add_argument("--assets", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--lock", type=Path, default=ROOT / "configs/auto_research.assets.lock.yaml"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "artifacts/runs/20260905_auto_research"
    )
    args = parser.parse_args()
    frozen = load_frozen_lock(args.lock)
    if not args.packages and not args.assets and not args.verify_only:
        parser.error("Select --packages, --assets or --verify-only")
    cache = ROOT / "artifacts/cache/auto_research"
    run = args.run_dir.resolve()
    groups = []
    if args.packages or args.verify_only:
        groups.append("diagnostic_packages")
    if args.assets or args.verify_only:
        groups.append("diagnostic_assets")
    if args.verify_only:
        verified = [
            verify_cached_asset(ROOT, name, record)
            for group in groups
            for name, record in frozen[group].items()
        ]
        resources = verify_packaged_resources(frozen)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "mode": "verify_only",
                    "network_used": False,
                    "asset_count": len(verified),
                    "versioned_lock_sha256": digest(args.lock),
                    "adaface_revision": frozen["diagnostic_source_revisions"]["adaface"],
                    "packaged_resources": resources,
                },
                indent=2,
            )
        )
        return
    budget = Budget(run / "resource_ledger.json")
    lockpath = cache / "assets.json"
    lock = json.loads(lockpath.read_text(encoding="utf-8")) if lockpath.exists() else {}
    lock["versioned_lock_sha256"] = digest(args.lock)
    if args.packages:
        # Inference-only dependencies. Do not run old training requirements or change
        # torch/numpy/ORT. No live PyPI metadata selection is needed for locked wheels.
        for name, record in frozen["diagnostic_packages"].items():
            lock[name] = acquire_locked_asset(budget, ROOT, name, record)
            path = locked_path(ROOT, record["path"])
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(path)],
                check=True,
            )
            if importlib.metadata.version(name) != record["version"]:
                raise ValueError(f"Offline package version differs from frozen lock: {name}")
            dump(lockpath, lock)
        verify_packaged_resources(frozen)
    if args.assets:
        failures = {}
        for name, record in frozen["diagnostic_assets"].items():
            try:
                lock[name] = acquire_locked_asset(budget, ROOT, name, record)
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
            dump(lockpath, lock)
        lock["adaface_revision"] = frozen["diagnostic_source_revisions"]["adaface"]
        dump(lockpath, lock)
        dump(
            run / "asset_bootstrap_status.json",
            {
                "failures": failures,
                "assets_lock": str(lockpath),
                "versioned_lock_sha256": digest(args.lock),
            },
        )
        if failures:
            raise RuntimeError(f"Asset downloads failed: {failures}")


if __name__ == "__main__":
    main()
