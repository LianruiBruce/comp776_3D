"""Write new provenance supplements without modifying completed experiment tasks.

Default: CPU-only checks of official model hashes, pinned checkouts and installed
new diagnostic packages against their downloaded wheels. ``--latents`` is a
separate serial, budgeted GPU action and must wait for the current GPU worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from id_layers.auto_research import Budget, digest, dump, outside_source  # noqa: E402


def verify_file(path: Path, expected: str) -> dict:
    observed = digest(path)
    if observed != expected:
        raise ValueError(f"Locked asset mismatch: {path}; {observed} != {expected}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": observed,
        "bytes": path.stat().st_size,
        "matches_expected": True,
    }


def verify_checkout(path: Path, expected: str) -> dict:
    base = ["git", "-c", f"safe.directory={path.resolve().as_posix()}", "-C", str(path)]
    revision = subprocess.check_output([*base, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(
        [*base, "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != expected or changes:
        raise ValueError(
            f"Pinned checkout mismatch/modified tracked files: {path}; "
            f"revision={revision}, changes={changes}"
        )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "revision": revision,
        "expected_revision": expected,
        "tracked_worktree_clean": True,
    }


def verify_installed_wheel(package: str, wheel_record: dict) -> dict:
    """Hash every installed package payload against the exact locked wheel.

    dist-info is excluded because installers change RECORD/INSTALLER/direct_url.
    Payload bytes, including LPIPS's small learned Alex calibration weights, are
    verified individually; wheel bytes themselves are SHA256-verified first.
    """
    wheel = Path(wheel_record["path"])
    verified = verify_file(wheel, wheel_record["sha256"])
    distribution = importlib.metadata.distribution(package)
    payload = []
    with zipfile.ZipFile(wheel) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith("/") or ".dist-info/" in member:
                continue
            if ".data/" in member:
                raise ValueError(
                    f"Unsupported wheel relocation requires explicit mapping: {member}"
                )
            installed = Path(distribution.locate_file(member)).resolve()
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            if not installed.is_relative_to(ROOT):
                raise ValueError(
                    "Diagnostic package installed outside isolated repository environment: "
                    f"{installed}"
                )
            record = verify_file(installed, expected)
            record["wheel_member"] = member
            payload.append(record)
    return {
        "package": package,
        "version": distribution.version,
        "wheel": verified,
        "payload_file_count": len(payload),
        "installed_payload": payload,
    }


def cpu_verification(config: dict, output: Path) -> dict:
    lock_path = ROOT / "configs/assets.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    models = lock["models"]
    photo_cache = ROOT / "artifacts/cache/models/PhotoMaker"
    weights = []
    for relative, expected in models["realvisxl_v4_fp16"]["weight_sha256"].items():
        weights.append(verify_file(photo_cache / "RealVisXL_V4.0" / relative, expected))
    weights.append(
        verify_file(
            photo_cache / "PhotoMaker-V2/photomaker-v2.bin",
            models["photomaker_v2"]["weight_sha256"],
        )
    )
    lv = models["lvface_t_glint360k"]
    weights.append(
        verify_file(
            ROOT / "artifacts/cache/models/LVFace/LVFace-T_Glint360K/LVFace-T_Glint360K.pt",
            lv["weight_sha256"],
        )
    )
    face = models["insightface_buffalo_l_scrfd10g"]
    for kind in ("detector", "recognition"):
        weights.append(
            verify_file(
                ROOT
                / "artifacts/cache/models/insightface/models/buffalo_l"
                / face[kind + "_filename"],
                face[kind + "_sha256"],
            )
        )
    checkouts = [
        verify_checkout(ROOT / "third_party" / directory, models[key]["code_revision"])
        for directory, key in (
            ("PhotoMaker", "photomaker_v2"),
            ("InsightFace", "insightface_buffalo_l_scrfd10g"),
            ("LVFace", "lvface_t_glint360k"),
        )
    ]
    cache = ROOT / config["cache_root"]
    extra_path = cache / "assets.json"
    extra = json.loads(extra_path.read_text(encoding="utf-8"))
    extras = []
    for name, record in sorted(extra.items()):
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            extras.append(
                {
                    "asset": name,
                    "source_url": record.get("url"),
                    **verify_file(Path(record["path"]), record["sha256"]),
                }
            )
    packages = [verify_installed_wheel(name, extra[name]) for name in ("lpips", "mediapipe")]
    lpips_payload = packages[0]["installed_payload"]
    calibration = next(
        row for row in lpips_payload if row["wheel_member"] == "lpips/weights/v0.1/alex.pth"
    )
    dump(output / "project_assets.lock.json", lock)
    dump(output / "diagnostic_assets.lock.json", extra)
    record = {
        "status": "complete",
        "action": "CPU_only_asset_verification",
        "gpu_used": False,
        "project_lock_sha256": digest(lock_path),
        "supplement_lock_sha256": digest(extra_path),
        "weights": weights,
        "checkouts": checkouts,
        "diagnostic_assets": extras,
        "installed_diagnostic_packages": packages,
        "lpips_alex_v01_calibration": calibration,
        "scope": (
            "supplement recorded after earlier completed A/B tasks; "
            "those historical artifacts remain unchanged"
        ),
    }
    dump(output / "cpu_asset_verification.json", record)
    return record


def latent_verification(config: dict, output: Path) -> dict:
    from id_layers.auto_diagnostics import load_generation_manifest, validate_generation_records
    from id_layers.generation import make_fixed_latents, tensor_sha256

    run = ROOT / config["run_dir"]
    source = ROOT / config["source_run"]
    frozen = yaml.safe_load((source / "config.resolved.yaml").read_text(encoding="utf-8"))
    records = load_generation_manifest(source / "generation_manifest.jsonl")
    validate_generation_records(
        records,
        source,
        expected_count=192,
        expected_identities=frozen["data"]["identities"],
        expected_conditions=[row["name"] for row in frozen["generation"]["conditions"]],
        expected_prompts=list(frozen["generation"]["prompts"]),
    )
    device = frozen["generation"]["generator_device"]
    if device != "cuda":
        raise ValueError(f"Frozen old pilot expected CUDA RNG, observed {device}")
    by_group = {}
    for row in records:
        key = (row["identity_id"], int(row["base_seed"]))
        by_group.setdefault(key, []).append(row)
    verified = []
    budget = Budget(run / "resource_ledger.json")
    with budget.gpu("frozen_16_initial_latents_verification", 30):
        for (identity, seed), rows in sorted(by_group.items()):
            first = rows[0]
            ordinal = frozen["data"]["identities"].index(identity)
            expected_effective = seed + ordinal * 100_000
            if first["effective_seed"] != expected_effective:
                raise ValueError("Original effective seed deviates from frozen identity ordinal")
            latent = make_fixed_latents(
                expected_effective, first["height"], first["width"], device=device
            )
            observed = tensor_sha256(latent)
            if any(row["latent_sha256"] != observed for row in rows):
                raise ValueError(f"Recomputed initial CUDA latent hash mismatch: {identity}/{seed}")
            verified.append(
                {
                    "identity_id": identity,
                    "base_seed": seed,
                    "effective_seed": expected_effective,
                    "dtype": str(latent.dtype),
                    "shape": list(latent.shape),
                    "sha256": observed,
                    "verified_output_cell_count": len(rows),
                }
            )
    if len(verified) != 16 or sum(row["verified_output_cell_count"] for row in verified) != 192:
        raise ValueError("Latent verification accounting incomplete")
    result = {
        "status": "complete",
        "action": "GPU_initial_latent_regeneration",
        "device": device,
        "source_manifest_sha256": digest(source / "generation_manifest.jsonl"),
        "groups_verified": len(verified),
        "cells_covered": 192,
        "records": verified,
    }
    dump(output / "latent_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/auto_research.yaml")
    parser.add_argument(
        "--latents",
        action="store_true",
        help="Separate budgeted CUDA action; requires idle GPU slot",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ("_latents" if args.latents else "_cpu")
    output = outside_source(
        ROOT / config["run_dir"] / "provenance" / name, ROOT / config["source_run"]
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = (
            latent_verification(config, output)
            if args.latents
            else cpu_verification(config, output)
        )
        print(
            json.dumps(
                {"status": result["status"], "action": result["action"], "output": str(output)},
                indent=2,
            )
        )
    except Exception as error:
        dump(
            output / "failure.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
