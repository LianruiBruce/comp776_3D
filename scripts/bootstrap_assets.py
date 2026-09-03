from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_checked(url: str, target: Path, expected_sha256: str) -> None:
    if target.exists():
        observed = sha256_file(target)
        if observed == expected_sha256:
            print(f"verified: {target}")
            return
        raise RuntimeError(f"Refusing to overwrite checksum-mismatched file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        raise RuntimeError(f"Remove stale partial download before retrying: {temporary}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary.open("wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=target.name
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    observed = sha256_file(temporary)
    if observed != expected_sha256:
        raise RuntimeError(f"Downloaded checksum mismatch for {target}: {observed}")
    temporary.replace(target)


def git_output(arguments: list[str]) -> str:
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT).strip()


def verify_clean_checkout(target: Path, expected_revision: str) -> None:
    safe = target.as_posix()
    base = ["git", "-c", f"safe.directory={safe}", "-C", str(target)]
    observed = git_output([*base, "rev-parse", "HEAD"])
    if observed != expected_revision:
        raise RuntimeError(f"Checkout mismatch at {target}: {observed} != {expected_revision}")
    dirty = git_output([*base, "status", "--porcelain", "--untracked-files=all"])
    if dirty:
        raise RuntimeError(f"Third-party checkout has local changes at {target}:\n{dirty}")


def ensure_git_checkout(spec: dict, target: Path) -> None:
    expected = spec["code_revision"]
    if target.exists():
        verify_clean_checkout(target, expected)
        print(f"verified: {target} @ {expected} (clean)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(target)], check=True)
    subprocess.run(
        ["git", "-C", str(target), "remote", "add", "origin", spec["code_repository"]],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth", "1", "origin", expected], check=True
    )
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
    verify_clean_checkout(target, expected)


def ensure_lvface_weights(spec: dict, target: Path) -> None:
    if target.exists():
        observed = sha256_file(target)
        if observed != spec["weight_sha256"]:
            raise RuntimeError(f"LVFace weight mismatch: {observed}")
        print(f"verified: {target}")
        return
    downloaded = Path(
        hf_hub_download(
            repo_id=spec["weight_repository"],
            filename=spec["weight_filename"],
            revision=spec["weight_revision"],
            local_dir=target.parents[1],
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve() != target.resolve():
        shutil.copy2(downloaded, target)
    observed = sha256_file(target)
    if observed != spec["weight_sha256"]:
        raise RuntimeError(f"LVFace weight mismatch after download: {observed}")


def ensure_insightface_detector(spec: dict, cache_root: Path) -> None:
    archive = cache_root / spec["archive_filename"]
    download_checked(spec["archive_url"], archive, spec["archive_sha256"])
    target = cache_root / "models" / "buffalo_l" / spec["detector_filename"]
    if target.exists():
        observed = sha256_file(target)
        if observed != spec["detector_sha256"]:
            raise RuntimeError(f"InsightFace detector mismatch: {observed}")
        print(f"verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with zipfile.ZipFile(archive) as source:
        try:
            member = source.getinfo(spec["detector_filename"])
        except KeyError as error:
            raise RuntimeError(
                f"Missing {spec['detector_filename']} in {archive}"
            ) from error
        with source.open(member) as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
    observed = sha256_file(temporary)
    if observed != spec["detector_sha256"]:
        raise RuntimeError(f"Extracted InsightFace detector mismatch: {observed}")
    temporary.replace(target)
    print(f"verified: {target}")


def ensure_insightface_recognition(spec: dict, cache_root: Path) -> None:
    archive = cache_root / spec["archive_filename"]
    download_checked(spec["archive_url"], archive, spec["archive_sha256"])
    target = cache_root / "models" / "buffalo_l" / spec["recognition_filename"]
    if target.exists():
        observed = sha256_file(target)
        if observed != spec["recognition_sha256"]:
            raise RuntimeError(f"InsightFace recognition model mismatch: {observed}")
        print(f"verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with zipfile.ZipFile(archive) as source:
        try:
            member = source.getinfo(spec["recognition_filename"])
        except KeyError as error:
            raise RuntimeError(
                f"Missing {spec['recognition_filename']} in {archive}"
            ) from error
        with source.open(member) as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
    observed = sha256_file(temporary)
    if observed != spec["recognition_sha256"]:
        raise RuntimeError(f"Extracted InsightFace recognition model mismatch: {observed}")
    temporary.replace(target)
    print(f"verified: {target}")


def ensure_huggingface_file(spec: dict, target: Path) -> None:
    if target.exists():
        observed = sha256_file(target)
        if observed != spec["weight_sha256"]:
            raise RuntimeError(f"Hugging Face weight mismatch at {target}: {observed}")
        print(f"verified: {target}")
        return
    downloaded = Path(
        hf_hub_download(
            repo_id=spec["weight_repository"],
            filename=spec["weight_filename"],
            revision=spec["weight_revision"],
            local_dir=target.parent,
        )
    )
    if downloaded.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, target)
    observed = sha256_file(target)
    if observed != spec["weight_sha256"]:
        raise RuntimeError(f"Hugging Face weight mismatch after download: {observed}")


def ensure_huggingface_snapshot(spec: dict, target: Path) -> None:
    snapshot_download(
        repo_id=spec["repository"],
        revision=spec["revision"],
        local_dir=target,
        allow_patterns=spec["allow_patterns"],
    )
    for relative, expected in spec["weight_sha256"].items():
        path = target / relative
        if not path.is_file():
            raise RuntimeError(f"Missing required snapshot weight: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Snapshot weight mismatch at {path}: {observed}")
        print(f"verified: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify pinned research assets")
    parser.add_argument(
        "--accept-research-terms",
        action="store_true",
        help="Acknowledge that FEI data and LVFace weights are local research-only assets",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "e1", "generation", "all"],
        default="smoke",
        help=(
            "smoke downloads FEI-aligned/LVFace; e1 adds FEI-full/SCRFD; "
            "generation/all also add the pinned PhotoMaker V2 stack"
        ),
    )
    arguments = parser.parse_args()
    if not arguments.accept_research_terms:
        parser.error("Read the dataset/model cards, then pass --accept-research-terms")

    lock_path = ROOT / "configs" / "assets.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    dataset_names = ["fei_aligned"]
    if arguments.profile in {"e1", "generation", "all"}:
        dataset_names.append("fei_full")
    for dataset_name in dataset_names:
        for archive in lock["datasets"][dataset_name]["archives"]:
            download_checked(
                archive["url"],
                ROOT / "data" / "raw" / "fei" / archive["filename"],
                archive["sha256"],
            )

    model = lock["models"]["lvface_t_glint360k"]
    ensure_git_checkout(model, ROOT / "third_party" / "LVFace")
    ensure_lvface_weights(
        model,
        ROOT
        / "artifacts"
        / "cache"
        / "models"
        / "LVFace"
        / "LVFace-T_Glint360K"
        / "LVFace-T_Glint360K.pt",
    )
    if arguments.profile in {"e1", "generation", "all"}:
        ensure_git_checkout(
            lock["models"]["insightface_buffalo_l_scrfd10g"],
            ROOT / "third_party" / "InsightFace",
        )
        ensure_insightface_detector(
            lock["models"]["insightface_buffalo_l_scrfd10g"],
            ROOT / "artifacts" / "cache" / "models" / "insightface",
        )
    if arguments.profile in {"generation", "all"}:
        insightface = lock["models"]["insightface_buffalo_l_scrfd10g"]
        ensure_insightface_recognition(
            insightface,
            ROOT / "artifacts" / "cache" / "models" / "insightface",
        )
        photomaker = lock["models"]["photomaker_v2"]
        ensure_git_checkout(photomaker, ROOT / "third_party" / "PhotoMaker")
        ensure_huggingface_file(
            photomaker,
            ROOT
            / "artifacts"
            / "cache"
            / "models"
            / "PhotoMaker"
            / "PhotoMaker-V2"
            / photomaker["weight_filename"],
        )
        ensure_huggingface_snapshot(
            lock["models"]["realvisxl_v4_fp16"],
            ROOT / "artifacts" / "cache" / "models" / "PhotoMaker" / "RealVisXL_V4.0",
        )
    print("All pinned assets are present and verified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"asset bootstrap failed: {error}", file=sys.stderr)
        raise
