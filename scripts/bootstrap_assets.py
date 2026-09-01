from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import requests
import yaml
from huggingface_hub import hf_hub_download
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
        raise RuntimeError(f"LVFace checkout mismatch: {observed} != {expected_revision}")
    dirty = git_output([*base, "status", "--porcelain", "--untracked-files=all"])
    if dirty:
        raise RuntimeError(f"LVFace checkout has local changes:\n{dirty}")


def ensure_lvface_checkout(spec: dict, target: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify pinned research assets")
    parser.add_argument(
        "--accept-research-terms",
        action="store_true",
        help="Acknowledge that FEI data and LVFace weights are local research-only assets",
    )
    arguments = parser.parse_args()
    if not arguments.accept_research_terms:
        parser.error("Read the dataset/model cards, then pass --accept-research-terms")

    lock_path = ROOT / "configs" / "assets.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    for archive in lock["datasets"]["fei_aligned"]["archives"]:
        download_checked(
            archive["url"],
            ROOT / "data" / "raw" / "fei" / archive["filename"],
            archive["sha256"],
        )

    model = lock["models"]["lvface_t_glint360k"]
    ensure_lvface_checkout(model, ROOT / "third_party" / "LVFace")
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
    print("All pinned assets are present and verified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"asset bootstrap failed: {error}", file=sys.stderr)
        raise
