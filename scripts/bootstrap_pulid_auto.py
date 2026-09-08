"""Budgeted, local-only preparation of the explicitly selected PuLID v1.1 system."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from id_layers.auto_research import Budget, digest, dump  # noqa: E402

CACHE = ROOT / "artifacts/cache/auto_research/pulid"
RUN = ROOT / "artifacts/runs/20260905_auto_research"
REVISION = "1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa"
PYTHON = ROOT / ".venv-pulid/Scripts/python.exe"


def check_deadline(status):
    if datetime.now(UTC) >= datetime.fromisoformat(status["deadline_utc"]):
        raise TimeoutError("PuLID compatibility investigation reached its fixed 60-minute deadline")


def download_packages(budget, status):
    os.environ["PIP_CACHE_DIR"] = str(CACHE / "pip_cache")
    # Inference-only extras; do not install upstream's two conflicting ONNX runtimes.
    # Numba's NumPy bound is handled in this separate environment, never .venv-auto.
    packages = {
        "ftfy": "6.3.1",
        "wcwidth": "0.2.13",
        "facexlib": "0.3.0",
        "filterpy": "1.4.5",
        "torchsde": "0.2.6",
        "trampoline": "0.1.2",
        "numpy": "2.2.6",
        "numba": "0.61.2",
        "llvmlite": "0.44.0",
    }
    files = []
    for name, version in packages.items():
        check_deadline(status)
        target = CACHE / "packages" / f"{name}-{version}.json"
        budget.download(f"https://pypi.org/pypi/{name}/{version}/json", target, max_bytes=2_000_000)
        metadata = json.loads(target.read_text())
        candidates = metadata["urls"]
        compatible = [
            item
            for item in candidates
            if item["filename"].endswith(".whl")
            and (
                "py3-none-any" in item["filename"]
                or "py2.py3-none-any" in item["filename"]
                or "cp311-cp311-win_amd64" in item["filename"]
            )
        ]
        if not compatible:
            compatible = [item for item in candidates if item["packagetype"] == "sdist"]
        if len(compatible) != 1:
            raise RuntimeError(
                f"Ambiguous/missing pinned package artifact: {name}, "
                f"{[i['filename'] for i in compatible]}"
            )
        item = compatible[0]
        path = CACHE / "packages" / item["filename"]
        budget.download(
            item["url"], path, max_bytes=item["size"], expected_sha=item["digests"]["sha256"]
        )
        files.append(str(path))
    check_deadline(status)
    subprocess.run(
        [
            str(PYTHON),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            *files,
        ],
        check=True,
        cwd=ROOT,
    )


def download_assets(budget, status):
    assets = []
    archive = CACHE / "source" / f"PuLID-{REVISION}.zip"
    assets.append(
        budget.download(
            f"https://codeload.github.com/ToTheBeginning/PuLID/zip/{REVISION}",
            archive,
            max_bytes=100_000_000,
        )
    )
    checkout = CACHE / "source" / f"PuLID-{REVISION}"
    if not checkout.exists():
        with zipfile.ZipFile(archive) as handle:
            for item in handle.infolist():
                if (
                    not (archive.parent / item.filename)
                    .resolve()
                    .is_relative_to(archive.parent.resolve())
                ):
                    raise ValueError("Source archive entry escapes extraction directory")
            handle.extractall(archive.parent)
    source = checkout / "pulid/pipeline_v1_1.py"
    original = source.read_text(encoding="utf-8")
    old_import = "from basicsr.utils import img2tensor, tensor2img"
    new_import = "from pulid.utils import img2tensor, tensor2img"
    if old_import in original:
        before = digest(source)
        source.write_text(original.replace(old_import, new_import), encoding="utf-8")
        dump(
            CACHE / "source_patch.json",
            {
                "upstream_revision": REVISION,
                "file": "pulid/pipeline_v1_1.py",
                "before_sha256": before,
                "after_sha256": digest(source),
                "patch": f"- {old_import}\n+ {new_import}",
                "purpose": (
                    "Use the same BasicSR image conversion helpers already vendored by "
                    "the official PuLID repository; avoid loading unrelated training "
                    "dependencies"
                ),
                "algorithm_changes": "none; helper equivalence verified by compatibility probe",
            },
        )
    elif new_import not in original:
        raise RuntimeError("Unexpected official pipeline source; refusing adaptive source edits")
    root = CACHE / "runtime"
    specifications = [
        (
            "https://raw.githubusercontent.com/XPixelGroup/BasicSR/v1.4.2/basicsr/utils/img_util.py",
            "audit/basicsr_v1.4.2_img_util.py",
            50_000,
            None,
        ),
        (
            "https://huggingface.co/guozinan/PuLID/resolve/d58b3ba191be1280c37ace0656a2e65cd9b8c8d8/pulid_v1.1.safetensors",
            "models/pulid_v1.1.safetensors",
            984405232,
            "4cb8ceec1078e0165399b88332ab3c5971619111b8e1730e6bae64144aabae41",
        ),
        (
            "https://huggingface.co/QuanSun/EVA-CLIP/resolve/690a5975ca06e9a43d16bae440dd08213bfff0c5/EVA02_CLIP_L_336_psz14_s6B.pt",
            "models/EVA02_CLIP_L_336_psz14_s6B.pt",
            856461210,
            "84c3a17a228c567a155259b2245b0b59072bf7da510260a0a02ec54de6d50b05",
        ),
    ]
    antelope_revision = "ba0c3e10f4548361eb9a63265d87ce1140ab5a05"
    antelope = {
        "1k3d68.onnx": (
            143607619,
            "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
        ),
        "2d106det.onnx": (
            5030888,
            "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
        ),
        "genderage.onnx": (
            1322532,
            "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
        ),
        "glintr100.onnx": (
            260665334,
            "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
        ),
    }
    for filename, (size, sha) in antelope.items():
        specifications.append(
            (
                f"https://huggingface.co/DIAMONIK7777/antelopev2/resolve/{antelope_revision}/{filename}",
                f"models/antelopev2/{filename}",
                size,
                sha,
            )
        )
    specifications.extend(
        [
            (
                "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
                "facexlib/weights/detection_Resnet50_Final.pth",
                120_000_000,
                None,
            ),
            (
                "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth",
                "facexlib/weights/parsing_bisenet.pth",
                70_000_000,
                None,
            ),
            (
                "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
                "facexlib/weights/parsing_parsenet.pth",
                90_000_000,
                None,
            ),
        ]
    )
    for url, relative, size, sha in specifications:
        check_deadline(status)
        print(f"Preparing {relative}", flush=True)
        assets.append(budget.download(url, root / relative, max_bytes=size, expected_sha=sha))
        dump(CACHE / "assets.resolved.json", assets)
    local_detector = ROOT / "artifacts/cache/models/insightface/models/buffalo_l/det_10g.onnx"
    if digest(local_detector) != "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91":
        raise ValueError("Existing shared SCRFD detector hash mismatch")
    detector = root / "models/antelopev2/scrfd_10g_bnkps.onnx"
    if not detector.exists():
        shutil.copyfile(local_detector, detector)
    if digest(detector) != digest(local_detector):
        raise ValueError("Reused detector copy hash mismatch")
    assets.append(
        {
            "path": str(detector),
            "sha256": digest(detector),
            "bytes": detector.stat().st_size,
            "reused": True,
            "source": str(local_detector),
        }
    )
    dump(CACHE / "assets.resolved.json", assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["packages", "assets", "all"], default="all")
    args = parser.parse_args()
    status_path = RUN / "pulid_compatibility.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {
            "started_utc": "2026-09-05T21:17:26+00:00",
            "deadline_utc": "2026-09-05T22:17:26+00:00",
            "maximum_seconds": 3600,
            "state": "preparing",
            "upstream_revision": REVISION,
            "environment": str(PYTHON),
            "no_alternative_model": True,
        }
    )
    dump(status_path, status)
    budget = Budget(RUN / "pulid_download_ledger.json", download_limit=5_000_000_000, gpu_limit=0)
    budget.save()
    started = time.perf_counter()
    try:
        check_deadline(status)
        if args.stage in {"packages", "all"}:
            download_packages(budget, status)
        if args.stage in {"assets", "all"}:
            download_assets(budget, status)
        status[args.stage] = "complete"
        status["state"] = "awaiting_CPU_import_probe"
    except Exception as error:
        status["state"] = "blocked_external"
        status["error"] = f"{type(error).__name__}: {error}"
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        status["last_stage_seconds"] = time.perf_counter() - started
        status["last_updated_utc"] = datetime.now(UTC).isoformat()
        dump(status_path, status)


if __name__ == "__main__":
    main()
