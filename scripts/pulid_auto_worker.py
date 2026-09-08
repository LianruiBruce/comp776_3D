"""Offline PuLID v1.1 compatibility probe / fixed serial generation worker."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from id_layers.auto_generation import (  # noqa: E402
    PuLIDV11Backend,
    prepare_generation_suite,
    run_generation_cells,
    smoke_tasks,
)
from id_layers.auto_research import Budget, digest, dump  # noqa: E402

CACHE = ROOT / "artifacts/cache/auto_research/pulid"
RUN = ROOT / "artifacts/runs/20260905_auto_research"
REVISION = "1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa"
SOURCE = CACHE / "source" / f"PuLID-{REVISION}"
RUNTIME = CACHE / "runtime"


def verify_pulid_source():
    """Check executed source against the pinned archive plus the declared patch."""
    source_patch = json.loads((CACHE / "source_patch.json").read_text())
    archive = SOURCE.with_suffix(".zip")
    asset_records = json.loads((CACHE / "assets.resolved.json").read_text())
    archive_record = next(
        record for record in asset_records if Path(record["path"]).resolve() == archive.resolve()
    )
    if digest(archive) != archive_record["sha256"]:
        raise ValueError("PuLID pinned source archive hash mismatch")
    with zipfile.ZipFile(archive) as handle:
        archived_python = set()
        for entry in handle.infolist():
            if entry.is_dir():
                continue
            relative = Path(entry.filename).relative_to(SOURCE.name)
            local = SOURCE / relative
            if relative.suffix == ".py":
                archived_python.add(relative.as_posix())
            expected = (
                source_patch["after_sha256"]
                if relative.as_posix() == source_patch["file"]
                else hashlib.sha256(handle.read(entry)).hexdigest()
            )
            if digest(local) != expected:
                raise ValueError(
                    f"Executed PuLID source differs from pinned archive/declared patch: {relative}"
                )
        actual_python = {path.relative_to(SOURCE).as_posix() for path in SOURCE.rglob("*.py")}
        if actual_python != archived_python:
            raise ValueError("Unexpected Python source file added to PuLID checkout")


def configure_imports():
    verify_pulid_source()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(CACHE / "hf")
    os.environ["TORCH_HOME"] = str(CACHE / "torch")
    os.environ["NUMBA_CACHE_DIR"] = str(CACHE / "numba")
    sys.path.insert(0, str(ROOT / "third_party/InsightFace/python-package"))
    sys.path.insert(0, str(SOURCE))


def import_probe():
    configure_imports()
    import cv2
    import facexlib
    import numba
    import numpy as np
    import pulid.pipeline_v1_1 as module
    import torch
    from pulid import utils

    basic_file = RUNTIME / "audit/basicsr_v1.4.2_img_util.py"
    original = ast.parse(basic_file.read_text(encoding="utf-8"))
    vendored = ast.parse((SOURCE / "pulid/utils.py").read_text(encoding="utf-8"))
    functions = ("img2tensor", "tensor2img")
    same_ast = {}
    for name in functions:
        source_node = next(
            n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        local_node = next(
            n for n in vendored.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        # Upstream PuLID removes the redundant final "else" after an if-return.
        # Normalize only that exact control-flow-preserving syntax difference.
        for node in (source_node, local_node):
            tail = node.body[-1]
            if isinstance(tail, ast.If) and tail.orelse and isinstance(tail.body[-1], ast.Return):
                node.body.extend(tail.orelse)
                tail.orelse = []
        same_ast[name] = ast.dump(source_node, include_attributes=False) == ast.dump(
            local_node, include_attributes=False
        )
    if not all(same_ast.values()):
        raise RuntimeError("Vendored BasicSR helper AST does not match official v1.4.2")
    spec = importlib.util.spec_from_file_location("basic_sr_official_image_util_audit", basic_file)
    basic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic)
    rng = np.random.default_rng(20260905)
    trials = []
    for dtype in (np.uint8, np.float32, np.float64):
        for swap in (False, True):
            image = rng.integers(0, 256, (13, 17, 3), dtype=np.uint8).astype(dtype)
            first, second = (
                basic.img2tensor(image.copy(), bgr2rgb=swap),
                utils.img2tensor(image.copy(), bgr2rgb=swap),
            )
            same_tensor = torch.equal(first, second)
            same_image = np.array_equal(
                basic.tensor2img(first / 255, rgb2bgr=swap),
                utils.tensor2img(second / 255, rgb2bgr=swap),
            )
            trials.append(
                {
                    "dtype": str(dtype),
                    "swap_channels": swap,
                    "bitwise_equal_tensor": same_tensor,
                    "bitwise_equal_image": same_image,
                }
            )
    if not all(r["bitwise_equal_tensor"] and r["bitwise_equal_image"] for r in trials):
        raise RuntimeError("Image conversion compatibility probe failed")
    import subprocess

    package_lock = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    (CACHE / "environment.lock.txt").write_text(package_lock, encoding="utf-8")
    return {
        "state": "CPU_import_probe_complete",
        "source_revision": REVISION,
        "helper_ast_equal_after_redundant_else_normalization": same_ast,
        "helper_numerical_probes": trials,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numba": numba.__version__,
        "facexlib": facexlib.__version__,
        "python": sys.executable,
        "gpu_inference_attempted": False,
        "pipeline_class": module.PuLIDPipeline.__name__,
    }


@contextmanager
def local_asset_resolution():
    """Redirect known upstream fetch calls to already hashed local assets only."""
    configure_imports()
    import urllib.request

    import eva_clip.pretrained as eva_pretrained
    import facexlib.detection as face_detection
    import facexlib.parsing as face_parsing
    import pulid.pipeline_v1_1 as module
    import requests

    assets = json.loads((CACHE / "assets.resolved.json").read_text())
    for record in assets:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"PuLID local asset hash mismatch: {path}")

    def reject_network(*args, **kwargs):
        raise RuntimeError("Implicit network access forbidden in offline PuLID worker")

    def local_hf(repo_id, filename, *args, **kwargs):
        valid = {
            ("guozinan/PuLID", "pulid_v1.1.safetensors"): RUNTIME / "models/pulid_v1.1.safetensors",
            ("QuanSun/EVA-CLIP", "EVA02_CLIP_L_336_psz14_s6B.pt"): RUNTIME
            / "models/EVA02_CLIP_L_336_psz14_s6B.pt",
        }
        if (repo_id, filename) not in valid:
            raise ValueError(f"Unexpected model request: {repo_id}/{filename}")
        return str(valid[(repo_id, filename)])

    def local_snapshot(repo_id, *args, **kwargs):
        if repo_id != "DIAMONIK7777/antelopev2":
            raise ValueError(f"Unexpected snapshot request: {repo_id}")
        return str(RUNTIME / "models/antelopev2")

    def local_face_file(url, *args, **kwargs):
        filename = url.rsplit("/", 1)[-1]
        if filename not in {
            "detection_Resnet50_Final.pth",
            "parsing_bisenet.pth",
            "parsing_parsenet.pth",
        }:
            raise ValueError(f"Unexpected facexlib request: {url}")
        path = RUNTIME / "facexlib/weights" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    previous_cwd = Path.cwd()
    original_face_analysis = module.FaceAnalysis

    def required_face_analysis(*args, **kwargs):
        # Only detection and recognition contribute to PuLID conditioning.
        # Do not run unrelated demographic or auxiliary attribute models.
        kwargs["allowed_modules"] = ["detection", "recognition"]
        return original_face_analysis(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "FaceAnalysis", required_face_analysis))
        stack.enter_context(patch.object(module, "hf_hub_download", local_hf))
        stack.enter_context(patch.object(module, "snapshot_download", local_snapshot))
        stack.enter_context(patch.object(eva_pretrained, "hf_hub_download", local_hf))
        stack.enter_context(patch.object(face_detection, "load_file_from_url", local_face_file))
        stack.enter_context(patch.object(face_parsing, "load_file_from_url", local_face_file))
        stack.enter_context(patch.object(requests.sessions.Session, "request", reject_network))
        stack.enter_context(patch.object(urllib.request, "urlopen", reject_network))
        os.chdir(RUNTIME)
        try:
            yield module
        finally:
            os.chdir(previous_cwd)


def make_backend(tasks):
    import torch

    base_model = ROOT / "artifacts/cache/models/PhotoMaker/RealVisXL_V4.0"
    with local_asset_resolution() as module, torch.inference_mode():
        pipeline = module.PuLIDPipeline(sdxl_repo=str(base_model), sampler="dpmpp_2m")
        pipeline.pipe.set_progress_bar_config(disable=True)
        pipeline.pipe.enable_vae_tiling()
        # The locked RealVisXL VAE declares force_upcast=True. The official custom
        # sampler calls vae.decode directly, so honor that setting explicitly.
        if pipeline.pipe.vae.config.force_upcast:
            pipeline.pipe.vae.to(dtype=torch.float32)
        backend = PuLIDV11Backend(pipeline, tasks, source_revision=REVISION, source_path=SOURCE)
        # These modules are used only by get_id_embedding(), completed above.
        # Keep the UNet's id_adapter_attn_layers resident: they are still active.
        for component in (
            pipeline.clip_vision_model,
            pipeline.id_adapter,
            pipeline.face_helper.face_det,
            pipeline.face_helper.face_parse,
        ):
            component.to("cpu")
        pipeline.app = None
        pipeline.handler_ante = None
        pipeline.debug_img_list = []
        backend.inactive_encoders_released = True
        backend.face_analysis_modules = ["detection", "recognition"]
        # In the full matrix, release SDXL components while VAE alone decodes.
        # Device-only moves preserve every parameter's dtype/value and draw no RNG.
        # The UNet attention adapters move with their parent and return before
        # the next denoising call. No tiling geometry or quantization changes.
        original_decode = pipeline.pipe.vae.decode

        def decode_with_staged_residency(*args, **kwargs):
            parked = [pipeline.pipe.unet, pipeline.pipe.text_encoder, pipeline.pipe.text_encoder_2]
            for component in parked:
                component.to("cpu")
            torch.cuda.empty_cache()
            try:
                return original_decode(*args, **kwargs)
            finally:
                for component in parked:
                    component.to(pipeline.device)

        pipeline.pipe.vae.decode = decode_with_staged_residency
        backend.vae_decode_residency = "unet_and_text_encoders_temporarily_cpu; dtype_unchanged"
        gc.collect()
        torch.cuda.empty_cache()
    return backend


def smoke_face_gate(output):
    """Predeclared engineering gate: 16 decoded outputs and exactly one face each."""
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    from id_layers.generation import read_jsonl

    detector = FaceAnalysis(
        name="buffalo_l",
        root=str(ROOT / "artifacts/cache/models/insightface"),
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    records = []
    for row in read_jsonl(output / "generation_manifest.jsonl"):
        record = {"sample_id": row["sample_id"], "generation_status": row["status"]}
        if row["status"] == "complete":
            image = cv2.imread(str(output / row["output_relative_path"]))
            record["decoded"] = image is not None
            if image is not None:
                record["spatially_nonconstant"] = bool(np.any(np.ptp(image.reshape(-1, 3), axis=0)))
                record["detected_face_count"] = len(detector.get(image))
        records.append(record)
    passed = len(records) == 16 and all(
        r.get("decoded") and r.get("spatially_nonconstant") and r.get("detected_face_count") == 1
        for r in records
    )
    result = {
        "gate_passed": passed,
        "expected_cells": 16,
        "records": records,
        "criterion": "All 16 outputs decoded, spatially nonconstant, and exactly one SCRFD face",
        "detector": "locked SCRFD-10G, CPUExecutionProvider,640x640,threshold0.5",
        "interpretation": "Engineering validity only; no likeness/quality score",
    }
    dump(RUN / "pulid_engineering_smoke_gate.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("imports", "smoke", "full"), default="imports")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status_path = RUN / "pulid_compatibility.json"
    status = json.loads(status_path.read_text())
    if datetime.now(UTC) > datetime.fromisoformat(status["deadline_utc"]) and args.mode != "full":
        raise TimeoutError("PuLID fixed 60-minute compatibility deadline has elapsed")
    try:
        probe = import_probe()
        dump(CACHE / "CPU_import_probe.json", probe)
        if status.get("error"):
            status.setdefault("resolved_investigation_history", []).append(
                {
                    "previous_error": status.pop("error"),
                    "previous_traceback": status.pop("traceback", None),
                    "resolved_at_utc": datetime.now(UTC).isoformat(),
                    "resolution": (
                        "CPU import probe and image helper equivalence now pass; "
                        "redundant else-after-return is normalized for AST comparison"
                    ),
                }
            )
        source_patch_path = CACHE / "source_patch.json"
        source_patch = json.loads(source_patch_path.read_text())
        source_patch["helper_equivalence_verified"] = True
        source_patch["verification_artifact"] = str(CACHE / "CPU_import_probe.json")
        dump(source_patch_path, source_patch)
        if args.mode == "imports":
            status["state"] = "awaiting_GPU_slot"
            print(json.dumps(probe, indent=2))
            return
        import pandas as pd
        import yaml

        from id_layers.generation import generation_environment

        tasks, selection, _ = prepare_generation_suite(
            pd.read_csv(ROOT / "data/manifests/fei_full_seed20260902.csv"), ROOT / "data"
        )
        tasks = [
            replace(
                t,
                prompt=t.prompt.replace("person img", "person"),
                num_inference_steps=25,
                guidance_scale=7.0,
                start_merge_step=0,
            )
            for t in tasks
        ]
        if args.mode == "smoke":
            tasks = smoke_tasks(tasks, selection)
        elif not json.loads((RUN / "pulid_engineering_smoke_gate.json").read_text())["gate_passed"]:
            raise RuntimeError("Full PuLID matrix requires passed 16-cell engineering smoke")
        output = (
            Path(args.run_dir).resolve() if args.run_dir else RUN / f"generation_pulid_{args.mode}"
        )
        budget = Budget(RUN / "resource_ledger.json")
        remaining = budget.state["gpu_limit_seconds"] - budget.state["gpu_seconds"]
        reserve = min(remaining, 300 + len(tasks) * 90)
        if args.mode == "smoke":
            reserve = min(
                reserve,
                (
                    datetime.fromisoformat(status["deadline_utc"]) - datetime.now(UTC)
                ).total_seconds(),
            )
        metadata = {
            "resolved_config": {
                "task": "C",
                "model": "PuLID-v1.1",
                "phase": args.mode,
                "sampler": "dpmpp_2m",
                "steps": 25,
                "cfg": 7.0,
                "id_scale": 0.8,
                "num_zero": 20,
                "ortho_v2": True,
                "vae_precision": "float32 honoring locked VAE force_upcast=True",
                "latent_rng": "cpu",
                "base": str(ROOT / "artifacts/cache/models/PhotoMaker/RealVisXL_V4.0"),
            },
            "asset_lock": {
                "project": yaml.safe_load((ROOT / "configs/assets.lock.yaml").read_text()),
                "pulid": json.loads((CACHE / "assets.resolved.json").read_text()),
                "source_patch": json.loads((CACHE / "source_patch.json").read_text()),
            },
            "input_subset": selection["input_subset"],
            "selection": selection,
            "environment": generation_environment(),
        }
        start = time.perf_counter()
        with budget.gpu(f"PuLID-{args.mode}", reserve):
            result = run_generation_cells(
                tasks,
                output,
                lambda: make_backend(tasks),
                run_metadata=metadata,
                forbidden_source=ROOT
                / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot",
                resume=args.resume,
                latent_device="cpu",
                can_start=lambda estimate: time.perf_counter() - start + estimate < reserve,
                estimated_load_seconds=300,
                estimated_cell_seconds=90,
            )
        status[args.mode] = result
        if args.mode == "smoke":
            gate = smoke_face_gate(output)
            status["smoke_gate_passed"] = gate["gate_passed"]
        status["state"] = f"{args.mode}_{result['state']}"
        print(json.dumps(result, indent=2))
    except Exception as error:
        status["state"] = "blocked_external"
        status["error"] = f"{type(error).__name__}: {error}"
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        status["updated_utc"] = datetime.now(UTC).isoformat()
        dump(status_path, status)


if __name__ == "__main__":
    main()
