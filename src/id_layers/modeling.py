from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import sha256_file


@dataclass
class ExtractionResult:
    embeddings: dict[tuple[int, str], np.ndarray]
    metadata: dict[str, Any]


def compare_extraction_results(
    first: ExtractionResult, second: ExtractionResult
) -> dict[str, Any]:
    if first.embeddings.keys() != second.embeddings.keys():
        raise RuntimeError("Repeated extraction produced different representation keys")
    maximum = 0.0
    fingerprints: dict[str, str] = {}
    for key in sorted(first.embeddings):
        first_values = np.ascontiguousarray(first.embeddings[key])
        second_values = np.ascontiguousarray(second.embeddings[key])
        if first_values.shape != second_values.shape:
            raise RuntimeError(f"Repeated extraction shape mismatch for {key}")
        maximum = max(maximum, float(np.abs(first_values - second_values).max()))
        layer, representation = key
        fingerprints[f"layer_{layer:02d}__{representation}"] = hashlib.sha256(
            first_values.tobytes()
        ).hexdigest()
    return {
        "all_exact": maximum == 0.0,
        "max_abs_difference": maximum,
        "embedding_sha256": fingerprints,
    }


def _third_party_revision(path: Path) -> str:
    base = [
        "git",
        "-c",
        f"safe.directory={path.as_posix()}",
        "-C",
        str(path),
    ]
    try:
        revision = subprocess.check_output(
            [*base, "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        dirty = subprocess.check_output(
            [*base, "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Unable to verify third-party revision at {path}: {error}") from error
    if dirty:
        raise RuntimeError(f"Third-party LVFace checkout has local changes:\n{dirty}")
    return revision


def load_lvface_model(
    model_config: dict[str, Any], locked_model: dict[str, Any]
) -> torch.nn.Module:
    source_path = Path(model_config["third_party_path"])
    weight_path = Path(model_config["weights_path"])
    if not source_path.is_dir():
        raise FileNotFoundError(
            f"Missing {source_path}. Run scripts/bootstrap_assets.py --accept-research-terms"
        )
    observed_revision = _third_party_revision(source_path)
    if observed_revision != locked_model["code_revision"]:
        raise ValueError(
            f"LVFace code revision mismatch: {observed_revision} != {locked_model['code_revision']}"
        )
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Missing {weight_path}. Run scripts/bootstrap_assets.py --accept-research-terms"
        )
    observed_hash = sha256_file(weight_path)
    if observed_hash != locked_model["weight_sha256"]:
        raise ValueError(
            f"LVFace weight checksum mismatch: {observed_hash} != {locked_model['weight_sha256']}"
        )

    source_string = str(source_path.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    from backbones import get_model  # type: ignore[import-not-found]

    model = get_model(model_config["architecture"])
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warm_up(model: torch.nn.Module, device: torch.device, image_size: int) -> None:
    with torch.inference_mode():
        dummy = torch.zeros(2, 3, image_size, image_size, device=device)
        for _ in range(2):
            model(dummy)
    _device_synchronize(device)


def extract_layer_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device_name: str,
    representations: Iterable[str],
    image_size: int,
) -> ExtractionResult:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = model.to(device)
    model.eval()
    requested = list(representations)
    supported = {"token_mean", "head_projected"}
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported representations: {unknown}")

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, block in enumerate(model.blocks, start=1):

        def capture(
            _module: torch.nn.Module, _inputs: Any, output: torch.Tensor, idx=layer_index
        ) -> None:
            captured[idx] = output

        handles.append(block.register_forward_hook(capture))

    buffers: dict[tuple[int, str], list[np.ndarray]] = {
        (layer, representation): []
        for layer in range(1, len(model.blocks) + 1)
        for representation in requested
    }
    max_final_difference = 0.0
    sample_count = 0

    try:
        _warm_up(model, device, image_size)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        _device_synchronize(device)
        start = time.perf_counter()

        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=device.type == "cuda")
                captured.clear()
                official = model(images)
                official_normalized = F.normalize(official.float(), dim=1)
                sample_count += images.shape[0]

                if len(captured) != len(model.blocks):
                    raise RuntimeError(
                        f"Expected {len(model.blocks)} hooked layers, captured {len(captured)}"
                    )
                for layer_index in range(1, len(model.blocks) + 1):
                    tokens = model.norm(captured[layer_index].float())
                    if "token_mean" in requested:
                        pooled = F.normalize(tokens.mean(dim=1), dim=1)
                        buffers[(layer_index, "token_mean")].append(
                            pooled.cpu().numpy().astype(np.float32, copy=False)
                        )
                    if "head_projected" in requested:
                        projected = F.normalize(model.feature(tokens.flatten(1)).float(), dim=1)
                        buffers[(layer_index, "head_projected")].append(
                            projected.cpu().numpy().astype(np.float32, copy=False)
                        )
                        if layer_index == len(model.blocks):
                            difference = (projected - official_normalized).abs().max().item()
                            max_final_difference = max(max_final_difference, difference)

        _device_synchronize(device)
        elapsed = time.perf_counter() - start
        peak_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
    finally:
        for handle in handles:
            handle.remove()

    embeddings = {key: np.concatenate(parts, axis=0) for key, parts in buffers.items()}
    if any(value.shape[0] != sample_count for value in embeddings.values()):
        raise RuntimeError("Layer embedding sample counts are inconsistent")
    if not all(np.isfinite(value).all() for value in embeddings.values()):
        raise RuntimeError("Non-finite layer embeddings detected")

    metadata = {
        "device": str(device),
        "sample_count": sample_count,
        "layer_count": len(model.blocks),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "images_per_second": sample_count / elapsed,
        "peak_memory_allocated_bytes": peak_bytes,
        "final_head_max_abs_difference": max_final_difference,
        "representations": requested,
    }
    return ExtractionResult(embeddings=embeddings, metadata=metadata)


def check_determinism(
    model: torch.nn.Module, sample: torch.Tensor, device_name: str
) -> dict[str, float | bool]:
    device = torch.device(device_name)
    model = model.to(device).eval()
    sample = sample.to(device)
    with torch.inference_mode():
        first = model(sample).float()
        second = model(sample).float()
    return {
        "max_abs_difference": float((first - second).abs().max().item()),
        "all_finite": bool(torch.isfinite(first).all() and torch.isfinite(second).all()),
    }
