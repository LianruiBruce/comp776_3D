"""Deterministic native-resolution VAE controls, independent of generation claims.

The frozen RealVisXL VAE is a codec diagnostic. Its errors are not an additive
or uniquely identifiable fraction of the full generation likeness gap.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import auto_diagnostics as diag
from .auto_research import digest, dump, outside_source

VAE_PREPROCESSING_VERSION = (
    "rgb_exif_native_reflect_bottomright_multiple8_float32_mode_directdecode_v1"
)


def prepare_vae_input(
    image: Image.Image, *, multiple: int = 8
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Preserve photo geometry; pad bottom/right, with no resizing or stretching."""
    if multiple < 1:
        raise ValueError("Padding multiple must be positive")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    bottom, right = (-height) % multiple, (-width) % multiple
    padded = (
        np.pad(rgb, ((0, bottom), (0, right), (0, 0)), mode="reflect") if bottom or right else rgb
    )
    tensor = torch.from_numpy(
        np.ascontiguousarray((padded.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1))
    )[None]
    return tensor, {
        "original_width": width,
        "original_height": height,
        "padded_width": width + right,
        "padded_height": height + bottom,
        "padding_left": 0,
        "padding_top": 0,
        "padding_right": right,
        "padding_bottom": bottom,
        "padding_mode": "reflect",
        "preprocessing_version": VAE_PREPROCESSING_VERSION,
    }


def restore_vae_output(tensor: torch.Tensor, geometry: dict[str, Any]) -> np.ndarray:
    """Native HWC RGB float32 in [0,1], clipping only after the decode."""
    if tensor.ndim != 4 or tensor.shape[:2] != (1, 3):
        raise ValueError("Expected one decoded RGB BCHW tensor")
    if tensor.shape[-2:] != (geometry["padded_height"], geometry["padded_width"]):
        raise ValueError("VAE decoder changed padded spatial dimensions")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Nonfinite VAE reconstruction")
    rgb = ((tensor.detach().float().cpu()[0] + 1) / 2).clamp(0, 1)
    return (
        rgb[:, : geometry["original_height"], : geometry["original_width"]]
        .permute(1, 2, 0)
        .numpy()
        .copy()
    )


def decode_posterior_mode(vae: Any, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the VAE's raw posterior mode, without diffusion latent scaling.

    A diffusion pipeline uses z_diff = z_raw * scaling_factor and hands the
    decoder z_diff / scaling_factor. Applying only one side would be wrong.
    Direct VAE reconstruction requires neither operation.
    """
    if tensor.dtype != torch.float32:
        raise ValueError("VAE diagnostic requires float32 input")
    with torch.inference_mode():
        latent = vae.encode(tensor, return_dict=True).latent_dist.mode()
        if latent.dtype != torch.float32 or not bool(torch.isfinite(latent).all()):
            raise ValueError("VAE posterior mode must be finite float32")
        decoded = vae.decode(latent, return_dict=True).sample
    if decoded.dtype != torch.float32 or not bool(torch.isfinite(decoded).all()):
        raise ValueError("VAE decoder must produce finite float32")
    return latent, decoded


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, Any]:
    """Native RGB [0,1] PSNR/SSIM with explicit constant-image conventions."""
    from skimage.metrics import structural_similarity

    original, reconstructed = (
        np.asarray(original, dtype=np.float64),
        np.asarray(reconstructed, dtype=np.float64),
    )
    if original.shape != reconstructed.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("Reconstruction metrics require matching HWC RGB dimensions")
    if min(original.shape[:2]) < 11:
        raise ValueError("Gaussian SSIM requires both image dimensions >=11")
    if (
        not np.isfinite(original).all()
        or not np.isfinite(reconstructed).all()
        or min(float(original.min()), float(reconstructed.min())) < 0
        or max(float(original.max()), float(reconstructed.max())) > 1
    ):
        raise ValueError("Reconstruction pixels must be finite in [0,1]")
    mse = float(np.square(original - reconstructed).mean())
    ssim = float(
        structural_similarity(
            original,
            reconstructed,
            data_range=1.0,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )
    return {
        "mse_rgb_0_1": mse,
        "psnr_db": -10 * math.log10(mse) if mse > 0 else None,
        "psnr_is_infinite": mse == 0,
        "ssim_rgb": ssim,
        "ssim_definition": "Gaussian sigma1.5 population covariance; channel mean; data_range1",
    }


def _write_csv(path: Path, records: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    (records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)).to_csv(
        temporary, index=False
    )
    temporary.replace(path)


def _read_terminal_unit(
    path: Path, source_hash: str, config_hash: str, output: Path
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["source_sha256"] != source_hash or record["config_sha256"] != config_hash:
        raise ValueError("VAE resume provenance changed")
    if record["status"] not in {"complete", "failed"}:
        raise ValueError("VAE unit record has unknown status")
    if record["status"] == "failed" and not record.get("error"):
        raise ValueError("Failed VAE unit record lacks an explicit reason")
    if record["status"] == "complete":
        for name in ("control", "reconstruction"):
            path = diag.resolve_inside(output, record[name + "_relative_path"])
            if digest(path) != record[name + "_sha256"]:
                raise ValueError(f"VAE completed output hash changed: {path}")
    return record


def _unit_reconstruction(
    vae: Any, row: dict[str, Any], root: Path, output: Path, config_hash: str, *, repeat: bool
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    original_path = diag.resolve_inside(root / "data", row["source_relative_path"])
    if digest(original_path) != row["source_sha256"]:
        raise ValueError(f"VAE source hash mismatch: {sample_id}")
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "identity_id": row["identity_id"],
        "condition": row["condition"],
        "source_sha256": row["source_sha256"],
        "config_sha256": config_hash,
        "status": "failed",
        "error": None,
    }
    started = time.perf_counter()
    try:
        original = diag.load_rgb(original_path)
        tensor, geometry = prepare_vae_input(original)
        tensor = tensor.to(device="cuda", dtype=torch.float32)
        latent, decoded = decode_posterior_mode(vae, tensor)
        float_rgb = restore_vae_output(decoded, geometry)
        uint8_rgb = np.rint(float_rgb * 255).clip(0, 255).astype(np.uint8)
        original_float = np.asarray(original, dtype=np.float64) / 255
        metrics_float = reconstruction_metrics(original_float, float_rgb)
        metrics_png = reconstruction_metrics(original_float, uint8_rgb.astype(np.float64) / 255)
        record.update(geometry)
        record.update({"float__" + key: value for key, value in metrics_float.items()})
        record.update({"png__" + key: value for key, value in metrics_png.items()})
        latent_array = latent.detach().cpu().numpy()
        record["raw_posterior_mode_sha256"] = hashlib.sha256(
            latent_array.tobytes(order="C")
        ).hexdigest()
        record["raw_posterior_mode_shape"] = list(latent_array.shape)
        if repeat:
            latent_repeat, decoded_repeat = decode_posterior_mode(vae, tensor)
            delta = {
                "sample_id": sample_id,
                "latent_max_abs_delta": float((latent - latent_repeat).abs().max().item()),
                "decoder_max_abs_delta": float((decoded - decoded_repeat).abs().max().item()),
                "absolute_tolerance": 1e-6,
            }
            delta["within_tolerance"] = (
                max(delta["latent_max_abs_delta"], delta["decoder_max_abs_delta"]) <= 1e-6
            )
            record["repeatability"] = delta
            del latent_repeat, decoded_repeat
            if not delta["within_tolerance"]:
                raise ValueError("Deterministic VAE mode repeat exceeded 1e-6 tolerance")
        for name, image in (("control", original), ("reconstruction", Image.fromarray(uint8_rgb))):
            relative = Path(name) / f"{sample_id}.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            record[name + "_relative_path"] = relative.as_posix()
            record[name + "_sha256"] = digest(target)
        record["control_pixel_identity_verified"] = np.array_equal(
            np.asarray(diag.load_rgb(output / record["control_relative_path"])),
            np.asarray(original),
        )
        if not record["control_pixel_identity_verified"]:
            raise ValueError("Passthrough control changed decoded source pixels")
        record["status"] = "complete"
        del latent, decoded, tensor
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        gc.collect()
        torch.cuda.empty_cache()
    record["elapsed_seconds"] = time.perf_counter() - started
    return record


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    norms = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norms <= 1e-12 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Invalid identity embedding in VAE diagnostic")
    return float(np.clip(np.dot(left.astype(np.float64), right.astype(np.float64)) / norms, -1, 1))


def run_vae(root: Path, source: Path, output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Run task B into an independent directory; completed/failed units are frozen.

    GPU resource reservation/serialization belongs to the root runner. This
    function never downloads assets and never calls the old generation evaluator.
    """
    from diffusers import AutoencoderKL

    from .auto_evaluation import AdaFaceEncoder, make_encoders, make_lpips

    root, source, output = Path(root).resolve(), Path(source).resolve(), Path(output).resolve()
    outside_source(output, source)
    manifest_path = source / "dataset_manifest.csv"
    dataset = pd.read_csv(manifest_path, keep_default_na=False).sort_values("sample_id")
    diag.unique_by_id(dataset.to_dict("records"))
    if len(dataset) != 56 or dataset.identity_id.nunique() != 8:
        raise ValueError("The frozen VAE pilot requires exactly 56 source photos from 8 identities")
    model_root = root / "artifacts/cache/models/PhotoMaker/RealVisXL_V4.0"
    weights_path = model_root / "vae/diffusion_pytorch_model.fp16.safetensors"
    expected_weight = lock["models"]["realvisxl_v4_fp16"]["weight_sha256"][
        "vae/diffusion_pytorch_model.fp16.safetensors"
    ]
    if digest(weights_path) != expected_weight:
        raise ValueError("Frozen RealVisXL VAE weight hash mismatch")
    config = {
        "schema_version": 1,
        "task": "B_vae_reconstruction",
        "source_manifest_sha256": digest(manifest_path),
        "source_photo_count": 56,
        "vae_weight_sha256": expected_weight,
        "vae_config_sha256": digest(model_root / "vae/config.json"),
        "precision": "float32",
        "posterior": "mode",
        "latent_scaling": "none; decode raw VAE posterior mode",
        "preprocessing_version": VAE_PREPROCESSING_VERSION,
        "tiling_enabled": True,
        "native_pixel_metrics": (
            "decoded source RGB vs clipped float reconstruction AND saved uint8 PNG"
        ),
        "lpips_spaces": ["native", "whole256_aspect_padded", "fixed_face256"],
        "face_alignment": (
            "fixed original SCRFD five points; separately redetect controls and reconstructions"
        ),
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_rows = dataset.to_dict("records")
    # Verify inputs even when returning an already completed task on resume.
    for row in source_rows:
        if (
            digest(diag.resolve_inside(root / "data", row["source_relative_path"]))
            != row["source_sha256"]
        ):
            raise ValueError(f"Frozen real-photo source hash mismatch: {row['sample_id']}")
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if summary["config_sha256"] != config_hash:
            raise ValueError("Completed VAE summary configuration mismatch")
        for row in source_rows:
            if (
                _read_terminal_unit(
                    output / "units" / f"{row['sample_id']}.json",
                    row["source_sha256"],
                    config_hash,
                    output,
                )
                is None
            ):
                raise ValueError("Completed VAE summary is missing an expected terminal unit")
        return summary
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and json.loads(
        (output / "config.json").read_text(encoding="utf-8")
    ) != config:
        raise ValueError("VAE task configuration changed on resume")
    dump(output / "config.json", config)
    _write_csv(output / "manifest.csv", dataset)
    units = []
    pending = []
    for row in source_rows:
        unit = _read_terminal_unit(
            output / "units" / f"{row['sample_id']}.json", row["source_sha256"], config_hash, output
        )
        if unit is None:
            pending.append(row)
        else:
            units.append(unit)
    if pending:
        vae = (
            AutoencoderKL.from_pretrained(
                model_root,
                subfolder="vae",
                variant="fp16",
                torch_dtype=torch.float32,
                local_files_only=True,
                use_safetensors=True,
            )
            .to("cuda")
            .eval()
            .requires_grad_(False)
        )
        vae.enable_tiling()
        dump(
            output / "vae_runtime.json",
            {
                "dtype": str(next(vae.parameters()).dtype),
                "scaling_factor_config": float(vae.config.scaling_factor),
                "scaling_factor_applied": False,
                "force_upcast_config": bool(vae.config.force_upcast),
                "tiling_enabled": bool(vae.use_tiling),
                "tile_sample_min_size": int(vae.tile_sample_min_size),
                "tile_latent_min_size": int(vae.tile_latent_min_size),
                "tiling_activated_for_native_640x480": int(vae.tile_sample_min_size) < 640
                or int(vae.tile_sample_min_size) < 480,
            },
        )
        for row in pending:
            unit = _unit_reconstruction(
                vae,
                row,
                root,
                output,
                config_hash,
                repeat=row["sample_id"] == source_rows[0]["sample_id"],
            )
            dump(output / "units" / f"{row['sample_id']}.json", unit)
            units.append(unit)
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    units.sort(key=lambda row: row["sample_id"])
    _write_csv(output / "reconstruction_metrics.csv", units)
    by_id = {row["sample_id"]: row for row in source_rows}
    good = [unit for unit in units if unit["status"] == "complete"]
    failures: dict[str, str] = {}
    if not good:
        # A fully failed upstream task still has all 56 immutable unit records.
        # Do not start more GPU models or emit means of empty arrays.
        summary = {
            "status": "complete_with_failures",
            "config_sha256": config_hash,
            "source_photo_count": 56,
            "reconstructed_count": 0,
            "failed_unit_count": len(units),
            "model_failures": {"downstream": "No successfully reconstructed photos"},
            "scope": "development codec diagnostic; not additive generation loss attribution",
            "mean_native_float_psnr_db": None,
            "mean_native_float_ssim": None,
            "identity_equal_mean_psnr_db": None,
            "identity_equal_mean_ssim": None,
            "repeatability": [unit["repeatability"] for unit in units if "repeatability" in unit],
            "identity_changes": [],
            "lpips_by_space": [],
        }
        dump(output / "summary.json", summary)
        return summary
    alignment_rows: list[dict[str, Any]] = []
    crops: dict[str, Path] = {}
    face256: dict[str, Path] = {}
    native_paths: dict[str, Path] = {}
    identifiers: dict[str, dict[str, Any]] = {}
    face, lv = make_encoders(root, lock)
    for unit in good:
        sample_id = unit["sample_id"]
        original_metadata = by_id[sample_id]
        original_points = json.loads(original_metadata["face_landmarks_5_json"])
        original_valid = (
            original_metadata["alignment_status"] == "aligned"
            and int(original_metadata["detected_face_count"]) == 1
        )
        for role in ("control", "reconstruction"):
            path = output / unit[role + "_relative_path"]
            image = diag.load_rgb(path)
            native_id = f"{sample_id}__{role}"
            native_paths[native_id] = path
            if original_valid:
                for size in (112, 256):
                    crop, matrix = diag.fixed_face_crop(image, original_points, size=size)
                    target = output / f"fixed{size}" / f"{native_id}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(target)
                    (crops if size == 112 else face256)[
                        native_id + "__fixed" if size == 112 else native_id
                    ] = target
                identifiers[native_id + "__fixed"] = {
                    "sample_id": native_id + "__fixed",
                    "source_sha256": unit[role + "_sha256"],
                }
            alignment_rows.append(
                {
                    "sample_id": sample_id,
                    "role": role,
                    "alignment": "fixed_original",
                    "status": "complete" if original_valid else "failed",
                    "detected_face_count": int(original_metadata["detected_face_count"]),
                    "error": None if original_valid else "original_alignment_not_single_face",
                }
            )
            _, meta = face.encode_and_align(path)
            valid = meta["alignment_status"] == "aligned" and meta["end_to_end_valid"]
            alignment_rows.append(
                {
                    "sample_id": sample_id,
                    "role": role,
                    "alignment": "independent_redetection",
                    "status": "complete" if valid else "failed",
                    **meta,
                    "error": None
                    if valid
                    else meta.get("alignment_failure_reason") or "multiple_faces_detected",
                }
            )
            if valid:
                crop, _ = diag.fixed_face_crop(image, meta["face_landmarks_5"], size=112)
                sid = native_id + "__redetected"
                target = output / "redetected112" / f"{sid}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target)
                crops[sid] = target
                identifiers[sid] = {"sample_id": sid, "source_sha256": unit[role + "_sha256"]}
    _write_csv(output / "alignment_coverage.csv", alignment_rows)
    # Release the detector, which is no longer needed during FR and LPIPS passes.
    del face
    encoders = {"lvface": lv}
    try:
        encoders["adaface"] = AdaFaceEncoder(root / "artifacts/cache/auto_research")
    except Exception as error:
        failures["adaface"] = f"{type(error).__name__}: {error}"
    embedding_rows = []
    for name, encoder in encoders.items():
        ordered = sorted(crops)
        if not ordered:
            failures[name] = "No successfully aligned images"
            continue
        cache_path = output / f"embeddings_{name}.npz"
        descriptors = [identifiers[sid] for sid in ordered]
        if cache_path.exists():
            values = diag.load_keyed_array_cache(
                cache_path,
                descriptors,
                preprocessing_version="vae_png_fixed_or_redetect_arcface112_v1",
            )["embedding"]
        else:
            values = encoder.encode_paths([crops[sid] for sid in ordered])
            diag.save_keyed_array_cache(
                cache_path,
                descriptors,
                {"embedding": values},
                preprocessing_version="vae_png_fixed_or_redetect_arcface112_v1",
            )
        embeddings = dict(zip(ordered, values, strict=True))
        for unit in units:
            sid = unit["sample_id"]
            for comparison, lhs, rhs in (
                (
                    "fixed_original_alignment",
                    f"{sid}__control__fixed",
                    f"{sid}__reconstruction__fixed",
                ),
                (
                    "independent_redetection",
                    f"{sid}__control__redetected",
                    f"{sid}__reconstruction__redetected",
                ),
                (
                    "reconstruction_alignment_jitter",
                    f"{sid}__reconstruction__fixed",
                    f"{sid}__reconstruction__redetected",
                ),
            ):
                valid = lhs in embeddings and rhs in embeddings
                cosine = _cosine(embeddings[lhs], embeddings[rhs]) if valid else None
                embedding_rows.append(
                    {
                        "sample_id": sid,
                        "identity_id": unit["identity_id"],
                        "condition": unit["condition"],
                        "evaluator": name,
                        "comparison": comparison,
                        "status": "complete" if valid else "failed",
                        "error": None
                        if valid
                        else "generation_or_single_face_alignment_unavailable",
                        "identity_cosine": cosine,
                        "identity_cosine_distance": 1 - cosine if cosine is not None else None,
                    }
                )
    _write_csv(output / "identity_changes.csv", embedding_rows)
    del encoders, lv, encoder
    gc.collect()
    torch.cuda.empty_cache()
    pairs = [
        {
            "pair_id": "vae__" + unit["sample_id"],
            "sample_id": unit["sample_id"],
            "identity_id": unit["identity_id"],
            "condition": unit["condition"],
            "lhs_id": unit["sample_id"] + "__control",
            "rhs_id": unit["sample_id"] + "__reconstruction",
        }
        for unit in units
    ]
    lpips = make_lpips(root / "artifacts/cache/auto_research")
    perceptual = lpips.score_pairs(pairs, native_paths, space="whole") + lpips.score_pairs(
        pairs, face256, space="face"
    )
    for pair in pairs:
        if pair["lhs_id"] not in native_paths or pair["rhs_id"] not in native_paths:
            perceptual.append(
                {
                    **pair,
                    "space": "native",
                    "metric": "lpips_alex_v0.1",
                    "status": "failed",
                    "distance": None,
                    "error": "reconstruction_unavailable",
                }
            )
            continue
        tensors = [
            np.ascontiguousarray(
                (
                    np.asarray(diag.load_rgb(native_paths[pair[key]]), dtype=np.float32) / 127.5 - 1
                ).transpose(2, 0, 1)
            )
            for key in ("lhs_id", "rhs_id")
        ]
        distance = float(lpips.distance_arrays(*tensors)[0])
        perceptual.append(
            {
                **pair,
                "space": "native",
                "metric": "lpips_alex_v0.1",
                "status": "complete",
                "distance": distance,
                "error": None,
            }
        )
    _write_csv(output / "lpips_distances.csv", perceptual)
    del lpips
    gc.collect()
    torch.cuda.empty_cache()
    numeric = pd.DataFrame(units)
    identity_summary = (
        numeric.loc[numeric.status == "complete"]
        .groupby("identity_id")[
            ["float__psnr_db", "float__ssim_rgb", "png__psnr_db", "png__ssim_rgb"]
        ]
        .mean()
    )
    _write_csv(output / "identity_summary.csv", identity_summary.reset_index())
    embedding_frame = pd.DataFrame(embedding_rows)
    summary = {
        "status": "complete" if not failures and len(good) == 56 else "complete_with_failures",
        "config_sha256": config_hash,
        "source_photo_count": 56,
        "reconstructed_count": len(good),
        "failed_unit_count": len(units) - len(good),
        "model_failures": failures,
        "scope": "development codec diagnostic; not additive generation loss attribution",
        "mean_native_float_psnr_db": float(
            numeric.loc[numeric.status == "complete", "float__psnr_db"].mean()
        ),
        "mean_native_float_ssim": float(
            numeric.loc[numeric.status == "complete", "float__ssim_rgb"].mean()
        ),
        "identity_equal_mean_psnr_db": float(identity_summary["float__psnr_db"].mean()),
        "identity_equal_mean_ssim": float(identity_summary["float__ssim_rgb"].mean()),
        "repeatability": [unit["repeatability"] for unit in units if "repeatability" in unit],
        "identity_changes": json.loads(
            embedding_frame.groupby(["evaluator", "comparison"])["identity_cosine"]
            .agg(["mean", "count"])
            .reset_index()
            .to_json(orient="records")
        ),
        "lpips_by_space": json.loads(
            pd.DataFrame(perceptual)
            .groupby("space")["distance"]
            .agg(["mean", "count"])
            .reset_index()
            .to_json(orient="records")
        ),
    }
    dump(output / "summary.json", summary)
    return summary
