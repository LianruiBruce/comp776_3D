"""Read-only, ID-keyed diagnostics for frozen generation experiments.

These measurements describe pixels, model representations and landmark outputs.
None is a human likeness, aesthetic-quality or copying-rate measurement. Optional
model dependencies are imported only when their adapters are constructed; model
files must already exist locally and are never downloaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

PIXEL_PREPROCESSING_VERSION = "rgb_exif_transpose_v1"
LPIPS_PREPROCESSING_VERSION = "rgb_exif_longest256_bilinear_gray128_v1"
FACE_PREPROCESSING_VERSION = "arcface112_template_scaled256_linear_black_v1"
QC_PREPROCESSING_VERSION = "rgb_exif_luma601_longest256_bilinear_interior_laplacian_v1"
ARC_FACE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_hash(value: Any, name: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256 for {name}")
    return value


def resolve_inside(root: Path, relative: str | Path) -> Path:
    """Reject traversal and absolute paths even when the destination happens to exist."""
    relative = Path(relative)
    root = Path(root).resolve()
    if relative.is_absolute():
        raise ValueError(f"Expected relative path, got {relative}")
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"Path escapes source root: {relative}")
    return target


def unique_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Every record requires a nonempty string sample_id")
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        result[sample_id] = dict(row)
    return result


def load_generation_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{number}")
                records.append(value)
    if not records:
        raise ValueError("Generation manifest is empty")
    unique_by_id(records)
    return records


def validate_generation_records(
    records: Sequence[Mapping[str, Any]],
    source_root: Path,
    *,
    expected_count: int | None = None,
    expected_seeds: Sequence[int] = (776, 1776),
    expected_identities: Sequence[str] | None = None,
    expected_conditions: Sequence[str] | None = None,
    expected_prompts: Sequence[str] | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate a complete cell ledger, retaining explicitly failed generation cells.

    The old pilot should be passed explicit 192-cell, 8-ID, 6-condition and
    2-prompt expectations. Failed cells remain in the count; this function does
    not replace them or claim that their absent images were successfully scored.
    """
    by_id = unique_by_id(records)
    if not by_id:
        raise ValueError("Empty generation ledger")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} cells, found {len(records)}")
    seeds = tuple(int(seed) for seed in expected_seeds)
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("This diagnostic requires exactly two distinct fixed seeds")
    required = {
        "identity_id",
        "condition",
        "prompt_id",
        "base_seed",
        "effective_seed",
        "status",
        "latent_sha256",
    }
    cells: set[tuple[str, str, str, int]] = set()
    latents: dict[tuple[str, int], set[tuple[Any, Any]]] = defaultdict(set)
    verified = 0
    failures = []
    for sample_id, row in by_id.items():
        if required - row.keys():
            raise ValueError(f"Missing fields for {sample_id}: {sorted(required - row.keys())}")
        base_seed = int(row["base_seed"])
        if base_seed not in seeds:
            raise ValueError(f"Unexpected base seed for {sample_id}")
        cell = (str(row["identity_id"]), str(row["condition"]), str(row["prompt_id"]), base_seed)
        if cell in cells:
            raise ValueError(f"Duplicate generation cell: {cell}")
        cells.add(cell)
        if row["status"] not in {"complete", "failed"}:
            raise ValueError(f"Unfinished/unknown generation status for {sample_id}")
        # A failed renderer can still have a known latent. A failed latent creation
        # is an explicit missing provenance event, not a fabricated hash.
        if row.get("latent_sha256") is not None:
            latent_hash = _check_hash(row["latent_sha256"], sample_id + " latent")
            latents[(cell[0], base_seed)].add((int(row["effective_seed"]), latent_hash))
        elif row["status"] == "complete":
            raise ValueError(f"Complete cell lacks latent provenance: {sample_id}")
        if row["status"] == "failed":
            if not row.get("error") and not row.get("error_type"):
                raise ValueError(f"Failed cell lacks failure reason: {sample_id}")
            failures.append(sample_id)
            continue
        expected_hash = _check_hash(row.get("output_sha256"), sample_id)
        output = resolve_inside(source_root, str(row["output_relative_path"]))
        if verify_files:
            if file_sha256(output) != expected_hash:
                raise ValueError(f"Output hash mismatch: {sample_id}")
            verified += 1
    bad_latents = [key for key, values in latents.items() if len(values) != 1]
    if bad_latents:
        raise ValueError(f"Mismatched shared initial latents/effective seeds: {bad_latents}")
    ids = (
        list(expected_identities)
        if expected_identities is not None
        else sorted({c[0] for c in cells})
    )
    conditions = (
        list(expected_conditions)
        if expected_conditions is not None
        else sorted({c[1] for c in cells})
    )
    prompts = (
        list(expected_prompts) if expected_prompts is not None else sorted({c[2] for c in cells})
    )
    expected_cells = {
        (identity, condition, prompt, seed)
        for identity in ids
        for condition in conditions
        for prompt in prompts
        for seed in seeds
    }
    if cells != expected_cells:
        raise ValueError(
            f"Factorial ledger mismatch: missing={sorted(expected_cells - cells)[:5]}, "
            f"unexpected={sorted(cells - expected_cells)[:5]}"
        )
    pairs = build_seed_pairs(
        records, expected_seeds=seeds, expected_pair_count=len(expected_cells) // 2
    )
    return {
        "status": "complete",
        "cell_count": len(records),
        "complete_image_count": len(records) - len(failures),
        "failed_cell_count": len(failures),
        "failed_sample_ids": sorted(failures),
        "files_hash_verified": verified,
        "seed_pair_count": len(pairs),
        "identity_count": len(ids),
        "condition_count": len(conditions),
        "prompt_count": len(prompts),
    }


def build_seed_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] = (776, 1776),
    expected_pair_count: int | None = None,
) -> list[dict[str, Any]]:
    """Pair by identity/condition/prompt, never by row position or nearest image."""
    by_id = unique_by_id(records)
    seeds = tuple(sorted(int(value) for value in expected_seeds))
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("Exactly two distinct seeds are required")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in by_id.values():
        groups[(str(row["identity_id"]), str(row["condition"]), str(row["prompt_id"]))].append(row)
    pairs = []
    same_fields = (
        "donor_identity_id",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "guidance_scale",
        "num_inference_steps",
        "start_merge_step",
        "reference_sample_ids",
        "global_reference_sample_ids",
        "patch_reference_sample_ids",
        "global_embedding_identity_id",
        "patch_image_identity_id",
    )
    for key, group in sorted(groups.items()):
        if len(group) != 2 or tuple(sorted(int(row["base_seed"]) for row in group)) != seeds:
            raise ValueError(f"Wrong seed pairing for {key}")
        left, right = sorted(group, key=lambda row: int(row["base_seed"]))
        for field in same_fields:
            if left.get(field) != right.get(field):
                raise ValueError(f"Seed pair changes {field}: {key}")
        if int(left["effective_seed"]) - seeds[0] != int(right["effective_seed"]) - seeds[1]:
            raise ValueError(f"Effective seed offset mismatch: {key}")
        if left.get("latent_sha256") and left.get("latent_sha256") == right.get("latent_sha256"):
            raise ValueError(f"Distinct seeds share identical initial latent: {key}")
        pairs.append(
            {
                "pair_id": "seed__" + "__".join(key),
                "identity_id": key[0],
                "condition": key[1],
                "prompt_id": key[2],
                "lhs_id": left["sample_id"],
                "rhs_id": right["sample_id"],
                "lhs_seed": seeds[0],
                "rhs_seed": seeds[1],
                "lhs_sha256": left.get("output_sha256"),
                "rhs_sha256": right.get("output_sha256"),
                "lhs_latent_sha256": left.get("latent_sha256"),
                "rhs_latent_sha256": right.get("latent_sha256"),
                "generation_pair_status": "complete"
                if all(r["status"] == "complete" for r in group)
                else "generation_failed",
            }
        )
    if expected_pair_count is not None and len(pairs) != expected_pair_count:
        raise ValueError(f"Expected {expected_pair_count} seed pairs, found {len(pairs)}")
    return pairs


def build_reference_pairs(
    records: Sequence[Mapping[str, Any]],
    common_reference_by_identity: Mapping[str, str],
    references_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score common K1 and all distinct actually injected global/patch references.

    Repeated slots are counted, but one file is scored once. The old text-only
    schedule loads reference tensors without injecting them; null route identity
    fields therefore define its actual input sets as empty.
    """
    result = []
    for sample_id, row in sorted(unique_by_id(records).items()):
        identity = str(row["identity_id"])
        if identity not in common_reference_by_identity:
            raise ValueError(f"No common K1 reference for {identity}")
        common = common_reference_by_identity[identity]
        if (
            common not in references_by_id
            or str(references_by_id[common]["identity_id"]) != identity
        ):
            raise ValueError(f"Common reference identity mismatch for {identity}")
        global_refs = (
            list(row.get("global_reference_sample_ids", []))
            if row.get("global_embedding_identity_id") is not None
            else []
        )
        patch_refs = (
            list(row.get("patch_reference_sample_ids", []))
            if row.get("patch_image_identity_id") is not None
            else []
        )
        for route, ids, owner in (
            ("global", global_refs, row.get("global_embedding_identity_id")),
            ("patch", patch_refs, row.get("patch_image_identity_id")),
        ):
            for reference_id in ids:
                if reference_id not in references_by_id:
                    raise ValueError(f"Unknown {route} reference: {reference_id}")
                if str(references_by_id[reference_id]["identity_id"]) != str(owner):
                    raise ValueError(f"{route} reference identity mismatch: {sample_id}")
        gc, pc = Counter(global_refs), Counter(patch_refs)
        for reference_id in sorted({common, *global_refs, *patch_refs}):
            reference = references_by_id[reference_id]
            result.append(
                {
                    "pair_id": f"reference__{sample_id}__{reference_id}",
                    "sample_id": sample_id,
                    "identity_id": identity,
                    "condition": row["condition"],
                    "prompt_id": row["prompt_id"],
                    "base_seed": int(row["base_seed"]),
                    "lhs_id": sample_id,
                    "rhs_id": reference_id,
                    "lhs_sha256": row.get("output_sha256"),
                    "rhs_sha256": reference["source_sha256"],
                    "is_common_k1": reference_id == common,
                    "is_actual_input_reference": reference_id in gc or reference_id in pc,
                    "global_slot_count": gc[reference_id],
                    "patch_slot_count": pc[reference_id],
                    "actual_distinct_reference_count": len(set(global_refs + patch_refs)),
                }
            )
    return result


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"RGB\0{rgb.width}\0{rgb.height}\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def image_qc(path: Path, sample_id: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Technical statistics only; clipping thresholds (2/253) are fixed here.

    Blur is the variance of a 4-neighbour Laplacian on BT.601 luma at a
    256-pixel longest side, excluding the border. No success threshold is used.
    """
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "path": str(path),
        "status": "failed",
        "error": None,
        "file_sha256": None,
        "pixel_sha256": None,
        "width": None,
        "height": None,
        "mean_luma_0_255": None,
        "near_black_fraction": None,
        "near_white_fraction": None,
        "blur_laplacian_variance_longest256": None,
        "preprocessing_version": QC_PREPROCESSING_VERSION,
    }
    try:
        observed = file_sha256(path)
    except OSError as error:
        result["error"] = f"file_read_failed:{type(error).__name__}"
        return result
    result["file_sha256"] = observed
    if expected_sha256 is not None and observed != _check_hash(expected_sha256, sample_id):
        raise ValueError(f"Source hash mismatch: {sample_id}")
    try:
        rgb = load_rgb(path)
        pixels = np.asarray(rgb, dtype=np.float64)
        scale = 256.0 / max(rgb.size)
        resized = rgb.resize(
            (max(3, round(rgb.width * scale)), max(3, round(rgb.height * scale))),
            Image.Resampling.BILINEAR,
        )
        small = np.asarray(resized, dtype=np.float64) @ np.asarray([0.299, 0.587, 0.114])
        laplace = (
            small[:-2, 1:-1]
            + small[2:, 1:-1]
            + small[1:-1, :-2]
            + small[1:-1, 2:]
            - 4 * small[1:-1, 1:-1]
        )
        result.update(
            status="complete",
            pixel_sha256=pixel_sha256(rgb),
            width=rgb.width,
            height=rgb.height,
            mean_luma_0_255=float((pixels @ np.asarray([0.299, 0.587, 0.114])).mean()),
            near_black_fraction=float((pixels.max(axis=2) <= 2).mean()),
            near_white_fraction=float((pixels.min(axis=2) >= 253).mean()),
            blur_laplacian_variance_longest256=float(laplace.var()),
        )
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        result["error"] = f"image_decode_failed:{type(error).__name__}"
    return result


def duplicate_groups(qc_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique_by_id(qc_records)
    result: dict[str, Any] = {
        "record_count": len(qc_records),
        "decode_failure_count": sum(row.get("status") != "complete" for row in qc_records),
    }
    for key in ("file_sha256", "pixel_sha256"):
        groups: dict[str, list[str]] = defaultdict(list)
        for row in qc_records:
            if row.get(key):
                groups[str(row[key])].append(str(row["sample_id"]))
        duplicates = [
            {"sha256": sha, "sample_ids": sorted(ids), "count": len(ids)}
            for sha, ids in sorted(groups.items())
            if len(ids) > 1
        ]
        result[key + "_groups"] = duplicates
        result[key + "_duplicate_pair_count"] = sum(
            group["count"] * (group["count"] - 1) // 2 for group in duplicates
        )
    return result


def prepare_lpips_whole(image: Image.Image, size: int = 256) -> np.ndarray:
    """RGB CHW in [-1,1], fixed square canvas, centered gray128 aspect padding."""
    if size < 64:
        raise ValueError("LPIPS AlexNet canvas must be at least 64 pixels")
    image = image.convert("RGB")
    scale = size / max(image.size)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.ascontiguousarray(
        (np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
    )


def fixed_face_crop(
    image: Image.Image, landmarks_5: Sequence[Sequence[float]], size: int = 256
) -> tuple[Image.Image, np.ndarray]:
    """One similarity transform from the fixed 112-point template, scaled to size.

    This is deliberately defined independently of InsightFace's alternative
    128-based horizontal padding. The same matrix can be applied to a VAE
    reconstruction using ``apply_face_transform`` without redetection.
    """
    import cv2
    from skimage.transform import SimilarityTransform

    points = np.asarray(landmarks_5, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Expected five finite 2-D face landmarks")
    transform = SimilarityTransform()
    if not transform.estimate(points, ARC_FACE_112 * (size / 112.0)):
        raise ValueError("Degenerate face landmarks")
    matrix = transform.params[:2, :]
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
        raise ValueError("Invalid face alignment transform")
    pixels = cv2.warpAffine(
        np.asarray(image.convert("RGB")),
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return Image.fromarray(pixels), matrix


def apply_face_transform(image: Image.Image, matrix: np.ndarray, size: int = 256) -> Image.Image:
    import cv2

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Face transform must be finite 2x3")
    pixels = cv2.warpAffine(
        np.asarray(image.convert("RGB")),
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return Image.fromarray(pixels)


def save_keyed_array_cache(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    *,
    preprocessing_version: str,
) -> None:
    """Create a new non-object NPZ with explicit IDs, hashes and preprocessing.

    Store only successfully evaluated records; retain failure rows separately in
    the complete per-image ledger. Nonfinite arrays are rejected, not masked.
    """
    unique_by_id(records)
    if not records or not arrays or not preprocessing_version:
        raise ValueError("Cache requires nonempty records, arrays and preprocessing version")
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {path}")
    ids = [str(row["sample_id"]) for row in records]
    hashes = [_check_hash(row["source_sha256"], str(row["sample_id"])) for row in records]
    reserved = {"sample_ids", "source_sha256", "preprocessing_version", "schema_version"}
    if reserved & arrays.keys():
        raise ValueError("Array key conflicts with cache metadata")
    payload: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(ids, dtype="U"),
        "source_sha256": np.asarray(hashes, dtype="U64"),
        "preprocessing_version": np.asarray(preprocessing_version),
        "schema_version": np.asarray(1, dtype=np.int64),
    }
    for name, value in arrays.items():
        value = np.asarray(value)
        if (
            value.ndim < 1
            or value.shape[0] != len(records)
            or value.dtype.kind not in "fiu b".replace(" ", "")
        ):
            raise ValueError(f"Invalid row count or dtype for cache array {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite cache array {name}")
        payload[name] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Open an exclusive file: neither silent extension changes nor overwrites.
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **payload)


def load_keyed_array_cache(
    path: Path,
    expected_records: Sequence[Mapping[str, Any]],
    *,
    preprocessing_version: str,
) -> dict[str, np.ndarray]:
    """Return arrays in requested ID order, verifying exact IDs and source hashes."""
    unique_by_id(expected_records)
    with np.load(path, allow_pickle=False) as stored:
        required = {"sample_ids", "source_sha256", "preprocessing_version", "schema_version"}
        if not required.issubset(stored.files):
            raise ValueError(
                "Legacy/unkeyed NPZ cannot be trusted without separate frozen-order verification"
            )
        if (
            int(stored["schema_version"].item()) != 1
            or str(stored["preprocessing_version"].item()) != preprocessing_version
        ):
            raise ValueError("Cache schema/preprocessing version mismatch")
        ids, hashes = stored["sample_ids"].tolist(), stored["source_sha256"].tolist()
        if len(ids) != len(set(ids)) or len(ids) != len(hashes):
            raise ValueError("Duplicate IDs or inconsistent metadata in cache")
        indices = {sample_id: i for i, sample_id in enumerate(ids)}
        if set(indices) != {row["sample_id"] for row in expected_records}:
            raise ValueError("Cache sample-ID set mismatch")
        order = []
        for row in expected_records:
            index = indices[row["sample_id"]]
            if hashes[index] != _check_hash(row["source_sha256"], row["sample_id"]):
                raise ValueError(f"Cache source hash mismatch: {row['sample_id']}")
            order.append(index)
        result = {}
        for name in set(stored.files) - required:
            value = stored[name]
            if value.ndim < 1 or value.shape[0] != len(ids) or not np.isfinite(value).all():
                raise ValueError(f"Invalid cache array: {name}")
            result[name] = value[order].copy()
    return result


class LPIPSAlexAdapter:
    """Official LPIPS Alex v0.1 with explicit local weights and no downloads.

    ``alexnet_weights`` is torchvision's alexnet-owt-7be5be79.pth. The LPIPS
    package's bundled v0.1/alex.pth can be passed as ``lpips_weights``; its hash
    must also be recorded by the caller's asset lock.
    """

    def __init__(
        self,
        *,
        alexnet_weights: Path,
        lpips_weights: Path,
        device: str = "cuda",
        alexnet_sha256: str | None = None,
        lpips_sha256: str | None = None,
    ) -> None:
        import lpips
        import torch

        for path, expected in ((alexnet_weights, alexnet_sha256), (lpips_weights, lpips_sha256)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Local LPIPS asset missing: {path}")
            if expected is not None and file_sha256(path) != _check_hash(expected, str(path)):
                raise ValueError(f"LPIPS asset hash mismatch: {path}")
        self.torch = torch
        self.device = torch.device(device)
        # pnet_rand prevents torchvision's implicit network request. We then
        # load every feature layer from the pinned official AlexNet checkpoint.
        self.model = lpips.LPIPS(
            net="alex", version="0.1", pnet_rand=True, model_path=str(lpips_weights), verbose=False
        ).eval()
        state = torch.load(alexnet_weights, map_location="cpu", weights_only=True)
        mapping = {
            "slice1.0": "features.0",
            "slice2.3": "features.3",
            "slice3.6": "features.6",
            "slice4.8": "features.8",
            "slice5.10": "features.10",
        }
        feature_state = self.model.net.state_dict()
        for key in list(feature_state):
            module, parameter = key.rsplit(".", 1)
            if module not in mapping:
                raise ValueError(f"Unexpected LPIPS AlexNet feature layer: {key}")
            feature_state[key] = state[f"{mapping[module]}.{parameter}"]
        self.model.net.load_state_dict(feature_state, strict=True)
        self.model.to(self.device)
        self.model.requires_grad_(False)

    def distance_arrays(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """BCHW arrays in [-1,1]; finite scalar distance per pair, identical=0."""
        left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
        if left.ndim == 3:
            left, right = left[None], right[None]
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError("LPIPS arrays require matching Bx3xHxW dimensions")
        if (
            not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or max(float(abs(left).max()), float(abs(right).max())) > 1.00001
        ):
            raise ValueError("LPIPS arrays must be finite RGB in [-1,1]")
        with self.torch.inference_mode():
            output = self.model(
                self.torch.from_numpy(left).to(self.device),
                self.torch.from_numpy(right).to(self.device),
                normalize=False,
            )
            values = output.reshape(-1).cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("LPIPS produced nonfinite distance")
        return values

    def score_pairs(
        self,
        pairs: Sequence[Mapping[str, Any]],
        image_paths: Mapping[str, Path],
        *,
        space: str = "whole",
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """For face space pass already aligned 256px crops, preserving fixed geometry."""
        if space not in {"whole", "face"} or batch_size < 1:
            raise ValueError("Invalid LPIPS space or batch size")
        if len({pair["pair_id"] for pair in pairs}) != len(pairs):
            raise ValueError("Duplicate LPIPS pair_id")
        tensors: dict[str, np.ndarray] = {}
        errors: dict[str, str] = {}
        for sample_id in sorted({str(pair[key]) for pair in pairs for key in ("lhs_id", "rhs_id")}):
            try:
                if sample_id not in image_paths:
                    raise FileNotFoundError("Image/aligned crop unavailable")
                rgb = load_rgb(image_paths[sample_id])
                if space == "whole":
                    tensors[sample_id] = prepare_lpips_whole(rgb)
                else:
                    if rgb.size != (256, 256):
                        raise ValueError("Face LPIPS requires the fixed 256x256 aligned crop")
                    tensors[sample_id] = np.ascontiguousarray(
                        (np.asarray(rgb, dtype=np.float32) / 127.5 - 1).transpose(2, 0, 1)
                    )
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                errors[sample_id] = f"{type(error).__name__}:{error}"
        results = []
        valid: list[tuple[int, Mapping[str, Any]]] = []
        for pair in pairs:
            failures = {
                key: errors[str(pair[key])]
                for key in ("lhs_id", "rhs_id")
                if str(pair[key]) in errors
            }
            row = {
                **pair,
                "space": space,
                "metric": "lpips_alex_v0.1",
                "distance": None,
                "status": "failed" if failures else "pending",
                "error": failures or None,
            }
            results.append(row)
            if not failures:
                valid.append((len(results) - 1, pair))
        for start in range(0, len(valid), batch_size):
            batch = valid[start : start + batch_size]
            values = self.distance_arrays(
                np.stack([tensors[str(pair["lhs_id"])] for _, pair in batch]),
                np.stack([tensors[str(pair["rhs_id"])] for _, pair in batch]),
            )
            for (index, _), value in zip(batch, values, strict=True):
                results[index].update(status="complete", distance=float(value))
        return results


def aggregate_reference_distances(
    scored_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_pairs:
        groups[(str(row["sample_id"]), str(row["space"]))].append(row)
    results = []
    for (sample_id, space), rows in sorted(groups.items()):
        common = [row for row in rows if row["is_common_k1"]]
        actual = [row for row in rows if row["is_actual_input_reference"]]
        if len(common) != 1:
            raise ValueError(f"Expected one common reference for {sample_id}")
        good = [float(row["distance"]) for row in actual if row["status"] == "complete"]
        complete_actual = bool(actual) and len(good) == len(actual)
        results.append(
            {
                "sample_id": sample_id,
                "identity_id": rows[0]["identity_id"],
                "condition": rows[0]["condition"],
                "prompt_id": rows[0]["prompt_id"],
                "base_seed": rows[0]["base_seed"],
                "space": space,
                "common_k1_distance": common[0]["distance"]
                if common[0]["status"] == "complete"
                else None,
                "actual_reference_count": len(actual),
                "valid_actual_reference_count": len(good),
                "actual_reference_mean_distance": float(np.mean(good)) if complete_actual else None,
                "actual_reference_min_distance": float(min(good)) if complete_actual else None,
                "actual_reference_status": "not_applicable_no_injection"
                if not actual
                else "complete"
                if complete_actual
                else "failed",
            }
        )
    return results


def seed_embedding_distances(
    pairs: Sequence[Mapping[str, Any]],
    embeddings_by_id: Mapping[str, np.ndarray],
    *,
    evaluator: str,
) -> list[dict[str, Any]]:
    results = []
    for pair in pairs:
        row = {
            **pair,
            "evaluator": evaluator,
            "embedding_cosine_distance": None,
            "status": "failed",
            "error": None,
        }
        try:
            left, right = [
                np.asarray(embeddings_by_id[str(pair[key])], dtype=np.float64)
                for key in ("lhs_id", "rhs_id")
            ]
            if (
                left.ndim != 1
                or left.shape != right.shape
                or not np.isfinite(left).all()
                or not np.isfinite(right).all()
            ):
                raise ValueError("Invalid embedding shape/value")
            norms = float(np.linalg.norm(left) * np.linalg.norm(right))
            if norms <= 1e-12:
                raise ValueError("Zero-norm identity embedding")
            row.update(
                status="complete",
                embedding_cosine_distance=float(1 - np.clip(left @ right / norms, -1, 1)),
            )
        except (KeyError, ValueError) as error:
            row["error"] = f"{type(error).__name__}:{error}"
        results.append(row)
    return results


def summarize_landmarker_result(result: Any) -> dict[str, Any]:
    """Reject no/multiple faces for primary continuous expression/pose summaries."""
    count = len(result.face_landmarks)
    row: dict[str, Any] = {
        "status": "failed",
        "error": None,
        "detected_face_count": count,
        "mouth_smile_left": None,
        "mouth_smile_right": None,
        "mouth_smile_mean": None,
        "jaw_open": None,
        "pose_matrix_4x4": None,
        "pose_extrinsic_xyz_degrees": None,
        "pose_convention": (
            "closest_proper_rotation; extrinsic xyz; MediaPipe canonical camera axes; "
            "not calibrated physical pose"
        ),
    }
    if count != 1:
        row["error"] = "no_face_detected" if count == 0 else "multiple_faces_detected"
        return row
    if len(result.face_blendshapes) != 1 or len(result.facial_transformation_matrixes) != 1:
        row["error"] = "blendshapes_or_matrix_missing"
        return row
    coefficients = {item.category_name: float(item.score) for item in result.face_blendshapes[0]}
    keys = ("mouthSmileLeft", "mouthSmileRight", "jawOpen")
    if any(key not in coefficients or not math.isfinite(coefficients[key]) for key in keys):
        row["error"] = "invalid_blendshape_coefficients"
        return row
    matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        row["error"] = "invalid_transformation_matrix"
        return row
    u, singular_values, vt = np.linalg.svd(matrix[:3, :3])
    if singular_values.min() <= 1e-8:
        row["error"] = "singular_transformation_matrix"
        return row
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    y = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
    if abs(math.cos(y)) > 1e-7:
        x, z = (
            math.atan2(rotation[2, 1], rotation[2, 2]),
            math.atan2(rotation[1, 0], rotation[0, 0]),
        )
    else:
        x, z = math.atan2(-rotation[1, 2], rotation[1, 1]), 0.0
    row.update(
        status="complete",
        mouth_smile_left=coefficients[keys[0]],
        mouth_smile_right=coefficients[keys[1]],
        mouth_smile_mean=(coefficients[keys[0]] + coefficients[keys[1]]) / 2,
        jaw_open=coefficients[keys[2]],
        pose_matrix_4x4=matrix.tolist(),
        pose_extrinsic_xyz_degrees=np.degrees([x, y, z]).tolist(),
    )
    return row


class MediaPipeLandmarkerAdapter:
    """IMAGE-mode CPU landmarking with up to two faces to detect ambiguity."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_sha256: str,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if file_sha256(model_path) != _check_hash(model_sha256, str(model_path)):
            raise ValueError("MediaPipe task asset hash mismatch")
        self.mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_presence_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )

    def analyze(
        self, path: Path, sample_id: str, *, expected_sha256: str | None = None
    ) -> dict[str, Any]:
        if expected_sha256 is not None and file_sha256(path) != _check_hash(
            expected_sha256, sample_id
        ):
            raise ValueError(f"Landmarker source hash mismatch: {sample_id}")
        try:
            pixels = np.ascontiguousarray(np.asarray(load_rgb(path)))
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=pixels)
            result = summarize_landmarker_result(self.landmarker.detect(image))
        except (OSError, ValueError, RuntimeError, Image.DecompressionBombError) as error:
            result = {
                "status": "failed",
                "error": f"{type(error).__name__}:{error}",
                "detected_face_count": None,
                "mouth_smile_left": None,
                "mouth_smile_right": None,
                "mouth_smile_mean": None,
                "jaw_open": None,
                "pose_matrix_4x4": None,
                "pose_extrinsic_xyz_degrees": None,
            }
        return {
            "sample_id": sample_id,
            "preprocessing_version": PIXEL_PREPROCESSING_VERSION,
            **result,
        }

    def close(self) -> None:
        self.landmarker.close()
