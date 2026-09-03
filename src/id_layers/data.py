from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import sha256_bytes, sha256_file

FEI_MEMBER_PATTERN = re.compile(r"^(?P<identity>\d+)(?P<condition>[ab])\.jpg$", re.IGNORECASE)
FEI_ORIGINAL_MEMBER_PATTERN = re.compile(
    r"^(?P<identity>\d+)-(?P<slot>\d{2})\.jpg$", re.IGNORECASE
)

# Frozen before model comparison after visual inspection of the acquisition sequence.
# These are derived protocol labels, not publisher annotations.
FEI_QUERY_SLOTS = frozenset({1, 3, 8, 10, 12, 14})
FEI_SLOT_CONDITIONS = {
    1: "left_profile_extreme",
    2: "left_profile",
    3: "left_three_quarter",
    4: "left_slight",
    5: "near_frontal_left",
    6: "near_frontal_right",
    7: "right_slight",
    8: "right_three_quarter_expression",
    9: "right_profile",
    10: "right_profile_extreme",
    11: "frontal_neutral",
    12: "frontal_smile",
    13: "frontal_scale",
    14: "frontal_low_light",
}


def split_identity_ids(
    identity_ids: list[int], split_counts: dict[str, int], seed: int
) -> dict[int, str]:
    expected = sum(int(value) for value in split_counts.values())
    if len(identity_ids) != expected:
        raise ValueError(f"Expected {expected} identities, received {len(identity_ids)}")
    shuffled = np.asarray(sorted(identity_ids), dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    assignment: dict[int, str] = {}
    offset = 0
    for split, count in split_counts.items():
        for identity in shuffled[offset : offset + int(count)]:
            assignment[int(identity)] = split
        offset += int(count)
    return assignment


def _validate_archives(
    archive_root: Path, locked_dataset: dict[str, Any]
) -> list[tuple[Path, dict[str, Any]]]:
    archives: list[tuple[Path, dict[str, Any]]] = []
    for spec in locked_dataset["archives"]:
        path = archive_root / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/bootstrap_assets.py --accept-research-terms"
            )
        observed = sha256_file(path)
        expected = spec["sha256"].lower()
        if observed.lower() != expected:
            raise ValueError(f"Archive checksum mismatch for {path}: {observed} != {expected}")
        archives.append((path, spec))
    return archives


def prepare_fei_manifest(
    data_config: dict[str, Any], locked_dataset: dict[str, Any], seed: int
) -> pd.DataFrame:
    archive_root = Path(data_config["archive_root"])
    processed_root = Path(data_config["processed_root"])
    manifest_path = Path(data_config["manifest_path"])
    archives = _validate_archives(archive_root, locked_dataset)
    processed_root.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, Any]] = {}
    for archive_path, archive_spec in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_name = Path(member.filename).name
                if member.is_dir():
                    continue
                if member_name != member.filename:
                    raise ValueError(f"Unsafe or nested ZIP member: {member.filename}")
                match = FEI_MEMBER_PATTERN.fullmatch(member_name)
                if not match:
                    raise ValueError(f"Unexpected FEI member: {member_name}")
                if member_name.lower() in records:
                    raise ValueError(f"Duplicate FEI member: {member_name}")

                content = archive.read(member)
                target = processed_root / member_name.lower()
                if target.exists():
                    if sha256_file(target) != sha256_bytes(content):
                        raise ValueError(f"Existing processed image differs: {target}")
                else:
                    target.write_bytes(content)

                identity = int(match.group("identity"))
                condition_code = match.group("condition").lower()
                records[member_name.lower()] = {
                    "sample_id": f"fei_{identity:04d}_{condition_code}",
                    "identity_id": f"fei_{identity:04d}",
                    "identity_index": identity,
                    "condition_code": condition_code,
                    "condition": data_config["condition_labels"][condition_code],
                    "relative_path": target.relative_to(processed_root.parents[1]).as_posix(),
                    "absolute_path": str(target.resolve()),
                    "sha256": sha256_bytes(content),
                    "source_archive": archive_spec["filename"],
                    "source_url": archive_spec["url"],
                    "license_id": locked_dataset["license_id"],
                    "face_detection_attempted": False,
                    "face_detected": None,
                    "alignment_source": "publisher_manual",
                    "alignment_status": "publisher_manually_aligned",
                    "usable_for_model": True,
                }

    expected_images = int(data_config["expected_identities"]) * int(
        data_config["expected_conditions_per_identity"]
    )
    if len(records) != expected_images:
        raise ValueError(f"Expected {expected_images} images, found {len(records)}")

    identity_ids = sorted({int(row["identity_index"]) for row in records.values()})
    expected_identities = int(data_config["expected_identities"])
    if len(identity_ids) != expected_identities:
        raise ValueError(f"Expected {expected_identities} identities, found {len(identity_ids)}")
    assignments = split_identity_ids(identity_ids, data_config["split_identity_counts"], seed)

    frame = pd.DataFrame(records.values())
    frame["split"] = frame["identity_index"].map(assignments)
    frame = frame.sort_values(["identity_index", "condition_code"]).reset_index(drop=True)

    counts = frame.groupby("identity_id").size()
    expected_conditions = int(data_config["expected_conditions_per_identity"])
    if not bool((counts == expected_conditions).all()):
        raise ValueError("Every FEI identity must have the expected number of conditions")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["absolute_path"]).to_csv(manifest_path, index=False)
    return frame


def _preprocessing_fingerprint(
    data_config: dict[str, Any], locked_detector: dict[str, Any]
) -> str:
    preprocessing = {
        key: value
        for key, value in data_config["preprocessing"].items()
        if key
        not in {
            "detector_cache_root",
            "detector_model_path",
            "insightface_checkout_path",
        }
    }
    relevant = {
        "dataset": data_config["name"],
        "expected_identities": int(data_config["expected_identities"]),
        "identity_subset_strategy": data_config.get("identity_subset_strategy"),
        "split_identity_counts": data_config["split_identity_counts"],
        "preprocessing": preprocessing,
        "detector_sha256": locked_detector["detector_sha256"],
        "insightface_code_revision": locked_detector["code_revision"],
    }
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_and_extract_fei_full_sources(
    data_config: dict[str, Any], locked_dataset: dict[str, Any]
) -> pd.DataFrame:
    archive_root = Path(data_config["archive_root"])
    source_root = Path(data_config["source_root"])
    archives = _validate_archives(archive_root, locked_dataset)
    source_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}

    for archive_path, archive_spec in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_name = Path(member.filename).name
                if member.is_dir():
                    continue
                if member_name != member.filename:
                    raise ValueError(f"Unsafe or nested ZIP member: {member.filename}")
                match = FEI_ORIGINAL_MEMBER_PATTERN.fullmatch(member_name)
                if not match:
                    raise ValueError(f"Unexpected FEI original member: {member_name}")
                normalized_name = member_name.lower()
                if normalized_name in records:
                    raise ValueError(f"Duplicate FEI original member: {member_name}")

                content = archive.read(member)
                target = source_root / normalized_name
                content_hash = sha256_bytes(content)
                if target.exists():
                    if sha256_file(target) != content_hash:
                        raise ValueError(f"Existing source image differs: {target}")
                else:
                    target.write_bytes(content)

                identity = int(match.group("identity"))
                slot = int(match.group("slot"))
                if slot not in FEI_SLOT_CONDITIONS:
                    raise ValueError(f"Unexpected FEI acquisition slot: {slot}")
                records[normalized_name] = {
                    "sample_id": f"fei_{identity:04d}_{slot:02d}",
                    "identity_id": f"fei_{identity:04d}",
                    "identity_index": identity,
                    "acquisition_slot": slot,
                    "condition": FEI_SLOT_CONDITIONS[slot],
                    "condition_label_source": "derived_visual_protocol_v1",
                    "protocol_role": "query" if slot in FEI_QUERY_SLOTS else "reference",
                    "source_relative_path": target.relative_to(
                        Path(data_config["archive_root"]).parents[1]
                    ).as_posix(),
                    "source_absolute_path": str(target.resolve()),
                    "source_sha256": content_hash,
                    "source_archive": archive_spec["filename"],
                    "source_url": archive_spec["url"],
                    "license_id": locked_dataset["license_id"],
                }

    expected_total = int(locked_dataset["expected_identities"]) * int(
        locked_dataset["expected_conditions_per_identity"]
    )
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total} FEI original images, found {len(records)}")
    frame = pd.DataFrame(records.values()).sort_values(
        ["identity_index", "acquisition_slot"]
    )
    counts = frame.groupby("identity_id").size()
    expected_conditions = int(locked_dataset["expected_conditions_per_identity"])
    if frame["identity_id"].nunique() != int(locked_dataset["expected_identities"]):
        raise ValueError("FEI original identity count does not match the asset lock")
    if not bool((counts == expected_conditions).all()):
        raise ValueError("Every FEI original identity must contain all 14 acquisition slots")
    return frame.reset_index(drop=True)


def _selected_fei_full_sources(data_config: dict[str, Any], sources: pd.DataFrame) -> pd.DataFrame:
    requested = int(data_config["expected_identities"])
    available_ids = sorted(sources["identity_index"].unique().tolist())
    if requested == len(available_ids):
        selected_ids = available_ids
    elif data_config.get("identity_subset_strategy") == "lowest_ids":
        selected_ids = available_ids[:requested]
    else:
        raise ValueError(
            "A partial FEI run must explicitly set identity_subset_strategy: lowest_ids"
        )
    selected = sources[sources["identity_index"].isin(selected_ids)].copy()
    if selected["identity_id"].nunique() != requested:
        raise ValueError(f"Expected {requested} selected identities")
    return selected.reset_index(drop=True)


def _restore_cached_absolute_paths(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    restored = frame.copy()
    absolute_paths: list[str | None] = []
    for row in restored.to_dict(orient="records"):
        if bool(row["usable_for_model"]):
            relative = row.get("relative_path")
            if not isinstance(relative, str) or not relative:
                raise ValueError("Cached usable sample is missing relative_path")
            target = (data_root / relative).resolve()
            if not target.is_file() or sha256_file(target) != row["sha256"]:
                raise ValueError(f"Cached aligned image is missing or differs: {target}")
            absolute_paths.append(str(target))
        else:
            absolute_paths.append(None)
    restored["absolute_path"] = absolute_paths
    return restored


def _validate_cached_fei_full_manifest(
    cached: pd.DataFrame,
    expected_sources: pd.DataFrame,
    preprocessing_fingerprint: str,
) -> pd.DataFrame:
    comparison_fields = [
        "split",
        "protocol_role",
        "condition",
        "acquisition_slot",
        "source_sha256",
    ]
    required_columns = {
        "sample_id",
        "preprocessing_fingerprint",
        *comparison_fields,
    }
    missing_columns = sorted(required_columns - set(cached.columns))
    if missing_columns:
        raise ValueError(
            f"Cached FEI-full manifest is missing required columns: {missing_columns}"
        )
    if cached["sample_id"].isna().any() or cached["sample_id"].duplicated().any():
        raise ValueError("Cached FEI-full manifest contains missing or duplicate sample_id values")

    observed_fingerprints = set(cached["preprocessing_fingerprint"].astype(str))
    if observed_fingerprints != {preprocessing_fingerprint}:
        raise ValueError(
            "Cached FEI-full preprocessing fingerprint differs from the resolved protocol"
        )

    observed = cached.set_index("sample_id", drop=False)
    expected = expected_sources.set_index("sample_id", drop=False)
    missing_samples = sorted(set(expected.index) - set(observed.index))
    unexpected_samples = sorted(set(observed.index) - set(expected.index))
    if missing_samples or unexpected_samples:
        raise ValueError(
            "Cached FEI-full sample set differs from source data: "
            f"missing={missing_samples[:5]}, unexpected={unexpected_samples[:5]}"
        )

    observed = observed.loc[expected.index]
    for field in comparison_fields:
        if field == "acquisition_slot":
            try:
                observed_numeric = pd.to_numeric(observed[field], errors="raise").to_numpy(
                    dtype=np.float64
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Cached FEI-full acquisition_slot contains a non-integer value"
                ) from error
            if not bool(
                np.isfinite(observed_numeric).all()
                and np.equal(observed_numeric, np.floor(observed_numeric)).all()
            ):
                raise ValueError(
                    "Cached FEI-full acquisition_slot contains a non-integer value"
                )
            observed_values = observed_numeric.astype(np.int64)
            expected_values = expected[field].to_numpy(dtype=np.int64)
        else:
            observed_values = observed[field].astype(str).to_numpy()
            expected_values = expected[field].astype(str).to_numpy()
        mismatch = observed_values != expected_values
        if bool(np.any(mismatch)):
            sample_ids = expected.index.to_numpy()[mismatch][:5].tolist()
            raise ValueError(
                f"Cached FEI-full {field} differs from the resolved protocol for "
                f"sample(s): {sample_ids}"
            )

    return observed.reset_index(drop=True)


def prepare_fei_full_manifest(
    data_config: dict[str, Any],
    locked_dataset: dict[str, Any],
    locked_detector: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    from .preprocessing import InsightFaceFivePointAligner

    sources = _selected_fei_full_sources(
        data_config, _validate_and_extract_fei_full_sources(data_config, locked_dataset)
    )
    assignments = split_identity_ids(
        sorted(sources["identity_index"].unique().tolist()),
        data_config["split_identity_counts"],
        seed,
    )
    sources["split"] = sources["identity_index"].map(assignments)
    manifest_path = Path(data_config["manifest_path"])
    fingerprint = _preprocessing_fingerprint(data_config, locked_detector)
    data_root = Path(data_config["archive_root"]).parents[1]

    if manifest_path.is_file() and bool(data_config.get("reuse_preprocessing", True)):
        cached = pd.read_csv(manifest_path)
        cached = _validate_cached_fei_full_manifest(cached, sources, fingerprint)
        return _restore_cached_absolute_paths(cached, data_root)

    aligner = InsightFaceFivePointAligner(data_config["preprocessing"], locked_detector)
    processed_root = Path(data_config["processed_root"])
    records: list[dict[str, Any]] = []
    for source in sources.to_dict(orient="records"):
        target = processed_root / f"{source['sample_id']}.png"
        alignment = aligner.align(Path(source["source_absolute_path"]), target)
        if bool(alignment["usable_for_model"]):
            alignment["relative_path"] = target.relative_to(data_root).as_posix()
            absolute_path: str | None = str(target.resolve())
        else:
            absolute_path = None
        records.append(
            {
                **source,
                **alignment,
                "absolute_path": absolute_path,
                "preprocessing_fingerprint": fingerprint,
                "detector_name": data_config["preprocessing"]["detector_model_name"],
                "detector_sha256": locked_detector["detector_sha256"],
                "insightface_code_revision": locked_detector["code_revision"],
                "detector_runtime_providers_json": json.dumps(aligner.runtime_providers),
            }
        )

    frame = pd.DataFrame(records).sort_values(
        ["identity_index", "acquisition_slot"]
    ).reset_index(drop=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["absolute_path", "source_absolute_path"]).to_csv(
        manifest_path, index=False
    )
    return frame


def prepare_dataset_manifest(
    data_config: dict[str, Any], asset_lock: dict[str, Any], seed: int
) -> pd.DataFrame:
    asset_name = data_config.get("asset_name", data_config["name"])
    locked_dataset = asset_lock["datasets"][asset_name]
    dataset_type = data_config.get("type", "fei_aligned")
    if dataset_type == "fei_aligned":
        return prepare_fei_manifest(data_config, locked_dataset, seed)
    if dataset_type == "fei_full":
        detector_name = data_config["preprocessing"]["detector_asset_name"]
        return prepare_fei_full_manifest(
            data_config,
            locked_dataset,
            asset_lock["models"][detector_name],
            seed,
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


class FaceManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, preprocessing: dict[str, Any]) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = int(preprocessing["image_size"])
        self.mean = torch.tensor(preprocessing["normalization_mean"], dtype=torch.float32)[
            :, None, None
        ]
        self.std = torch.tensor(preprocessing["normalization_std"], dtype=torch.float32)[
            :, None, None
        ]
        interpolation = preprocessing["interpolation"].lower()
        choices = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST,
        }
        if interpolation not in choices:
            raise ValueError(f"Unsupported interpolation: {interpolation}")
        self.interpolation = choices[interpolation]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        with Image.open(row["absolute_path"]) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), self.interpolation)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return {"image": tensor, "index": index, "sample_id": row["sample_id"]}
