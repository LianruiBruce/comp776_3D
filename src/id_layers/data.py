from __future__ import annotations

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
