from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

PUBLIC_SCHEMA_VERSION = "identity-layers-human-eval/v1"
RESPONSE_SCHEMA_VERSION = "identity-layers-human-response/v1"

_IMAGE_SUFFIXES = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
_PAIR_SPECS = (
    # P0 is the originally requested direct K=1 versus diverse K=4 comparison.
    ("P0", "k1", "k4_diverse"),
    # P1-P4 are the four frozen contrasts in generation_identity_pilot.yaml.
    ("P1", "k4_diverse", "k4_repeat"),
    ("P2", "k4_repeat", "k1"),
    ("P3", "global_target_patch_donor", "global_donor_patch_target"),
    ("P4", "k1", "text_only"),
)
_CONFLICT_VARIANTS = {
    "global_patch_conflict",
    "global_target_patch_donor",
    "global_donor_patch_target",
}
_FROZEN_FOIL_OFFSETS = (2, 3)
_PREFERRED_GALLERY_CONDITIONS = {
    "frontal_smile",
    "left_three_quarter",
    "right_three_quarter_expression",
}
_VARIANT_ALIASES = {
    "k1": "k1",
    "k_1": "k1",
    "1": "k1",
    "single": "k1",
    "single_reference": "k1",
    "one_reference": "k1",
    "k1_full": "k1",
    "k4_diverse": "k4_diverse",
    "k_4_diverse": "k4_diverse",
    "4_diverse": "k4_diverse",
    "diverse_k4": "k4_diverse",
    "four_diverse": "k4_diverse",
    "k4_diverse_full": "k4_diverse",
    "k4_repeat": "k4_repeat",
    "k_4_repeat": "k4_repeat",
    "4_repeat": "k4_repeat",
    "repeat_k4": "k4_repeat",
    "four_repeat": "k4_repeat",
    "k4_repeat_full": "k4_repeat",
    "global_patch_conflict": "global_patch_conflict",
    "global_patch_swap": "global_patch_conflict",
    "global_target_patch_donor": "global_target_patch_donor",
    "global_donor_patch_target": "global_donor_patch_target",
    "target_global_donor_patch": "global_patch_conflict",
    "conflict": "global_patch_conflict",
    "text_only": "text_only",
}


class HumanEvalError(ValueError):
    """Raised when an evaluation cannot be built without ambiguity or leakage."""


@dataclass(frozen=True)
class HumanEvalBuild:
    html_path: Path
    blinded_json_path: Path
    media_dir: Path
    public_study: dict[str, Any]
    private_answer_key: dict[str, Any]

    @property
    def trial_count(self) -> int:
        return len(self.public_study["trials"])


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _required_text(row: Mapping[str, Any], names: Sequence[str], context: str) -> str:
    value = _first_present(row, names)
    if value is None:
        raise HumanEvalError(f"Missing {names[0]} in {context}")
    return str(value).strip()


def _optional_text(row: Mapping[str, Any], names: Sequence[str], default: str) -> str:
    value = _first_present(row, names)
    return default if value is None else str(value).strip()


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise HumanEvalError(f"Expected a boolean value, got {value!r}")


def _as_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise HumanEvalError(f"{context} must be an integer, not a boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise HumanEvalError(f"{context} must be an integer, got {value!r}") from error
    if isinstance(value, float) and not value.is_integer():
        raise HumanEvalError(f"{context} must be an integer, got {value!r}")
    return parsed


def _read_records(path: Path, preferred_keys: Sequence[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise HumanEvalError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = None
            for key in preferred_keys:
                candidate = value.get(key)
                if isinstance(candidate, list):
                    rows = candidate
                    break
            if rows is None:
                list_values = [item for item in value.values() if isinstance(item, list)]
                if len(list_values) == 1:
                    rows = list_values[0]
                else:
                    raise HumanEvalError(
                        f"Could not find one of {list(preferred_keys)} as a list in {path}"
                    )
        else:
            raise HumanEvalError(f"Expected a JSON array or object in {path}")
        if not all(isinstance(row, dict) for row in rows):
            raise HumanEvalError(f"Every record in {path} must be a JSON object")
        return [dict(row) for row in rows]
    raise HumanEvalError(f"Unsupported table format for {path}; use CSV, JSON, or JSONL")


def _normal_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _normalize_variant(value: Any, context: str) -> str:
    token = _normal_token(value)
    try:
        return _VARIANT_ALIASES[token]
    except KeyError as error:
        accepted = sorted(set(_VARIANT_ALIASES.values()))
        raise HumanEvalError(
            f"Unsupported generation variant {value!r} in {context}; canonical values: {accepted}"
        ) from error


def _candidate_roots(source_table: Path, asset_root: Path | None) -> Iterable[Path]:
    seen: set[Path] = set()
    roots = []
    if asset_root is not None:
        roots.append(asset_root)
    roots.extend(source_table.resolve().parents)
    roots.append(Path.cwd())
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _resolve_image_path(value: Any, source_table: Path, asset_root: Path | None) -> Path:
    text = str(value).strip()
    lowered = text.lower()
    if lowered.startswith("data:"):
        raise HumanEvalError("Inline data/base64 image URIs are forbidden")
    if re.match(r"^[a-z][a-z0-9+.-]*://", lowered):
        raise HumanEvalError(f"Only local image files are supported, got URI {text!r}")
    raw = Path(text)
    candidates = (
        [raw]
        if raw.is_absolute()
        else [root / raw for root in _candidate_roots(source_table, asset_root)]
    )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved.suffix.lower() not in _IMAGE_SUFFIXES:
                raise HumanEvalError(f"Unsupported image extension: {resolved}")
            return resolved
    raise FileNotFoundError(f"Image referenced by {source_table} does not exist: {text}")


def _parse_identity_list(value: Any) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    else:
        text = str(value).strip()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise HumanEvalError("foil_identity_ids JSON must be an array")
            items = parsed
        else:
            items = re.split(r"[,;|]", text)
    result = [str(item).strip() for item in items if str(item).strip()]
    if len(result) != len(set(result)):
        raise HumanEvalError("foil_identity_ids must be unique")
    return result


def _stable_digest(*parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _opaque_id(prefix: str, seed: int, *parts: Any, length: int = 16) -> str:
    return f"{prefix}_{_stable_digest(seed, *parts)[:length]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _stage_image(source: Path, media_dir: Path, html_parent: Path) -> tuple[str, str]:
    digest = _sha256_file(source)
    filename = f"img_{digest[:24]}{source.suffix.lower()}"
    destination = media_dir / filename
    media_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != digest:
            raise RuntimeError(f"Opaque media collision at {destination}")
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    relative = Path(os.path.relpath(destination, html_parent)).as_posix()
    return quote(relative, safe="/"), digest


def _canonical_references(
    manifest_path: Path,
    asset_root: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    rows = _read_records(manifest_path, ("references", "samples", "manifest", "records"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        context = f"reference manifest row {index + 1}"
        usable = _first_present(row, ("usable_for_human_eval", "usable_for_model", "usable"))
        if usable is not None and not _as_bool(usable):
            continue
        identity = _required_text(row, ("identity_id", "subject_id", "person_id"), context)
        image_value = _first_present(
            row,
            (
                "human_eval_image_path",
                "relative_path",
                "image_path",
                "aligned_path",
                "processed_path",
                "source_relative_path",
                "path",
            ),
        )
        if image_value is None:
            raise HumanEvalError(f"Missing image path in {context}")
        explicit = _first_present(
            row,
            ("human_eval_reference", "is_human_eval_reference", "use_for_human_eval"),
        )
        grouped.setdefault(identity, []).append(
            {
                "identity_id": identity,
                "image_path": _resolve_image_path(image_value, manifest_path, asset_root),
                "sample_id": _optional_text(row, ("sample_id", "reference_id"), str(index)),
                "role": _normal_token(_optional_text(row, ("protocol_role", "role"), "")),
                "condition": _normal_token(
                    _optional_text(row, ("condition", "acquisition_condition"), "")
                ),
                "explicit": None if explicit is None else _as_bool(explicit),
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    for identity, candidates in grouped.items():
        explicitly_selected = [row for row in candidates if row["explicit"] is True]
        query_rows = [
            row
            for row in candidates
            if row["role"] in {"evaluation_reference", "human_eval_reference", "query"}
        ]
        frozen_gallery_rows = [
            row for row in query_rows if row["condition"] in _PREFERRED_GALLERY_CONDITIONS
        ]
        usable = explicitly_selected or frozen_gallery_rows or query_rows or candidates
        selected[identity] = sorted(
            usable,
            key=lambda row: (str(row["sample_id"]), row["image_path"].as_posix()),
        )
    if not selected:
        raise HumanEvalError(f"No usable human-evaluation references in {manifest_path}")
    return selected


def _canonical_results(results_path: Path, asset_root: Path | None) -> list[dict[str, Any]]:
    rows = _read_records(results_path, ("results", "generations", "outputs", "records"))
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context = f"generation results row {index + 1}"
        variant_value = _first_present(
            row,
            (
                "human_eval_variant",
                "variant",
                "reference_policy",
                "generation_condition",
                "condition",
            ),
        )
        if variant_value is None:
            raise HumanEvalError(f"Missing variant/reference_policy in {context}")
        variant = _normalize_variant(variant_value, context)
        usable = _first_present(row, ("usable_for_human_eval", "usable"))
        if usable is not None and not _as_bool(usable):
            continue
        status = _first_present(row, ("status", "generation_status"))
        if status is not None and _normal_token(status) not in {
            "complete",
            "completed",
            "ok",
            "success",
        }:
            raise HumanEvalError(
                f"Relevant generation is not complete in {context}: status={status!r}; "
                "failed rows must not be silently dropped"
            )
        generation_succeeded = _first_present(row, ("generation_succeeded",))
        if generation_succeeded is not None and not _as_bool(generation_succeeded):
            raise HumanEvalError(
                f"Relevant generation failed in {context}; failed rows must not be silently dropped"
            )
        image_value = _first_present(
            row,
            (
                "image_path",
                "output_path",
                "output_relative_path",
                "generated_image_path",
                "relative_path",
                "path",
            ),
        )
        if image_value is None:
            raise HumanEvalError(f"Missing generated image path in {context}")
        seed_value = _first_present(
            row,
            (
                "generation_seed",
                "effective_seed",
                "base_seed",
                "latent_seed",
                "seed",
                "sample_seed",
            ),
        )
        if seed_value is None:
            raise HumanEvalError(f"Missing generation_seed in {context}")
        global_identity = _first_present(
            row,
            (
                "global_embedding_identity_id",
                "global_source_identity_id",
                "global_identity_id",
                "target_identity_id",
                "identity_id",
                "subject_id",
            ),
        )
        patch_identity = _first_present(
            row,
            (
                "patch_image_identity_id",
                "patch_source_identity_id",
                "patch_identity_id",
                "donor_identity_id",
                "appearance_identity_id",
            ),
        )
        identity = _first_present(row, ("identity_id", "target_identity_id", "subject_id"))
        donor_identity = _first_present(row, ("donor_identity_id", "paired_identity_id"))
        if identity is None:
            raise HumanEvalError(f"Missing identity_id in {context}")
        if variant in _CONFLICT_VARIANTS and (
            global_identity is None or patch_identity is None
        ):
            raise HumanEvalError(
                f"Conflict row {index + 1} requires global_identity_id and patch_identity_id"
            )
        if variant in _CONFLICT_VARIANTS:
            identity = identity if identity is not None else global_identity
            donor_identity = donor_identity if donor_identity is not None else patch_identity
            experimental_pair = {str(identity).strip(), str(donor_identity).strip()}
            stream_pair = {str(global_identity).strip(), str(patch_identity).strip()}
            if len(experimental_pair) != 2 or experimental_pair != stream_pair:
                raise HumanEvalError(
                    f"Conflict row {index + 1} must map one target and one donor identity "
                    "exactly onto the global and patch streams"
                )
        record = {
            "row_index": index,
            "result_id": _optional_text(
                row, ("result_id", "generation_id", "sample_id"), f"row-{index}"
            ),
            "method": _optional_text(
                row, ("method", "generator", "model", "conditioner"), "unspecified_method"
            ),
            "identity_id": None if identity is None else str(identity).strip(),
            "donor_identity_id": (
                None if donor_identity is None else str(donor_identity).strip()
            ),
            "global_identity_id": (
                None if global_identity is None else str(global_identity).strip()
            ),
            "patch_identity_id": None if patch_identity is None else str(patch_identity).strip(),
            "prompt_id": _optional_text(row, ("prompt_id", "prompt_key", "prompt"), "default"),
            "generation_seed": _as_int(seed_value, f"generation_seed in {context}"),
            "output_index": _as_int(
                _first_present(row, ("output_index", "image_index", "replicate_index")) or 0,
                f"output_index in {context}",
            ),
            "case_id": _optional_text(
                row, ("comparison_id", "case_id", "group_id", "pair_id"), ""
            ),
            "variant": variant,
            "image_path": _resolve_image_path(image_value, results_path, asset_root),
            "foil_identity_ids": _parse_identity_list(
                _first_present(row, ("foil_identity_ids", "foil_ids", "distractor_identity_ids"))
            ),
        }
        records.append(record)
    if not records:
        raise HumanEvalError(f"No usable generation results in {results_path}")
    return records


def _pair_group_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    if record["case_id"]:
        case = ("explicit", record["case_id"])
    else:
        case = (
            "derived",
            record["identity_id"],
            record["prompt_id"],
            record["generation_seed"],
            record["output_index"],
        )
    return (record["method"], *case)


def _choose_references(
    references: dict[str, list[dict[str, Any]]],
    identity: str,
    count: int,
    seed: int,
    key: Any,
) -> list[Path]:
    try:
        candidates = list(references[identity])
    except KeyError as error:
        raise HumanEvalError(f"No evaluation reference for identity {identity!r}") from error
    if len(candidates) < count:
        raise HumanEvalError(
            f"Identity {identity!r} has {len(candidates)} usable reference images; "
            f"{count} are required"
        )
    rng = random.Random(int(_stable_digest(seed, "reference", key)[:16], 16))
    rng.shuffle(candidates)
    return [row["image_path"] for row in candidates[:count]]


def _build_pair_candidates(
    records: Sequence[dict[str, Any]],
    references: dict[str, list[dict[str, Any]]],
    references_per_identity: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["variant"] == "global_patch_conflict":
            continue
        key = _pair_group_key(record)
        by_variant = grouped.setdefault(key, {})
        variant = record["variant"]
        if variant in by_variant:
            raise HumanEvalError(
                "Ambiguous duplicate generation for group "
                f"{key!r} and variant {variant!r}; provide output_index or case_id"
            )
        by_variant[variant] = record

    candidates: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: repr(value)):
        by_variant = grouped[key]
        for block_code, first_variant, second_variant in _PAIR_SPECS:
            if first_variant not in by_variant or second_variant not in by_variant:
                continue
            first = by_variant[first_variant]
            second = by_variant[second_variant]
            if first["identity_id"] != second["identity_id"]:
                raise HumanEvalError(f"Pair group {key!r} mixes target identities")
            private_key = (block_code, key)
            candidates.append(
                {
                    "block_code": block_code,
                    "private_key": private_key,
                    "method": first["method"],
                    "identity_id": first["identity_id"],
                    "prompt_id": first["prompt_id"],
                    "generation_seed": first["generation_seed"],
                    "reference_paths": _choose_references(
                        references,
                        first["identity_id"],
                        references_per_identity,
                        seed,
                        private_key,
                    ),
                    "conditions": [
                        {
                            "variant": first_variant,
                            "result_id": first["result_id"],
                            "image_path": first["image_path"],
                        },
                        {
                            "variant": second_variant,
                            "result_id": second["result_id"],
                            "image_path": second["image_path"],
                        },
                    ],
                }
            )
    return candidates


def _choose_foils(
    record: Mapping[str, Any],
    references: dict[str, list[dict[str, Any]]],
    identity_pool: Sequence[str],
) -> list[str]:
    target = record["identity_id"]
    donor = record["donor_identity_id"]
    if donor is None:
        raise HumanEvalError(
            f"Generation {record['result_id']!r} has no donor_identity_id for 4-AFC"
        )
    if target == donor:
        raise HumanEvalError("Target and donor identity IDs must be different")
    provided = record["foil_identity_ids"]
    if provided:
        if len(provided) != 2:
            raise HumanEvalError("A conflict trial must provide exactly two foil_identity_ids")
        foils = list(provided)
    else:
        pool = list(identity_pool)
        if target not in pool:
            raise HumanEvalError(f"Target identity {target!r} is outside the frozen identity pool")
        if len(pool) < 4:
            raise HumanEvalError("A 4-AFC conflict trial requires at least two foil identities")
        target_index = pool.index(target)
        foils = [pool[(target_index + offset) % len(pool)] for offset in _FROZEN_FOIL_OFFSETS]
    if target in foils or donor in foils or len(set(foils)) != 2:
        raise HumanEvalError(
            "Frozen foil offsets produce target/donor overlap; provide two explicit "
            "foil_identity_ids or use the preregistered donor mapping"
        )
    missing = [identity for identity in [target, donor, *foils] if identity not in references]
    if missing:
        raise HumanEvalError(f"Missing evaluation references for 4-AFC identities: {missing}")
    return foils


def _build_identity_candidates(
    records: Sequence[dict[str, Any]],
    references: dict[str, list[dict[str, Any]]],
    references_per_identity: int,
    seed: int,
) -> list[dict[str, Any]]:
    study_identities = {
        str(identity)
        for record in records
        for identity in (record["identity_id"], record["donor_identity_id"])
        if identity is not None
    }
    if len(study_identities) < 4:
        study_identities.update(references)
    identity_pool = sorted(study_identities)
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for record in records:
        private_key = (
            "I1",
            record["method"],
            record["result_id"],
            record["variant"],
            record["prompt_id"],
            record["generation_seed"],
            record["output_index"],
        )
        if private_key in seen_keys:
            raise HumanEvalError(
                f"Generation result is not uniquely identifiable for 4-AFC: {private_key!r}"
            )
        seen_keys.add(private_key)
        foils = _choose_foils(record, references, identity_pool)
        identities_and_roles = [
            (record["identity_id"], "target_identity"),
            (record["donor_identity_id"], "donor_identity"),
            (foils[0], "foil_1"),
            (foils[1], "foil_2"),
        ]
        options = []
        for identity, role in identities_and_roles:
            image_paths = _choose_references(
                references,
                identity,
                references_per_identity,
                seed,
                (private_key, role),
            )
            options.append(
                {"identity_id": identity, "role": role, "image_paths": image_paths}
            )
        candidates.append(
            {
                "block_code": "I1",
                "private_key": private_key,
                "method": record["method"],
                "variant": record["variant"],
                "prompt_id": record["prompt_id"],
                "generation_seed": record["generation_seed"],
                "result_id": record["result_id"],
                "target_identity_id": record["identity_id"],
                "donor_identity_id": record["donor_identity_id"],
                "global_identity_id": record["global_identity_id"],
                "patch_identity_id": record["patch_identity_id"],
                "query_path": record["image_path"],
                "options": options,
            }
        )
    return candidates


def _balanced_orientations(candidates: Sequence[dict[str, Any]], seed: int) -> dict[str, bool]:
    assignments: dict[str, bool] = {}
    for block_code, _, _ in _PAIR_SPECS:
        block = sorted(
            (candidate for candidate in candidates if candidate["block_code"] == block_code),
            key=lambda candidate: repr(candidate["private_key"]),
        )
        flags = [bool(index % 2) for index in range(len(block))]
        rng = random.Random(int(_stable_digest(seed, "orientation", block_code)[:16], 16))
        rng.shuffle(flags)
        if flags and rng.random() < 0.5:
            flags = [not flag for flag in flags]
        for candidate, swap in zip(block, flags, strict=True):
            assignments[repr(candidate["private_key"])] = swap
    return assignments


def _public_and_private_trials(
    pair_candidates: Sequence[dict[str, Any]],
    identity_candidates: Sequence[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_trials: list[dict[str, Any]] = []
    private_trials: list[dict[str, Any]] = []
    orientations = _balanced_orientations(pair_candidates, seed)

    for candidate in pair_candidates:
        private_key = candidate["private_key"]
        trial_id = _opaque_id("trial", seed, private_key)
        conditions = list(candidate["conditions"])
        if orientations[repr(private_key)]:
            conditions.reverse()
        public_options = []
        option_roles: dict[str, str] = {}
        source_results: dict[str, str] = {}
        for condition in conditions:
            option_id = _opaque_id("option", seed, private_key, condition["variant"])
            public_options.append(
                {
                    "option_id": option_id,
                    "_source_path": condition["image_path"],
                }
            )
            option_roles[option_id] = condition["variant"]
            source_results[option_id] = condition["result_id"]
        public_trials.append(
            {
                "trial_id": trial_id,
                "block_code": candidate["block_code"],
                "kind": "pairwise",
                "reference_images": [
                    {"_source_path": path} for path in candidate["reference_paths"]
                ],
                "options": public_options,
                "special_responses": ["tie", "unjudgeable"],
            }
        )
        private_trials.append(
            {
                "trial_id": trial_id,
                "block_code": candidate["block_code"],
                "kind": "pairwise",
                "method": candidate["method"],
                "identity_id": candidate["identity_id"],
                "prompt_id": candidate["prompt_id"],
                "generation_seed": candidate["generation_seed"],
                "option_roles": option_roles,
                "source_result_ids": source_results,
            }
        )

    for candidate in identity_candidates:
        private_key = candidate["private_key"]
        trial_id = _opaque_id("trial", seed, private_key)
        options = list(candidate["options"])
        rng = random.Random(int(_stable_digest(seed, "afc-order", private_key)[:16], 16))
        rng.shuffle(options)
        public_options = []
        option_roles: dict[str, str] = {}
        option_identities: dict[str, str] = {}
        for option in options:
            option_id = _opaque_id("option", seed, private_key, option["role"])
            public_options.append(
                {
                    "option_id": option_id,
                    "_source_paths": option["image_paths"],
                }
            )
            option_roles[option_id] = option["role"]
            option_identities[option_id] = option["identity_id"]
        public_trials.append(
            {
                "trial_id": trial_id,
                "block_code": "I1",
                "kind": "four_afc",
                "query_image": {"_source_path": candidate["query_path"]},
                "options": public_options,
                "special_responses": ["none", "unjudgeable"],
            }
        )
        private_trials.append(
            {
                "trial_id": trial_id,
                "block_code": "I1",
                "kind": "four_afc",
                "method": candidate["method"],
                "variant": candidate["variant"],
                "prompt_id": candidate["prompt_id"],
                "generation_seed": candidate["generation_seed"],
                "source_result_id": candidate["result_id"],
                "target_identity_id": candidate["target_identity_id"],
                "donor_identity_id": candidate["donor_identity_id"],
                "global_identity_id": candidate["global_identity_id"],
                "patch_identity_id": candidate["patch_identity_id"],
                "option_roles": option_roles,
                "option_identity_ids": option_identities,
            }
        )

    rng = random.Random(int(_stable_digest(seed, "trial-order")[:16], 16))
    rng.shuffle(public_trials)
    trial_ids = [trial["trial_id"] for trial in public_trials]
    if len(trial_ids) != len(set(trial_ids)):
        raise HumanEvalError("Trial identifiers collided; make result rows uniquely identifiable")
    private_by_id = {trial["trial_id"]: trial for trial in private_trials}
    ordered_private = [private_by_id[trial["trial_id"]] for trial in public_trials]
    return public_trials, ordered_private


def _require_task_coverage(trials: Sequence[Mapping[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for trial in trials:
        block_code = str(trial["block_code"])
        counts[block_code] = counts.get(block_code, 0) + 1
    missing = [block for block in ("P0", "P1", "I1") if counts.get(block, 0) == 0]
    if missing:
        raise HumanEvalError(
            "The study must contain K1-vs-K4-diverse (P0), K4-diverse-vs-K4-repeat "
            f"(P1), and identity 4-AFC trials (I1); missing {missing}"
        )


def _stage_trial_media(
    trials: list[dict[str, Any]], media_dir: Path, html_parent: Path
) -> dict[str, str]:
    digests: dict[str, str] = {}

    def replace_source(image: dict[str, Any]) -> None:
        source = image.pop("_source_path")
        href, digest = _stage_image(source, media_dir, html_parent)
        image["src"] = href
        digests[href] = digest

    for trial in trials:
        for reference in trial.get("reference_images", []):
            replace_source(reference)
        if "query_image" in trial:
            replace_source(trial["query_image"])
        for option in trial["options"]:
            if "_source_paths" in option:
                source_paths = option.pop("_source_paths")
                option["images"] = []
                for source_path in source_paths:
                    image = {"_source_path": source_path}
                    replace_source(image)
                    option["images"].append(image)
            else:
                replace_source(option)
    return dict(sorted(digests.items()))


def _output_layout(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix:
        if output.suffix.lower() != ".html":
            raise HumanEvalError("A file output must use the .html extension")
        html_path = output
        blinded_json_path = output.with_name(f"{output.stem}.blinded.json")
        media_dir = output.with_name(f"{output.stem}_media")
    else:
        html_path = output / "index.html"
        blinded_json_path = output / "blinded_trials.json"
        media_dir = output / "media"
    return html_path.resolve(), blinded_json_path.resolve(), media_dir.resolve()


def _json_for_script(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_human_eval_html(study: Mapping[str, Any]) -> str:
    """Render a self-contained UI shell; face pixels remain external opaque media files."""
    payload = _json_for_script(study)
    template = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blinded face identity evaluation</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #f9fafb; }
    main { max-width: 1180px; margin: auto; padding: 24px; }
    .topbar, .nav, .status { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .topbar { justify-content: space-between; margin-bottom: 16px; }
    .panel { background: #1f2937; border: 1px solid #374151; border-radius: 14px; padding: 18px; }
    .images {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 16px;
    }
    .references { display: flex; gap: 10px; overflow-x: auto; margin: 12px 0 20px; }
    .references img { width: 130px; height: 130px; }
    img {
      display: block;
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: #0b1020;
      border-radius: 10px;
    }
    .choice {
      border: 2px solid transparent;
      border-radius: 12px;
      padding: 10px;
      background: #111827;
    }
    .choice.selected { border-color: #60a5fa; box-shadow: 0 0 0 2px #1d4ed8; }
    .choice-images { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; }
    .choice-images img { min-width: 0; }
    button, input { font: inherit; }
    button {
      border: 1px solid #6b7280;
      border-radius: 8px;
      padding: 9px 14px;
      cursor: pointer;
      background: #374151;
      color: #fff;
    }
    button:hover, button:focus-visible { background: #4b5563; outline: 2px solid #93c5fd; }
    button.primary { background: #2563eb; border-color: #3b82f6; }
    button.selected { background: #1d4ed8; border-color: #93c5fd; }
    button:disabled { opacity: .45; cursor: default; }
    .option-label { font-size: 1.1rem; font-weight: 700; margin: 8px 0; }
    .special { display: flex; gap: 10px; justify-content: center; margin: 20px 0 4px; }
    .nav { justify-content: space-between; margin-top: 18px; }
    .muted { color: #cbd5e1; }
    .warning { color: #fbbf24; min-height: 1.2em; }
    progress { width: min(420px, 100%); height: 14px; }
    input {
      border: 1px solid #6b7280;
      border-radius: 7px;
      padding: 8px;
      background: #111827;
      color: inherit;
    }
    h1, h2, p { margin-top: 0; }
    @media (max-width: 600px) { main { padding: 12px; } .panel { padding: 12px; } }
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <div>
      <h1 id="study-title">Blinded face identity evaluation</h1>
      <div class="status">
        <span id="progress-text"></span>
        <progress id="progress" max="1" value="0"></progress>
      </div>
    </div>
    <label>
      Participant code
      <input id="participant" autocomplete="off" maxlength="80" placeholder="optional">
    </label>
  </div>
  <p id="storage-warning" class="warning" role="status"></p>
  <section id="trial" class="panel" aria-live="polite"></section>
  <div class="nav">
    <button id="previous" type="button">Previous</button>
    <span id="position" class="muted"></span>
    <button id="next" type="button">Next</button>
    <button id="download" class="primary" type="button">Download responses JSON</button>
  </div>
</main>
<script id="study-data" type="application/json">__STUDY_JSON__</script>
<script>
(() => {
  "use strict";
  const study = JSON.parse(document.getElementById("study-data").textContent);
  const storageKey = `identity-layers-human-eval:${study.study_id}`;
  const allowedTrialIds = new Set(study.trials.map(t => t.trial_id));
  let state = {participant_id: "", responses: {}, current: 0};
  const warning = document.getElementById("storage-warning");

  function loadState() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (saved && typeof saved === "object") {
        state.participant_id = typeof saved.participant_id === "string" ? saved.participant_id : "";
        state.current = Number.isInteger(saved.current)
          ? Math.max(0, Math.min(saved.current, study.trials.length - 1))
          : 0;
        if (saved.responses && typeof saved.responses === "object") {
          for (const [trialId, response] of Object.entries(saved.responses)) {
            if (
              allowedTrialIds.has(trialId)
              && response
              && typeof response.selection === "string"
            ) {
              state.responses[trialId] = response;
            }
          }
        }
      }
    } catch (error) {
      warning.textContent = (
        "Browser storage is unavailable; download frequently to avoid losing responses."
      );
    }
  }

  function saveState() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch (error) {
      warning.textContent = "Responses could not be saved locally; use Download responses JSON.";
    }
  }

  function select(selection) {
    const trial = study.trials[state.current];
    const allowed = new Set([
      ...trial.options.map(option => option.option_id),
      ...trial.special_responses
    ]);
    if (!allowed.has(selection)) return;
    state.responses[trial.trial_id] = {selection, updated_at: new Date().toISOString()};
    saveState();
    render();
  }

  function image(src, alt) {
    const node = document.createElement("img");
    node.src = src;
    node.alt = alt;
    node.loading = "eager";
    node.decoding = "async";
    return node;
  }

  function choiceCard(option, label, selected, altPrefix) {
    const card = document.createElement("div");
    card.className = `choice${selected ? " selected" : ""}`;
    const sources = option.images || [{src: option.src}];
    const imageArea = document.createElement("div");
    imageArea.className = "choice-images";
    sources.forEach((item, index) => {
      imageArea.appendChild(image(item.src, `${altPrefix} ${label}, view ${index + 1}`));
    });
    card.appendChild(imageArea);
    const heading = document.createElement("div");
    heading.className = "option-label";
    heading.textContent = label;
    card.appendChild(heading);
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `Choose ${label}`;
    button.className = selected ? "selected" : "";
    button.addEventListener("click", () => select(option.option_id));
    card.appendChild(button);
    return card;
  }

  function specialButtons(trial, selected) {
    const area = document.createElement("div");
    area.className = "special";
    for (const value of trial.special_responses) {
      const button = document.createElement("button");
      button.type = "button";
      if (value === "tie") button.textContent = "Tie / equally similar";
      else if (value === "none") button.textContent = "None of these people";
      else button.textContent = "Unjudgeable";
      button.className = selected === value ? "selected" : "";
      button.addEventListener("click", () => select(value));
      area.appendChild(button);
    }
    return area;
  }

  function renderPairwise(container, trial, selected) {
    const title = document.createElement("h2");
    title.textContent = "Which generated face better preserves the reference identity?";
    container.appendChild(title);
    const help = document.createElement("p");
    help.className = "muted";
    help.textContent = (
      "Judge identity resemblance, not image quality, style, pose, lighting, or background."
    );
    container.appendChild(help);
    const referenceTitle = document.createElement("div");
    referenceTitle.className = "option-label";
    referenceTitle.textContent = "Identity reference";
    container.appendChild(referenceTitle);
    const refs = document.createElement("div");
    refs.className = "references";
    trial.reference_images.forEach((item, i) => {
      refs.appendChild(image(item.src, `Identity reference ${i + 1}`));
    });
    container.appendChild(refs);
    const options = document.createElement("div");
    options.className = "images";
    trial.options.forEach((option, i) => {
      const label = i === 0 ? "A" : "B";
      options.appendChild(
        choiceCard(option, label, selected === option.option_id, "Generated option")
      );
    });
    container.appendChild(options);
    container.appendChild(specialButtons(trial, selected));
  }

  function renderFourAfc(container, trial, selected) {
    const title = document.createElement("h2");
    title.textContent = "Which reference person does the generated face most resemble?";
    container.appendChild(title);
    const help = document.createElement("p");
    help.className = "muted";
    help.textContent = (
      "Choose one identity based only on the face. Use None or Unjudgeable when necessary."
    );
    container.appendChild(help);
    const queryTitle = document.createElement("div");
    queryTitle.className = "option-label";
    queryTitle.textContent = "Generated face";
    container.appendChild(queryTitle);
    const query = image(trial.query_image.src, "Generated face to identify");
    query.style.maxHeight = "430px";
    container.appendChild(query);
    const options = document.createElement("div");
    options.className = "images";
    const labels = ["A", "B", "C", "D"];
    trial.options.forEach((option, i) => {
      options.appendChild(
        choiceCard(option, labels[i], selected === option.option_id, "Identity option")
      );
    });
    container.appendChild(options);
    container.appendChild(specialButtons(trial, selected));
  }

  function render() {
    const count = Object.keys(state.responses).length;
    document.getElementById("study-title").textContent = study.title;
    document.getElementById("progress-text").textContent = (
      `${count} / ${study.trials.length} answered`
    );
    const progress = document.getElementById("progress");
    progress.max = study.trials.length;
    progress.value = count;
    document.getElementById("position").textContent = (
      `Question ${state.current + 1} of ${study.trials.length}`
    );
    document.getElementById("previous").disabled = state.current === 0;
    document.getElementById("next").disabled = state.current === study.trials.length - 1;
    const trial = study.trials[state.current];
    const selected = state.responses[trial.trial_id]?.selection || null;
    const container = document.getElementById("trial");
    container.replaceChildren();
    if (trial.kind === "pairwise") renderPairwise(container, trial, selected);
    else renderFourAfc(container, trial, selected);
  }

  function downloadResponses() {
    const responses = study.trials
      .filter(trial => state.responses[trial.trial_id])
      .map(trial => ({trial_id: trial.trial_id, ...state.responses[trial.trial_id]}));
    const output = {
      schema_version: "identity-layers-human-response/v1",
      study_id: study.study_id,
      participant_id: state.participant_id.trim() || null,
      exported_at: new Date().toISOString(),
      completed: responses.length === study.trials.length,
      trial_count: study.trials.length,
      responses
    };
    const blob = new Blob([JSON.stringify(output, null, 2) + "\n"], {type: "application/json"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${study.study_id}-responses.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  }

  document.getElementById("previous").addEventListener("click", () => {
    state.current -= 1;
    saveState();
    render();
  });
  document.getElementById("next").addEventListener("click", () => {
    state.current += 1;
    saveState();
    render();
  });
  document.getElementById("download").addEventListener("click", downloadResponses);
  const participant = document.getElementById("participant");
  participant.addEventListener("input", () => {
    state.participant_id = participant.value;
    saveState();
  });
  document.addEventListener("keydown", event => {
    if (event.target === participant) return;
    const trial = study.trials[state.current];
    const index = Number.parseInt(event.key, 10) - 1;
    if (index >= 0 && index < trial.options.length) select(trial.options[index].option_id);
    else if (event.key.toLowerCase() === "t") select("tie");
    else if (event.key.toLowerCase() === "n") select("none");
    else if (event.key.toLowerCase() === "u") select("unjudgeable");
    else if (event.key === "ArrowLeft" && state.current > 0) {
      state.current -= 1;
      saveState();
      render();
    } else if (event.key === "ArrowRight" && state.current < study.trials.length - 1) {
      state.current += 1;
      saveState();
      render();
    }
  });

  loadState();
  participant.value = state.participant_id;
  render();
})();
</script>
</body>
</html>
'''
    html = template.replace("__STUDY_JSON__", payload)
    if "data:image" in html.lower():
        raise AssertionError("The evaluator must never embed image data URIs")
    return html


def build_human_eval(
    manifest_path: str | Path,
    results_path: str | Path,
    output: str | Path,
    *,
    seed: int,
    asset_root: str | Path | None = None,
    references_per_identity: int = 3,
    title: str = "Blinded face identity evaluation",
) -> HumanEvalBuild:
    """Build blinded static evaluation artifacts from reference and generation tables.

    Public JSON and HTML contain no method names, identity IDs, condition labels, or original
    filenames. Images are copied to opaque content-addressed files instead of being embedded.
    The returned private key can be written to a researcher-only location with
    :func:`write_private_answer_key`.
    """
    if references_per_identity < 1:
        raise HumanEvalError("references_per_identity must be at least 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise HumanEvalError("seed must be an integer")
    manifest = Path(manifest_path).resolve()
    results = Path(results_path).resolve()
    root = None if asset_root is None else Path(asset_root).resolve()
    references = _canonical_references(manifest, root)
    records = _canonical_results(results, root)
    pair_candidates = _build_pair_candidates(
        records, references, references_per_identity, seed
    )
    identity_candidates = _build_identity_candidates(
        records,
        references,
        references_per_identity,
        seed,
    )
    public_trials, private_trials = _public_and_private_trials(
        pair_candidates, identity_candidates, seed
    )
    _require_task_coverage(public_trials)
    public_task_counts = {
        kind: sum(trial["kind"] == kind for trial in public_trials)
        for kind in sorted({trial["kind"] for trial in public_trials})
    }
    for trial in public_trials:
        trial.pop("block_code")

    html_path, blinded_json_path, media_dir = _output_layout(Path(output))
    media_digests = _stage_trial_media(public_trials, media_dir, html_path.parent)
    study_id = _opaque_id(
        "study", seed, [trial["trial_id"] for trial in public_trials], length=20
    )
    public_study = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "study_id": study_id,
        "title": str(title),
        "randomization": {
            "seed": seed,
            "algorithm": "sha256-derived streams; balanced pair orientation; shuffled trial order",
        },
        "trial_count": len(public_trials),
        "task_counts": public_task_counts,
        "media": {
            "policy": "external-content-addressed-files-no-inline-base64",
            "sha256": media_digests,
        },
        "trials": public_trials,
    }
    private_answer_key = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "study_id": study_id,
        "randomization_seed": seed,
        "source_manifest": str(manifest),
        "source_results": str(results),
        "trials": private_trials,
    }
    blinded_text = json.dumps(public_study, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if "data:image" in blinded_text.lower():
        raise AssertionError("The blinded JSON must never embed image data URIs")
    _write_text_atomic(blinded_json_path, blinded_text)
    _write_text_atomic(html_path, render_human_eval_html(public_study))
    return HumanEvalBuild(
        html_path=html_path,
        blinded_json_path=blinded_json_path,
        media_dir=media_dir,
        public_study=public_study,
        private_answer_key=private_answer_key,
    )


def write_private_answer_key(path: str | Path, answer_key: Mapping[str, Any]) -> Path:
    """Write the unblinding key; callers must keep this outside participant-facing output."""
    destination = Path(path).resolve()
    _write_text_atomic(
        destination,
        json.dumps(answer_key, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return destination


__all__ = [
    "HumanEvalBuild",
    "HumanEvalError",
    "PUBLIC_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "build_human_eval",
    "render_human_eval_html",
    "write_private_answer_key",
]
