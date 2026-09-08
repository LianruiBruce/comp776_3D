from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from id_layers.auto_diagnostics import (
    ARC_FACE_112,
    LPIPSAlexAdapter,
    aggregate_reference_distances,
    apply_face_transform,
    build_reference_pairs,
    build_seed_pairs,
    duplicate_groups,
    file_sha256,
    fixed_face_crop,
    image_qc,
    load_keyed_array_cache,
    prepare_lpips_whole,
    resolve_inside,
    save_keyed_array_cache,
    seed_embedding_distances,
    summarize_landmarker_result,
    validate_generation_records,
)


def _rows(tmp_path: Path) -> list[dict]:
    rows = []
    for condition in ("k1_full", "k4_repeat_full"):
        for seed in (776, 1776):
            name = f"a_{condition}_{seed}"
            path = tmp_path / f"{name}.png"
            Image.new("RGB", (32, 32), (seed % 255, 15, 67)).save(path)
            refs = ["a_real"] * (4 if condition == "k4_repeat_full" else 1)
            rows.append(
                {
                    "sample_id": name,
                    "identity_id": "a",
                    "donor_identity_id": "b",
                    "condition": condition,
                    "prompt_id": "neutral",
                    "prompt": "neutral person",
                    "base_seed": seed,
                    "effective_seed": seed,
                    "latent_sha256": str(seed // 1000) * 64,
                    "status": "complete",
                    "output_relative_path": path.name,
                    "output_sha256": file_sha256(path),
                    "reference_sample_ids": refs,
                    "global_reference_sample_ids": refs,
                    "patch_reference_sample_ids": refs,
                    "global_embedding_identity_id": "a",
                    "patch_image_identity_id": "a",
                }
            )
    return rows


def test_full_ledger_and_pair_results_are_order_invariant(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    validation = validate_generation_records(rows, tmp_path, expected_count=4)
    assert validation["files_hash_verified"] == 4
    assert validation["seed_pair_count"] == 2
    assert build_seed_pairs(rows) == build_seed_pairs(list(reversed(rows)))


@pytest.mark.parametrize(
    "failure", ["duplicate_id", "duplicate_cell", "bad_hash", "bad_latent", "bad_seed", "bad_pair"]
)
def test_ledger_rejects_invalid_provenance(tmp_path: Path, failure: str) -> None:
    rows = _rows(tmp_path)
    if failure == "duplicate_id":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif failure == "duplicate_cell":
        rows[1]["base_seed"] = rows[0]["base_seed"]
    elif failure == "bad_hash":
        rows[0]["output_sha256"] = "0" * 64
    elif failure == "bad_latent":
        rows[2]["latent_sha256"] = "f" * 64
    elif failure == "bad_seed":
        rows[0]["base_seed"] = 1
    elif failure == "bad_pair":
        rows[0]["prompt"] = "another prompt"
    with pytest.raises(ValueError):
        validate_generation_records(rows, tmp_path, expected_count=4)


def test_generation_failure_remains_in_cell_and_pair_counts(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    rows[0].update(
        status="failed", error="render failed", output_sha256=None, output_relative_path=None
    )
    result = validate_generation_records(rows, tmp_path, expected_count=4)
    assert result["failed_cell_count"] == 1
    assert result["complete_image_count"] == 3
    assert result["seed_pair_count"] == 2
    assert (
        sum(
            pair["generation_pair_status"] == "generation_failed" for pair in build_seed_pairs(rows)
        )
        == 1
    )


def test_factorial_cannot_drop_a_condition_silently(tmp_path: Path) -> None:
    rows = _rows(tmp_path)[:2]
    with pytest.raises(ValueError, match="Factorial ledger mismatch"):
        validate_generation_records(
            rows, tmp_path, expected_conditions=["k1_full", "k4_repeat_full"]
        )


def test_source_paths_cannot_escape_expected_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        resolve_inside(tmp_path, "../elsewhere/image.png")
    with pytest.raises(ValueError, match="relative"):
        resolve_inside(tmp_path, tmp_path / "image.png")


def test_qc_constant_corrupt_and_exact_decoded_duplicates(tmp_path: Path) -> None:
    first, second, white, corrupt = [
        tmp_path / name for name in ("first.png", "second.png", "white.png", "bad.png")
    ]
    image = Image.new("RGB", (64, 32), 0)
    image.save(first)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("comment", "Same pixels, different file bytes")
    image.save(second, pnginfo=metadata)
    Image.new("RGB", (64, 32), "white").save(white)
    corrupt.write_bytes(b"not an image")
    records = [image_qc(path, path.stem) for path in (first, second, white, corrupt)]
    assert records[0]["near_black_fraction"] == 1
    assert records[0]["blur_laplacian_variance_longest256"] == 0
    assert records[2]["near_white_fraction"] == 1
    assert records[3]["status"] == "failed"
    assert records[3]["pixel_sha256"] is None
    result = duplicate_groups(records)
    assert result["decode_failure_count"] == 1
    assert result["file_sha256_duplicate_pair_count"] == 0
    assert result["pixel_sha256_duplicate_pair_count"] == 1
    with pytest.raises(ValueError, match="hash mismatch"):
        image_qc(first, "first", expected_sha256="f" * 64)


def test_lpips_whole_preprocessing_does_not_stretch_rectangles() -> None:
    result = prepare_lpips_whole(Image.new("RGB", (640, 480), "red"))
    assert result.shape == (3, 256, 256)
    assert result[:, 0, 0] == pytest.approx([128 / 127.5 - 1] * 3, abs=1e-7)
    assert result[:, 32, 0] == pytest.approx([1, -1, -1])
    assert result[:, 223, 255] == pytest.approx([1, -1, -1])
    assert result[:, 224, 0] == pytest.approx([128 / 127.5 - 1] * 3, abs=1e-7)


def test_fixed_face_crop_reuses_identical_original_geometry() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")
    rng = np.random.default_rng(7)
    original = Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8))
    crop, matrix = fixed_face_crop(original, ARC_FACE_112 * (256 / 112))
    assert matrix == pytest.approx(np.asarray([[1, 0, 0], [0, 1, 0]]), abs=1e-12)
    reapplied = apply_face_transform(original, matrix)
    assert np.array_equal(np.asarray(crop), np.asarray(reapplied))
    assert np.array_equal(np.asarray(crop), np.asarray(original)[:256, :256])
    with pytest.raises(ValueError, match="landmarks"):
        fixed_face_crop(original, [[float("nan"), 0]] * 5)
    with pytest.raises(ValueError):
        fixed_face_crop(original, [[0, 0]] * 5)


def test_keyed_cache_shuffling_is_safe_and_overwrite_hash_drift_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.npz"
    records = [
        {"sample_id": "a", "source_sha256": "a" * 64},
        {"sample_id": "b", "source_sha256": "b" * 64},
    ]
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    save_keyed_array_cache(path, records, {"embedding": values}, preprocessing_version="test1")
    reversed_values = load_keyed_array_cache(path, records[::-1], preprocessing_version="test1")[
        "embedding"
    ]
    assert np.array_equal(reversed_values, values[::-1])
    with pytest.raises(FileExistsError):
        save_keyed_array_cache(path, records, {"embedding": values}, preprocessing_version="test1")
    changed = copy.deepcopy(records)
    changed[0]["source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        load_keyed_array_cache(path, changed, preprocessing_version="test1")
    with pytest.raises(ValueError, match="preprocessing"):
        load_keyed_array_cache(path, records, preprocessing_version="test2")
    with pytest.raises(ValueError, match="Duplicate"):
        load_keyed_array_cache(path, [records[0], records[0]], preprocessing_version="test1")
    with pytest.raises(ValueError, match="Nonfinite"):
        save_keyed_array_cache(
            tmp_path / "invalid.npz",
            records,
            {"embedding": np.asarray([[np.nan], [1]])},
            preprocessing_version="v1",
        )


def test_unkeyed_legacy_cache_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.npz"
    np.savez(path, embeddings=np.eye(2))
    with pytest.raises(ValueError, match="unkeyed"):
        load_keyed_array_cache(
            path, [{"sample_id": "x", "source_sha256": "a" * 64}], preprocessing_version="v1"
        )


def test_reference_pairs_preserve_common_k1_and_mark_inactive_text_route(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    text_row = copy.deepcopy(rows[0])
    text_row.update(
        sample_id="text",
        condition="text_only",
        global_embedding_identity_id=None,
        patch_image_identity_id=None,
    )
    refs = {"a_real": {"sample_id": "a_real", "identity_id": "a", "source_sha256": "a" * 64}}
    pairs = build_reference_pairs(rows + [text_row], {"a": "a_real"}, refs)
    repeated = [pair for pair in pairs if pair["condition"] == "k4_repeat_full"]
    assert len(repeated) == 2
    assert all(pair["actual_distinct_reference_count"] == 1 for pair in repeated)
    assert all(pair["patch_slot_count"] == 4 for pair in repeated)
    text = next(pair for pair in pairs if pair["condition"] == "text_only")
    assert text["is_common_k1"]
    assert not text["is_actual_input_reference"]
    scored = [{**pair, "space": "whole", "status": "complete", "distance": 0.3} for pair in pairs]
    aggregate = aggregate_reference_distances(scored)
    text_result = next(row for row in aggregate if row["condition"] == "text_only")
    assert text_result["common_k1_distance"] == 0.3
    assert text_result["actual_reference_mean_distance"] is None
    assert text_result["actual_reference_status"] == "not_applicable_no_injection"


def test_reference_donor_mismatch_is_rejected(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    rows[0]["global_embedding_identity_id"] = "wrong_person"
    refs = {"a_real": {"sample_id": "a_real", "identity_id": "a", "source_sha256": "a" * 64}}
    with pytest.raises(ValueError, match="reference identity mismatch"):
        build_reference_pairs(rows, {"a": "a_real"}, refs)


def test_partial_reference_failure_does_not_turn_into_optimistic_minimum(tmp_path: Path) -> None:
    rows = _rows(tmp_path)[:1]
    rows[0].update(
        global_reference_sample_ids=["a_real", "a_real2"],
        patch_reference_sample_ids=["a_real", "a_real2"],
    )
    refs = {
        name: {"sample_id": name, "identity_id": "a", "source_sha256": name[-1] * 64}
        for name in ("a_real", "a_real2")
    }
    pairs = build_reference_pairs(rows, {"a": "a_real"}, refs)
    scored = [
        {
            **pair,
            "space": "whole",
            "status": "complete" if pair["is_common_k1"] else "failed",
            "distance": 0.1 if pair["is_common_k1"] else None,
        }
        for pair in pairs
    ]
    row = aggregate_reference_distances(scored)[0]
    assert row["common_k1_distance"] == 0.1
    assert row["actual_reference_min_distance"] is None
    assert row["valid_actual_reference_count"] == 1


def test_seed_distances_handle_identity_change_and_missing_without_nan(tmp_path: Path) -> None:
    pair = build_seed_pairs(_rows(tmp_path))[0]
    vectors = {pair["lhs_id"]: np.asarray([2, 0]), pair["rhs_id"]: np.asarray([0, 3])}
    result = seed_embedding_distances([pair], vectors, evaluator="test")[0]
    assert result["embedding_cosine_distance"] == 1
    vectors[pair["rhs_id"]] = np.asarray([7, 0])
    assert (
        seed_embedding_distances([pair], vectors, evaluator="test")[0]["embedding_cosine_distance"]
        == 0
    )
    vectors.pop(pair["rhs_id"])
    failure = seed_embedding_distances([pair], vectors, evaluator="test")[0]
    assert failure["status"] == "failed"
    assert failure["embedding_cosine_distance"] is None
    assert failure["error"]


def _landmarks(count: int, *, matrix: np.ndarray | None = None) -> SimpleNamespace:
    coefficients = [
        SimpleNamespace(category_name=name, score=value)
        for name, value in (("mouthSmileLeft", 0.2), ("mouthSmileRight", 0.6), ("jawOpen", 0.1))
    ]
    return SimpleNamespace(
        face_landmarks=[[]] * count,
        face_blendshapes=[coefficients] * count,
        facial_transformation_matrixes=[np.eye(4) if matrix is None else matrix] * count,
    )


@pytest.mark.parametrize("count,error", [(0, "no_face_detected"), (2, "multiple_faces_detected")])
def test_landmarker_no_and_multiple_faces_are_explicit_failures(count: int, error: str) -> None:
    row = summarize_landmarker_result(_landmarks(count))
    assert row["error"] == error
    assert row["mouth_smile_mean"] is None
    assert row["pose_matrix_4x4"] is None


def test_landmarker_smile_is_continuous_and_matrix_scale_is_not_pose() -> None:
    matrix = np.eye(4)
    matrix[:3, :3] *= 2
    row = summarize_landmarker_result(_landmarks(1, matrix=matrix))
    assert row["status"] == "complete"
    assert row["mouth_smile_mean"] == pytest.approx(0.4)
    assert row["pose_extrinsic_xyz_degrees"] == pytest.approx([0, 0, 0])
    matrix[:3, :3] = 0
    bad = summarize_landmarker_result(_landmarks(1, matrix=matrix))
    assert bad["status"] == "failed"
    assert bad["error"] == "singular_transformation_matrix"


def test_lpips_missing_face_crop_has_failure_row_without_model_call(tmp_path: Path) -> None:
    adapter = LPIPSAlexAdapter.__new__(LPIPSAlexAdapter)
    pair = {"pair_id": "p", "lhs_id": "a", "rhs_id": "b"}
    result = adapter.score_pairs([pair], {}, space="face")[0]
    assert result["status"] == "failed"
    assert result["distance"] is None
    assert result["error"]
