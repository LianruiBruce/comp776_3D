import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from id_layers.auto_analysis import (
    ROOT,
    _project_path,
    compare_smoke_full_generation,
    identity_interval,
    paired_identity_contrast,
    plot_expression_comparison,
)


def test_pairing_is_by_ids_and_identity_means_not_row_count():
    frame = pd.DataFrame(
        [
            {
                "identity_id": identity,
                "prompt_id": "neutral",
                "base_seed": seed,
                "condition": condition,
                "value": value,
            }
            for identity, seeds, delta in (("a", (1, 2), 1), ("b", (1,), 3))
            for seed in seeds
            for condition, value in (("t", delta), ("c", 0))
        ]
    )
    pairs, result = paired_identity_contrast(
        frame, metric="value", factor="condition", treatment="t", control="c", resamples=50
    )
    shuffled_pairs, shuffled = paired_identity_contrast(
        frame.sample(frac=1, random_state=12),
        metric="value",
        factor="condition",
        treatment="t",
        control="c",
        resamples=50,
    )
    assert result["estimate"] == 2
    assert result["valid_pair_count"] == 3
    assert result == shuffled
    pd.testing.assert_frame_equal(pairs, shuffled_pairs)


def test_pairing_retains_missing_pairs_and_rejects_duplicate_cells():
    frame = pd.DataFrame(
        [
            {"identity_id": "a", "prompt_id": "p", "base_seed": 1, "condition": "t", "value": 2.0},
            {
                "identity_id": "a",
                "prompt_id": "p",
                "base_seed": 1,
                "condition": "c",
                "value": np.nan,
            },
        ]
    )
    pairs, result = paired_identity_contrast(
        frame, metric="value", factor="condition", treatment="t", control="c"
    )
    assert result["expected_pair_count"] == 1
    assert result["valid_pair_count"] == 0
    assert result["estimate"] is None
    assert np.isnan(pairs.difference.iloc[0])
    with pytest.raises(ValueError, match="Duplicate cells"):
        paired_identity_contrast(
            pd.concat([frame, frame]),
            metric="value",
            factor="condition",
            treatment="t",
            control="c",
        )


def test_intervals_reject_duplicate_identity_and_infinite_scores():
    with pytest.raises(ValueError, match="unique"):
        identity_interval(pd.Series([1, 2], index=["a", "a"]), resamples=10, seed=1)
    with pytest.raises(ValueError, match="Infinite"):
        identity_interval(pd.Series([1, np.inf], index=["a", "b"]), resamples=10, seed=1)


def test_relative_paths_are_resolved_against_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _project_path(Path("artifacts/runs/example")) == ROOT / "artifacts/runs/example"


def test_comparison_rejects_smoke_and_full_cohort_mismatch(tmp_path):
    paths = {}
    for name, identities in (("full", ["a", "b"]), ("smoke", ["a"])):
        path = tmp_path / name
        path.mkdir()
        (path / "resolved_config.json").write_text(
            json.dumps({"new_expression_protocol": True}), encoding="utf-8"
        )
        (path / "status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        pd.DataFrame({"identity_id": identities}).to_csv(
            path / "expression_identity.csv", index=False
        )
        for filename in ("expression_paired_summary.csv", "expression_summary.csv"):
            pd.DataFrame({"metric": ["mouth_smile_mean"]}).to_csv(path / filename, index=False)
        paths[name] = path
    with pytest.raises(ValueError, match="cohorts differ"):
        plot_expression_comparison(paths, tmp_path / "unused_output")
    assert not (tmp_path / "unused_output").exists()


def test_smoke_full_repeatability_requires_frozen_metadata_and_image_hash(tmp_path):
    base = {
        "sample_id": "SYNTHETIC",
        "identity_id": "SYNTHETIC_ID",
        "condition": "k1_full",
        "prompt_id": "neutral",
        "prompt": "SYNTHETIC PROMPT",
        "base_seed": 776,
        "effective_seed": 776,
        "latent_sha256": "1" * 64,
        "width": 2,
        "height": 2,
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "negative_prompt": "",
        "reference_sample_ids": ["SYNTHETIC_REF"],
        "global_reference_sample_ids": ["SYNTHETIC_REF"],
        "patch_reference_sample_ids": ["SYNTHETIC_REF"],
        "global_embedding_identity_id": "SYNTHETIC_ID",
        "patch_image_identity_id": "SYNTHETIC_ID",
        "status": "complete",
        "output_relative_path": "synthetic.png",
    }
    for name in ("smoke", "full"):
        path = tmp_path / name
        path.mkdir()
        Image.new("RGB", (2, 2), (1, 2, 3)).save(path / "synthetic.png")
        row = {
            **base,
            "output_sha256": hashlib.sha256((path / "synthetic.png").read_bytes()).hexdigest(),
        }
        (path / "generation_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = compare_smoke_full_generation(
        tmp_path / "smoke", tmp_path / "full", tmp_path / "comparison", expected_smoke_count=1
    )
    assert summary["file_exact_match_count"] == summary["pixel_exact_match_count"] == 1
    row["latent_sha256"] = "2" * 64
    (tmp_path / "full/generation_manifest.jsonl").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="pairing mismatch"):
        compare_smoke_full_generation(
            tmp_path / "smoke", tmp_path / "full", tmp_path / "bad_pair", expected_smoke_count=1
        )
    row["latent_sha256"] = "1" * 64
    (tmp_path / "full/generation_manifest.jsonl").write_text(json.dumps(row), encoding="utf-8")
    Image.new("RGB", (2, 2), (7, 8, 9)).save(tmp_path / "full/synthetic.png")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_smoke_full_generation(
            tmp_path / "smoke", tmp_path / "full", tmp_path / "bad_hash", expected_smoke_count=1
        )
