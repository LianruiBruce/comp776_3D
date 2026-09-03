from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from id_layers.generation_metrics import (
    build_identity_templates,
    cohort_geometry,
    contraction_ratio,
    gallery_identity_metrics,
    identity_bootstrap_mean,
    paired_identity_bootstrap,
)


def test_gallery_metrics_separate_aligned_and_end_to_end_rank1() -> None:
    templates = np.eye(3, dtype=np.float32)
    generated = np.asarray([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2]], dtype=np.float32)
    result = gallery_identity_metrics(
        generated,
        ["a", "b"],
        templates,
        ["a", "b", "c"],
        donor_ids=["b", "a"],
        end_to_end_valid=[True, False],
    )
    assert result["aligned_rank1"].tolist() == [True, True]
    assert result["end_to_end_rank1"].tolist() == [True, False]
    assert bool((result["target_minus_donor"] > 0).all())


def test_templates_average_each_identity_then_normalize() -> None:
    embeddings = np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    templates, identities = build_identity_templates(embeddings, ["a", "a", "b"])
    assert identities == ["a", "b"]
    assert np.allclose(np.linalg.norm(templates, axis=1), 1.0)
    assert templates[0, 0] > templates[0, 1]


def test_contraction_ratio_detects_collapsed_identity_geometry() -> None:
    real = np.eye(3, dtype=np.float32)
    generated = np.asarray(
        [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.99, 0.0, 0.1]], dtype=np.float32
    )
    assert contraction_ratio(generated, real) < 0.02
    assert cohort_geometry(real).covariance_effective_rank == pytest.approx(2.0)


def test_paired_bootstrap_averages_repeats_within_identity() -> None:
    frame = pd.DataFrame(
        {
            "identity": ["x", "x", "x", "x", "y", "y"],
            "condition": ["a", "a", "b", "b", "a", "b"],
            "score": [3.0, 5.0, 1.0, 3.0, 4.0, 1.0],
        }
    )
    result = paired_identity_bootstrap(
        frame,
        identity_column="identity",
        condition_column="condition",
        value_column="score",
        condition_a="a",
        condition_b="b",
        resamples=100,
        seed=7,
    )
    assert result["identity_count"] == 2
    assert result["estimate_a_minus_b"] == pytest.approx(2.5)


def test_gallery_rejects_missing_target_template() -> None:
    with pytest.raises(ValueError, match="missing from gallery"):
        gallery_identity_metrics(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            ["missing"],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            ["a"],
        )


def test_identity_bootstrap_mean_weights_identities_equally() -> None:
    frame = pd.DataFrame(
        {
            "identity": ["many", "many", "many", "one"],
            "score": [1.0, 1.0, 1.0, 3.0],
        }
    )
    result = identity_bootstrap_mean(
        frame,
        identity_column="identity",
        value_column="score",
        resamples=100,
        seed=1,
    )
    assert result["identity_count"] == 2
    assert result["estimate"] == pytest.approx(2.0)
