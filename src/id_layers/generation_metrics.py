from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def l2_normalize(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Normalize a vector or a matrix row-wise without silently accepting zeros."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in {1, 2}:
        raise ValueError("values must be a vector or a matrix")
    if not np.isfinite(array).all():
        raise ValueError("values contains non-finite entries")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if bool(np.any(norms <= eps)):
        raise ValueError("Cannot normalize a zero-length embedding")
    return array / norms


def build_identity_templates(
    embeddings: np.ndarray,
    identity_ids: Sequence[str],
    identity_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Average normalized query embeddings per identity and normalize again."""
    values = l2_normalize(embeddings)
    labels = np.asarray(list(identity_ids), dtype=object)
    if values.ndim != 2 or len(labels) != len(values):
        raise ValueError("embeddings and identity_ids must have the same length")
    order = list(identity_order) if identity_order is not None else sorted(set(labels))
    if not order or len(order) != len(set(order)):
        raise ValueError("identity_order must contain unique identities")
    unknown = sorted(set(labels) - set(order))
    if unknown:
        raise ValueError(f"identity_order omits query identities: {unknown}")
    missing = [identity for identity in order if not bool(np.any(labels == identity))]
    if missing:
        raise ValueError(f"No query embeddings for identities: {missing}")
    templates = np.stack(
        [values[labels == identity].mean(axis=0) for identity in order], axis=0
    )
    return l2_normalize(templates), order


def gallery_identity_metrics(
    embeddings: np.ndarray,
    target_ids: Sequence[str],
    templates: np.ndarray,
    template_ids: Sequence[str],
    *,
    donor_ids: Sequence[str | None] | None = None,
    end_to_end_valid: Sequence[bool] | None = None,
) -> pd.DataFrame:
    """Score generated faces against a real-query identity gallery.

    Failed or multi-face outputs may still have an aligned diagnostic embedding, but
    ``end_to_end_valid=False`` forces their end-to-end Rank-1 contribution to zero.
    """
    values = l2_normalize(embeddings)
    gallery = l2_normalize(templates)
    targets = list(target_ids)
    gallery_ids = list(template_ids)
    if values.ndim != 2 or gallery.ndim != 2 or values.shape[1] != gallery.shape[1]:
        raise ValueError("Generated and template embeddings must be compatible matrices")
    if len(values) != len(targets):
        raise ValueError("embeddings and target_ids must have the same length")
    if len(gallery) != len(gallery_ids) or len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError("template_ids must uniquely label every template")
    unknown_targets = sorted(set(targets) - set(gallery_ids))
    if unknown_targets:
        raise ValueError(f"Target identities missing from gallery: {unknown_targets}")
    donors = [None] * len(values) if donor_ids is None else list(donor_ids)
    if len(donors) != len(values):
        raise ValueError("donor_ids must have one value per generated embedding")
    valid = (
        np.ones(len(values), dtype=bool)
        if end_to_end_valid is None
        else np.asarray(end_to_end_valid, dtype=bool)
    )
    if valid.shape != (len(values),):
        raise ValueError("end_to_end_valid must have one value per generated embedding")

    gallery_index = {identity: index for index, identity in enumerate(gallery_ids)}
    similarities = values @ gallery.T
    rows: list[dict[str, Any]] = []
    for row_index, target in enumerate(targets):
        target_index = gallery_index[target]
        scores = similarities[row_index]
        impostor_mask = np.ones(len(gallery), dtype=bool)
        impostor_mask[target_index] = False
        max_impostor = float(scores[impostor_mask].max())
        target_similarity = float(scores[target_index])
        ranking = np.argsort(-scores, kind="stable")
        target_rank = int(np.flatnonzero(ranking == target_index)[0] + 1)
        donor = donors[row_index]
        donor_similarity = (
            float(scores[gallery_index[donor]]) if donor in gallery_index else np.nan
        )
        aligned_rank1 = target_rank == 1
        rows.append(
            {
                "target_similarity": target_similarity,
                "donor_similarity": donor_similarity,
                "target_minus_donor": (
                    target_similarity - donor_similarity
                    if np.isfinite(donor_similarity)
                    else np.nan
                ),
                "max_impostor_similarity": max_impostor,
                "target_impostor_margin": target_similarity - max_impostor,
                "target_rank": target_rank,
                "aligned_rank1": aligned_rank1,
                "end_to_end_valid": bool(valid[row_index]),
                "end_to_end_rank1": bool(valid[row_index] and aligned_rank1),
            }
        )
    return pd.DataFrame(rows)


def identity_centroids(
    embeddings: np.ndarray, identity_ids: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Return one normalized centroid per identity in sorted order."""
    labels = list(identity_ids)
    return build_identity_templates(embeddings, labels)


@dataclass(frozen=True)
class CohortGeometry:
    identity_count: int
    between_identity_spread: float
    mean_impostor_similarity: float
    covariance_effective_rank: float


def cohort_geometry(centroids: np.ndarray) -> CohortGeometry:
    """Describe cohort separation; input rows must be identity centroids."""
    values = l2_normalize(centroids).astype(np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("At least two identity centroids are required")
    similarity = values @ values.T
    upper = similarity[np.triu_indices(len(values), k=1)]
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2 / max(len(values) - 1, 1)
    positive = eigenvalues[eigenvalues > np.finfo(np.float64).eps]
    if len(positive) == 0:
        effective_rank = 0.0
    else:
        probabilities = positive / positive.sum()
        effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    return CohortGeometry(
        identity_count=len(values),
        between_identity_spread=float(np.mean(1.0 - upper)),
        mean_impostor_similarity=float(np.mean(upper)),
        covariance_effective_rank=effective_rank,
    )


def contraction_ratio(generated_centroids: np.ndarray, real_centroids: np.ndarray) -> float:
    """Return between-identity spread(generated) / spread(real)."""
    generated = cohort_geometry(generated_centroids)
    real = cohort_geometry(real_centroids)
    if real.between_identity_spread <= 0.0:
        raise ValueError("Real-query between-identity spread must be positive")
    return generated.between_identity_spread / real.between_identity_spread


def paired_identity_bootstrap(
    frame: pd.DataFrame,
    *,
    identity_column: str,
    condition_column: str,
    value_column: str,
    condition_a: str,
    condition_b: str,
    resamples: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Bootstrap a paired A-minus-B contrast after averaging within identity."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    required = {identity_column, condition_column, value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing bootstrap columns: {missing}")
    subset = frame[frame[condition_column].isin([condition_a, condition_b])]
    means = subset.groupby([identity_column, condition_column])[value_column].mean().unstack()
    if condition_a not in means or condition_b not in means:
        raise ValueError("Both conditions must be present")
    paired = means[[condition_a, condition_b]].dropna()
    if paired.empty:
        raise ValueError("No identities have observations in both conditions")
    differences = (paired[condition_a] - paired[condition_b]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "identity_count": int(len(differences)),
        "estimate_a_minus_b": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def identity_bootstrap_mean(
    frame: pd.DataFrame,
    *,
    identity_column: str,
    value_column: str,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap a mean after first averaging repeated observations per identity."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    missing = sorted({identity_column, value_column} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing bootstrap columns: {missing}")
    identity_values = (
        frame.groupby(identity_column, sort=True)[value_column]
        .mean()
        .dropna()
        .to_numpy(dtype=np.float64)
    )
    if len(identity_values) == 0:
        raise ValueError("No finite identity-level values to bootstrap")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(identity_values), size=(resamples, len(identity_values))
    )
    samples = identity_values[indices].mean(axis=1)
    return {
        "identity_count": int(len(identity_values)),
        "estimate": float(identity_values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }
