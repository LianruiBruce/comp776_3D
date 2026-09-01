from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def _verification_pair_scores(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    labels = np.asarray(labels)
    similarities = embeddings @ embeddings.T
    upper = np.triu_indices(len(labels), k=1)
    pair_scores = similarities[upper]
    pair_labels = labels[upper[0]] == labels[upper[1]]
    positive = pair_scores[pair_labels]
    negative = pair_scores[~pair_labels]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both positive and negative verification pairs are required")
    return positive, negative, similarities


def verification_metrics(
    embeddings: np.ndarray, labels: np.ndarray, far_targets: list[float]
) -> dict[str, Any]:
    labels = np.asarray(labels)
    positive, negative, similarities = _verification_pair_scores(embeddings, labels)
    pair_scores = np.concatenate([positive, negative])
    pair_labels = np.concatenate(
        [np.ones(positive.size, dtype=bool), np.zeros(negative.size, dtype=bool)]
    )

    binary = pair_labels.astype(np.int32)
    fpr, tpr, thresholds = roc_curve(binary, pair_scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    pooled_std = np.sqrt(0.5 * (positive.var(ddof=1) + negative.var(ddof=1)))

    result: dict[str, Any] = {
        "positive_pairs": int(positive.size),
        "negative_pairs": int(negative.size),
        "positive_cosine_mean": float(positive.mean()),
        "positive_cosine_std": float(positive.std(ddof=1)),
        "negative_cosine_mean": float(negative.mean()),
        "negative_cosine_std": float(negative.std(ddof=1)),
        "cosine_gap": float(positive.mean() - negative.mean()),
        "d_prime": float((positive.mean() - negative.mean()) / max(pooled_std, 1e-12)),
        "roc_auc": float(roc_auc_score(binary, pair_scores)),
        "eer": float(0.5 * (fpr[eer_index] + fnr[eer_index])),
        "eer_threshold": float(thresholds[eer_index]),
    }
    for target in far_targets:
        eligible = tpr[fpr <= target]
        result[f"tar_at_far_{target:g}"] = float(eligible.max()) if eligible.size else 0.0

    np.fill_diagonal(similarities, -np.inf)
    nearest = similarities.argmax(axis=1)
    result["loo_top1"] = float(np.mean(labels[nearest] == labels))
    return result


def _threshold_from_negative_scores(negative: np.ndarray, target_far: float) -> float:
    if not 0.0 <= target_far <= 1.0:
        raise ValueError(f"FAR target must be in [0, 1], received {target_far}")
    negative = np.asarray(negative, dtype=np.float64)
    unique = np.unique(negative)
    above_max = np.nextafter(float(unique.max()), np.inf)
    candidates = np.concatenate([[above_max], unique])
    fars = np.asarray([(negative >= threshold).mean() for threshold in candidates])
    eligible = np.flatnonzero(fars <= target_far)
    best_far = fars[eligible].max()
    best = eligible[fars[eligible] == best_far]
    return float(candidates[best].min())


def frozen_threshold_metrics(
    calibration_embeddings: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_embeddings: np.ndarray,
    evaluation_labels: np.ndarray,
    far_targets: list[float],
) -> dict[str, Any]:
    calibration_positive, calibration_negative, _ = _verification_pair_scores(
        calibration_embeddings, calibration_labels
    )
    evaluation_positive, evaluation_negative, _ = _verification_pair_scores(
        evaluation_embeddings, evaluation_labels
    )
    operating_points = {}
    for target in far_targets:
        threshold = _threshold_from_negative_scores(calibration_negative, target)
        calibration_positive_64 = calibration_positive.astype(np.float64, copy=False)
        calibration_negative_64 = calibration_negative.astype(np.float64, copy=False)
        evaluation_positive_64 = evaluation_positive.astype(np.float64, copy=False)
        evaluation_negative_64 = evaluation_negative.astype(np.float64, copy=False)
        operating_points[f"far_{target:g}"] = {
            "target_far": float(target),
            "threshold": threshold,
            "calibration_far": float((calibration_negative_64 >= threshold).mean()),
            "calibration_tar": float((calibration_positive_64 >= threshold).mean()),
            "evaluation_far": float((evaluation_negative_64 >= threshold).mean()),
            "evaluation_tar": float((evaluation_positive_64 >= threshold).mean()),
        }
    return {
        "calibration_positive_pairs": int(calibration_positive.size),
        "calibration_negative_pairs": int(calibration_negative.size),
        "evaluation_positive_pairs": int(evaluation_positive.size),
        "evaluation_negative_pairs": int(evaluation_negative.size),
        "operating_points": operating_points,
    }


def evaluate_all_layers(
    embeddings: dict[tuple[int, str], np.ndarray],
    manifest: pd.DataFrame,
    splits: list[str],
    far_targets: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in splits:
        mask = manifest["split"].to_numpy() == split
        labels = manifest.loc[mask, "identity_id"].to_numpy()
        for (layer, representation), values in sorted(embeddings.items()):
            metrics = verification_metrics(values[mask], labels, far_targets)
            rows.append(
                {
                    "split": split,
                    "layer": layer,
                    "representation": representation,
                    "embedding_dim": values.shape[1],
                    "samples": int(mask.sum()),
                    "identities": int(np.unique(labels).size),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["representation", "split", "layer"])


def evaluate_selected_layers(
    embeddings: dict[tuple[int, str], np.ndarray],
    manifest: pd.DataFrame,
    selection: dict[str, Any],
    split: str,
    far_targets: list[float],
) -> pd.DataFrame:
    mask = manifest["split"].to_numpy() == split
    labels = manifest.loc[mask, "identity_id"].to_numpy()
    rows = []
    for representation, record in selection["representations"].items():
        layer = int(record["selected_layer"])
        values = embeddings[(layer, representation)]
        rows.append(
            {
                "split": split,
                "layer": layer,
                "representation": representation,
                "embedding_dim": values.shape[1],
                "samples": int(mask.sum()),
                "identities": int(np.unique(labels).size),
                **verification_metrics(values[mask], labels, far_targets),
            }
        )
    return pd.DataFrame(rows).sort_values(["representation", "split", "layer"])


def evaluate_frozen_operating_points(
    embeddings: dict[tuple[int, str], np.ndarray],
    manifest: pd.DataFrame,
    selection: dict[str, Any],
    calibration_split: str,
    evaluation_split: str,
    far_targets: list[float],
) -> dict[str, Any]:
    calibration_mask = manifest["split"].to_numpy() == calibration_split
    evaluation_mask = manifest["split"].to_numpy() == evaluation_split
    calibration_labels = manifest.loc[calibration_mask, "identity_id"].to_numpy()
    evaluation_labels = manifest.loc[evaluation_mask, "identity_id"].to_numpy()
    result: dict[str, Any] = {
        "calibration_split": calibration_split,
        "evaluation_split": evaluation_split,
        "representations": {},
    }
    for representation, record in selection["representations"].items():
        layer = int(record["selected_layer"])
        values = embeddings[(layer, representation)]
        result["representations"][representation] = {
            "selected_layer": layer,
            **frozen_threshold_metrics(
                values[calibration_mask],
                calibration_labels,
                values[evaluation_mask],
                evaluation_labels,
                far_targets,
            ),
        }
    return result


def select_layers(
    metrics: pd.DataFrame, selection_split: str, selection_metric: str
) -> dict[str, Any]:
    selection_rows = metrics[metrics["split"] == selection_split]
    if selection_metric not in selection_rows.columns:
        raise ValueError(f"Unknown selection metric: {selection_metric}")
    result: dict[str, Any] = {
        "selection_split": selection_split,
        "selection_metric": selection_metric,
        "representations": {},
    }
    for representation, group in selection_rows.groupby("representation"):
        winner = group.loc[group[selection_metric].idxmax()]
        layer = int(winner["layer"])
        evidence = metrics[
            (metrics["representation"] == representation) & (metrics["layer"] == layer)
        ]
        result["representations"][representation] = {
            "selected_layer": layer,
            "metrics_by_split": evidence.set_index("split").to_dict(orient="index"),
        }
    return result
