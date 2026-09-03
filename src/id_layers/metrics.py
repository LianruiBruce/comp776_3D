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


def _template_query_scores(
    embeddings: np.ndarray, manifest: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    values = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    if len(values) != len(manifest):
        raise ValueError("Template/query manifest and embedding lengths differ")
    if "protocol_role" not in manifest:
        raise ValueError("Template/query evaluation requires protocol_role")

    labels = manifest["identity_id"].to_numpy()
    roles = manifest["protocol_role"].to_numpy()
    identity_order = sorted(np.unique(labels).tolist())
    templates: list[np.ndarray] = []
    template_labels: list[str] = []
    missing_reference_identities = 0
    for identity in identity_order:
        reference_mask = (labels == identity) & (roles == "reference")
        if not reference_mask.any():
            missing_reference_identities += 1
            continue
        template = values[reference_mask].mean(axis=0, keepdims=True)
        templates.append(_l2_normalize(template)[0])
        template_labels.append(str(identity))
    if len(templates) < 2:
        raise ValueError("Template/query evaluation requires at least two identity templates")

    template_array = np.stack(templates).astype(np.float32, copy=False)
    template_label_array = np.asarray(template_labels)
    query_mask = roles == "query"
    query_values = values[query_mask]
    query_labels = labels[query_mask].astype(str)
    has_template = np.isin(query_labels, template_label_array)
    excluded_queries = int((~has_template).sum())
    query_values = query_values[has_template]
    query_labels = query_labels[has_template]
    if len(query_values) == 0:
        raise ValueError("No aligned queries have a matching reference template")

    similarities = query_values @ template_array.T
    matches = query_labels[:, None] == template_label_array[None, :]
    positive = similarities[matches]
    negative = similarities[~matches]
    nearest = similarities.argmax(axis=1)
    recall_k = min(5, similarities.shape[1])
    top_k = np.argsort(-similarities, axis=1, kind="stable")[:, :recall_k]
    recall_at_5 = matches[np.arange(len(matches))[:, None], top_k].any(axis=1)
    metadata = {
        "template_identities": int(len(template_labels)),
        "template_queries": int(len(query_values)),
        "template_missing_reference_identities": int(missing_reference_identities),
        "template_excluded_queries_no_template": excluded_queries,
        "template_recall_at_5": float(recall_at_5.mean()),
    }
    return positive, negative, similarities, query_labels == template_label_array[nearest], metadata


def template_query_metrics(
    embeddings: np.ndarray, manifest: pd.DataFrame, far_targets: list[float]
) -> dict[str, Any]:
    positive, negative, _, correct, metadata = _template_query_scores(embeddings, manifest)
    pair_scores = np.concatenate([positive, negative])
    pair_labels = np.concatenate(
        [np.ones(positive.size, dtype=bool), np.zeros(negative.size, dtype=bool)]
    )
    fpr, tpr, thresholds = roc_curve(pair_labels.astype(np.int32), pair_scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    pooled_std = np.sqrt(0.5 * (positive.var(ddof=1) + negative.var(ddof=1)))
    result: dict[str, Any] = {
        **metadata,
        "template_positive_pairs": int(positive.size),
        "template_negative_pairs": int(negative.size),
        "template_positive_cosine_mean": float(positive.mean()),
        "template_positive_cosine_std": float(positive.std(ddof=1)),
        "template_negative_cosine_mean": float(negative.mean()),
        "template_negative_cosine_std": float(negative.std(ddof=1)),
        "template_cosine_gap": float(positive.mean() - negative.mean()),
        "template_d_prime": float(
            (positive.mean() - negative.mean()) / max(pooled_std, 1e-12)
        ),
        "template_roc_auc": float(roc_auc_score(pair_labels.astype(np.int32), pair_scores)),
        "template_eer": float(0.5 * (fpr[eer_index] + fnr[eer_index])),
        "template_eer_threshold": float(thresholds[eer_index]),
        "template_rank1": float(correct.mean()),
    }
    for target in far_targets:
        eligible = tpr[fpr <= target]
        result[f"template_tar_at_far_{target:g}"] = (
            float(eligible.max()) if eligible.size else 0.0
        )
    return result


def template_query_score_frame(
    embeddings: np.ndarray, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    positive, negative, _, correct, metadata = _template_query_scores(embeddings, manifest)
    reference_identities = set(
        manifest.loc[manifest["protocol_role"] == "reference", "identity_id"].astype(str)
    )
    query_frame = manifest[
        (manifest["protocol_role"] == "query")
        & manifest["identity_id"].astype(str).isin(reference_identities)
    ].copy()
    query_frame = query_frame.reset_index(drop=True)
    if len(query_frame) != len(positive):
        raise RuntimeError("Template positive scores do not align with query records")
    query_frame["positive_score"] = positive
    query_frame["rank1_correct"] = correct
    return query_frame, negative, metadata


def evaluate_selected_layers_by_condition(
    embeddings: dict[tuple[int, str], np.ndarray],
    manifest: pd.DataFrame,
    selection: dict[str, Any],
    splits: list[str],
    far_targets: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in splits:
        split_mask = manifest["split"].to_numpy() == split
        split_manifest = manifest.loc[split_mask].reset_index(drop=True)
        query_conditions = sorted(
            split_manifest.loc[
                split_manifest["protocol_role"] == "query", "condition"
            ].unique()
        )
        for representation, record in selection["representations"].items():
            layer = int(record["selected_layer"])
            split_values = embeddings[(layer, representation)][split_mask]
            for condition in query_conditions:
                condition_mask = (
                    (split_manifest["protocol_role"].to_numpy() == "reference")
                    | (
                        (split_manifest["protocol_role"].to_numpy() == "query")
                        & (split_manifest["condition"].to_numpy() == condition)
                    )
                )
                condition_manifest = split_manifest.loc[condition_mask].reset_index(drop=True)
                condition_values = split_values[condition_mask]
                rows.append(
                    {
                        "split": split,
                        "representation": representation,
                        "layer": layer,
                        "query_condition": condition,
                        **template_query_metrics(
                            condition_values, condition_manifest, far_targets
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["representation", "split", "query_condition"]
    )


def evaluate_frozen_selected_layers_by_condition(
    embeddings: dict[tuple[int, str], np.ndarray],
    manifest: pd.DataFrame,
    selection: dict[str, Any],
    calibration_split: str,
    evaluation_split: str,
    far_targets: list[float],
) -> pd.DataFrame:
    calibration_mask = manifest["split"].to_numpy() == calibration_split
    evaluation_mask = manifest["split"].to_numpy() == evaluation_split
    calibration_manifest = manifest.loc[calibration_mask].reset_index(drop=True)
    evaluation_manifest = manifest.loc[evaluation_mask].reset_index(drop=True)
    query_conditions = sorted(
        evaluation_manifest.loc[
            evaluation_manifest["protocol_role"] == "query", "condition"
        ].unique()
    )
    rows: list[dict[str, Any]] = []
    for representation, record in selection["representations"].items():
        layer = int(record["selected_layer"])
        calibration_values = embeddings[(layer, representation)][calibration_mask]
        _, calibration_negative, _ = template_query_score_frame(
            calibration_values, calibration_manifest
        )
        evaluation_values = embeddings[(layer, representation)][evaluation_mask]
        for target_far in far_targets:
            threshold = _threshold_from_negative_scores(calibration_negative, target_far)
            for condition in query_conditions:
                condition_mask = (
                    (evaluation_manifest["protocol_role"].to_numpy() == "reference")
                    | (
                        (evaluation_manifest["protocol_role"].to_numpy() == "query")
                        & (evaluation_manifest["condition"].to_numpy() == condition)
                    )
                )
                condition_frame, condition_negative, _ = template_query_score_frame(
                    evaluation_values[condition_mask],
                    evaluation_manifest.loc[condition_mask].reset_index(drop=True),
                )
                positive = condition_frame["positive_score"].to_numpy(dtype=np.float64)
                negative = condition_negative.astype(np.float64, copy=False)
                rows.append(
                    {
                        "split": evaluation_split,
                        "representation": representation,
                        "layer": layer,
                        "query_condition": condition,
                        "target_far": float(target_far),
                        "frozen_threshold": threshold,
                        "frozen_observed_far": float((negative >= threshold).mean()),
                        "frozen_tar": float((positive >= threshold).mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["representation", "query_condition", "target_far"]
    )


def _identity_means(frame: pd.DataFrame, value_column: str) -> pd.Series:
    return frame.groupby("identity_id", sort=True)[value_column].mean().astype(np.float64)


def bootstrap_selected_template_tar(
    embeddings: dict[tuple[int, str], np.ndarray],
    manifest: pd.DataFrame,
    selection: dict[str, Any],
    calibration_split: str,
    evaluation_split: str,
    far_targets: list[float],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    calibration_mask = manifest["split"].to_numpy() == calibration_split
    evaluation_mask = manifest["split"].to_numpy() == evaluation_split
    calibration_manifest = manifest.loc[calibration_mask].reset_index(drop=True)
    evaluation_manifest = manifest.loc[evaluation_mask].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for representation, record in sorted(selection["representations"].items()):
        selected_layer = int(record["selected_layer"])
        final_layer = max(
            layer
            for layer, candidate_representation in embeddings
            if candidate_representation == representation
        )
        layer_scores: dict[int, dict[str, Any]] = {}
        for layer in sorted({selected_layer, final_layer}):
            calibration_frame, calibration_negative, _ = template_query_score_frame(
                embeddings[(layer, representation)][calibration_mask], calibration_manifest
            )
            evaluation_frame, evaluation_negative, _ = template_query_score_frame(
                embeddings[(layer, representation)][evaluation_mask], evaluation_manifest
            )
            layer_scores[layer] = {
                "calibration_frame": calibration_frame,
                "calibration_negative": calibration_negative.astype(np.float64, copy=False),
                "evaluation_frame": evaluation_frame,
                "evaluation_negative": evaluation_negative.astype(np.float64, copy=False),
            }

        for target_far in far_targets:
            identity_values: dict[int, pd.Series] = {}
            thresholds: dict[int, float] = {}
            pooled_tars: dict[int, float] = {}
            pooled_fars: dict[int, float] = {}
            for layer, score_data in layer_scores.items():
                threshold = _threshold_from_negative_scores(
                    score_data["calibration_negative"], target_far
                )
                evaluation_frame = score_data["evaluation_frame"].copy()
                evaluation_frame["accepted"] = (
                    evaluation_frame["positive_score"].astype(np.float64) >= threshold
                ).astype(np.float64)
                identity_values[layer] = _identity_means(evaluation_frame, "accepted")
                thresholds[layer] = threshold
                pooled_tars[layer] = float(evaluation_frame["accepted"].mean())
                pooled_fars[layer] = float(
                    (score_data["evaluation_negative"] >= threshold).mean()
                )

            common_identities = identity_values[selected_layer].index.intersection(
                identity_values[final_layer].index
            )
            selected_values = identity_values[selected_layer].loc[common_identities].to_numpy()
            final_values = identity_values[final_layer].loc[common_identities].to_numpy()
            draw_indices = rng.integers(
                0, len(common_identities), size=(resamples, len(common_identities))
            )
            selected_draws = selected_values[draw_indices].mean(axis=1)
            final_draws = final_values[draw_indices].mean(axis=1)
            comparisons = [
                ("selected", selected_layer, selected_values, selected_draws),
                ("final", final_layer, final_values, final_draws),
                (
                    "selected_minus_final",
                    selected_layer,
                    selected_values - final_values,
                    selected_draws - final_draws,
                ),
            ]
            for comparison, layer, values, draws in comparisons:
                rows.append(
                    {
                        "representation": representation,
                        "comparison": comparison,
                        "selected_layer": selected_layer,
                        "final_layer": final_layer,
                        "reported_layer": layer,
                        "calibration_split": calibration_split,
                        "evaluation_split": evaluation_split,
                        "target_far": float(target_far),
                        "identity_count": int(len(common_identities)),
                        "identity_mean_estimate": float(values.mean()),
                        "bootstrap_ci_lower_95": float(np.quantile(draws, 0.025)),
                        "bootstrap_ci_upper_95": float(np.quantile(draws, 0.975)),
                        "bootstrap_resamples": int(resamples),
                        "bootstrap_seed": int(seed),
                        "selected_threshold": thresholds[selected_layer],
                        "final_threshold": thresholds[final_layer],
                        "selected_pooled_tar": pooled_tars[selected_layer],
                        "final_pooled_tar": pooled_tars[final_layer],
                        "selected_observed_far": pooled_fars[selected_layer],
                        "final_observed_far": pooled_fars[final_layer],
                    }
                )
    return pd.DataFrame(rows).sort_values(["representation", "comparison"])


def frozen_template_threshold_metrics(
    calibration_embeddings: np.ndarray,
    calibration_manifest: pd.DataFrame,
    evaluation_embeddings: np.ndarray,
    evaluation_manifest: pd.DataFrame,
    far_targets: list[float],
) -> dict[str, Any]:
    calibration_positive, calibration_negative, _, _, calibration_metadata = (
        _template_query_scores(calibration_embeddings, calibration_manifest)
    )
    evaluation_positive, evaluation_negative, _, _, evaluation_metadata = (
        _template_query_scores(evaluation_embeddings, evaluation_manifest)
    )
    calibration_positive = calibration_positive.astype(np.float64, copy=False)
    calibration_negative = calibration_negative.astype(np.float64, copy=False)
    evaluation_positive = evaluation_positive.astype(np.float64, copy=False)
    evaluation_negative = evaluation_negative.astype(np.float64, copy=False)
    operating_points = {}
    for target in far_targets:
        threshold = _threshold_from_negative_scores(calibration_negative, target)
        operating_points[f"far_{target:g}"] = {
            "target_far": float(target),
            "threshold": threshold,
            "calibration_far": float((calibration_negative >= threshold).mean()),
            "calibration_tar": float((calibration_positive >= threshold).mean()),
            "evaluation_far": float((evaluation_negative >= threshold).mean()),
            "evaluation_tar": float((evaluation_positive >= threshold).mean()),
        }
    return {
        "calibration": calibration_metadata,
        "evaluation": evaluation_metadata,
        "operating_points": operating_points,
    }


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
            split_manifest = manifest.loc[mask].reset_index(drop=True)
            metrics = verification_metrics(values[mask], labels, far_targets)
            if "protocol_role" in split_manifest:
                metrics.update(
                    template_query_metrics(values[mask], split_manifest, far_targets)
                )
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
        metrics = verification_metrics(values[mask], labels, far_targets)
        split_manifest = manifest.loc[mask].reset_index(drop=True)
        if "protocol_role" in split_manifest:
            metrics.update(template_query_metrics(values[mask], split_manifest, far_targets))
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
        representation_result = {
            "selected_layer": layer,
            **frozen_threshold_metrics(
                values[calibration_mask],
                calibration_labels,
                values[evaluation_mask],
                evaluation_labels,
                far_targets,
            ),
        }
        if "protocol_role" in manifest:
            representation_result["template_protocol"] = frozen_template_threshold_metrics(
                values[calibration_mask],
                manifest.loc[calibration_mask].reset_index(drop=True),
                values[evaluation_mask],
                manifest.loc[evaluation_mask].reset_index(drop=True),
                far_targets,
            )
        result["representations"][representation] = representation_result
    return result


def select_layers(
    metrics: pd.DataFrame,
    selection_split: str,
    selection_metric: str,
    tiebreak_metrics: list[str] | None = None,
) -> dict[str, Any]:
    selection_rows = metrics[metrics["split"] == selection_split]
    if selection_metric not in selection_rows.columns:
        raise ValueError(f"Unknown selection metric: {selection_metric}")
    tiebreak_metrics = list(tiebreak_metrics or [])
    missing_tiebreakers = [
        metric for metric in tiebreak_metrics if metric not in selection_rows.columns
    ]
    if missing_tiebreakers:
        raise ValueError(f"Unknown selection tiebreak metrics: {missing_tiebreakers}")
    result: dict[str, Any] = {
        "selection_split": selection_split,
        "selection_metric": selection_metric,
        "selection_tiebreak_metrics": tiebreak_metrics,
        "representations": {},
    }
    for representation, group in selection_rows.groupby("representation"):
        sort_columns = [selection_metric, *tiebreak_metrics, "layer"]
        ascending = [False] * (1 + len(tiebreak_metrics)) + [True]
        winner = group.sort_values(sort_columns, ascending=ascending).iloc[0]
        layer = int(winner["layer"])
        evidence = metrics[
            (metrics["representation"] == representation) & (metrics["layer"] == layer)
        ]
        result["representations"][representation] = {
            "selected_layer": layer,
            "metrics_by_split": evidence.set_index("split").to_dict(orient="index"),
        }
    return result
