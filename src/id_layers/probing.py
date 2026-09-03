from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from .modeling import non_affine_ln_token_mean

# Backward-compatible spelling from the initial probing API. The implementation lives
# in modeling so extraction and probe-side checks cannot silently diverge.
nonaffine_layernorm_token_mean = non_affine_ln_token_mean


class EqualCapacityAngularProbe(nn.Module):
    """A matched open-set readout for a cacheable normalized token mean.

    ``layer_norm_weight`` and ``layer_norm_bias`` are the trainable affine parameters
    of a fresh per-layer LayerNorm. The statistics-dependent, non-affine part of that
    LayerNorm must already have been computed by
    :func:`nonaffine_layernorm_token_mean`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        *,
        normalization_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Probe dimensions must be positive")
        if normalization_eps <= 0:
            raise ValueError("Normalization epsilon must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.normalization_eps = float(normalization_eps)
        self.layer_norm_weight = nn.Parameter(torch.ones(input_dim))
        self.layer_norm_bias = nn.Parameter(torch.zeros(input_dim))
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.layer_norm_weight)
        nn.init.zeros_(self.layer_norm_bias)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)

    def forward(self, cached_nonaffine_mean: torch.Tensor) -> torch.Tensor:
        if cached_nonaffine_mean.shape[-1] != self.input_dim:
            raise ValueError(
                "Probe input width differs from input_dim: "
                f"{cached_nonaffine_mean.shape[-1]} != {self.input_dim}"
            )
        affine = (
            cached_nonaffine_mean * self.layer_norm_weight + self.layer_norm_bias
        )
        projected = self.projection(affine)
        return F.normalize(projected, p=2, dim=-1, eps=self.normalization_eps)


class CosFaceClassifier(nn.Module):
    """Training-only normalized classifier with an additive cosine margin."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        scale: float = 64.0,
        margin: float = 0.4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_classes <= 1:
            raise ValueError("CosFace needs a positive width and at least two classes")
        if scale <= 0:
            raise ValueError("CosFace scale must be positive")
        if margin < 0:
            raise ValueError("CosFace margin must be non-negative")
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -1.0, 1.0)

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("CosFace embeddings must have shape [batch, embedding_dim]")
        return F.linear(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        ).clamp(-1.0, 1.0)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        cosine = self.cosine_logits(embeddings)
        if labels is not None:
            if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
                raise ValueError("CosFace labels must have shape [batch]")
            if labels.dtype != torch.long:
                raise TypeError("CosFace labels must use torch.long")
            if bool(((labels < 0) | (labels >= self.num_classes)).any()):
                raise ValueError("CosFace labels contain an out-of-range class")
            target_mask = F.one_hot(labels, num_classes=self.num_classes).to(
                dtype=cosine.dtype
            )
            cosine = cosine - target_mask * self.margin
        return cosine * self.scale

    def loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self(embeddings, labels), labels)


def build_matched_probe(
    input_dim: int,
    output_dim: int,
    num_classes: int,
    *,
    seed: int,
    scale: float = 64.0,
    margin: float = 0.4,
) -> tuple[EqualCapacityAngularProbe, CosFaceClassifier]:
    """Build a probe/classifier pair without changing the caller's CPU RNG state.

    Calling this function with the same seed and dimensions for every layer provides
    matched initialization for paired layer comparisons.
    """

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        probe = EqualCapacityAngularProbe(input_dim, output_dim)
        classifier = CosFaceClassifier(
            output_dim, num_classes, scale=scale, margin=margin
        )
    return probe, classifier


def iter_deterministic_batches(
    sample_count: int,
    batch_size: int,
    epochs: int,
    *,
    seed: int,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield matched shuffled CPU indices for each epoch."""

    if sample_count <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("sample_count, batch_size and epochs must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for epoch in range(epochs):
        permutation = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, batch_size):
            yield epoch, permutation[start : start + batch_size]


@dataclass(frozen=True)
class ProbeTrainingConfig:
    output_dim: int = 512
    epochs: int = 120
    batch_size: int = 128
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    polynomial_power: float = 1.0
    scale: float = 64.0
    margin: float = 0.4
    device: str = "cuda"
    deterministic: bool = True

    def validate(self) -> None:
        if self.output_dim <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Output width, epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.polynomial_power <= 0:
            raise ValueError("Polynomial power must be positive")
        if self.scale <= 0 or self.margin < 0:
            raise ValueError("Invalid CosFace hyperparameters")


@dataclass(frozen=True)
class EpochTrainingRecord:
    epoch: int
    mean_loss: float
    train_accuracy: float
    learning_rate_after_epoch: float


@dataclass
class ProbeTrainingResult:
    probe: EqualCapacityAngularProbe
    classifier: CosFaceClassifier
    class_labels: tuple[str, ...]
    seed: int
    history: tuple[EpochTrainingRecord, ...]
    config: dict[str, Any]
    optimizer_steps: int


def _encoded_labels(labels: Sequence[Any]) -> tuple[tuple[str, ...], torch.Tensor]:
    text_labels = np.asarray([str(value) for value in labels], dtype=str)
    classes, encoded = np.unique(text_labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Probe training requires at least two identity classes")
    return tuple(classes.tolist()), torch.from_numpy(encoded.astype(np.int64, copy=False))


def _feature_tensor(features: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        result = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
    else:
        result = torch.from_numpy(np.asarray(features, dtype=np.float32)).contiguous()
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("Cached probe features must have shape [sample, channel]")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Cached probe features contain non-finite values")
    return result


def train_equal_capacity_probe(
    features: np.ndarray | torch.Tensor,
    labels: Sequence[Any],
    *,
    seed: int,
    config: ProbeTrainingConfig,
) -> ProbeTrainingResult:
    """Train one fixed-epoch angular probe on cached frozen-backbone features."""

    config.validate()
    values = _feature_tensor(features)
    if len(labels) != len(values):
        raise ValueError("Feature and label counts differ")
    class_labels, encoded_cpu = _encoded_labels(labels)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe training requested but CUDA is unavailable")

    probe, classifier = build_matched_probe(
        values.shape[1],
        config.output_dim,
        len(class_labels),
        seed=seed,
        scale=config.scale,
        margin=config.margin,
    )
    probe = probe.to(device).train()
    classifier = classifier.to(device).train()
    values = values.to(device)
    encoded = encoded_cpu.to(device)
    optimizer = torch.optim.AdamW(
        [*probe.parameters(), *classifier.parameters()],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    batches_per_epoch = math.ceil(len(values) / config.batch_size)
    total_steps = batches_per_epoch * config.epochs
    records: list[EpochTrainingRecord] = []
    completed_steps = 0
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_samples = 0
    current_epoch = 0

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    if config.deterministic:
        torch.use_deterministic_algorithms(True)
    try:
        for epoch, indices_cpu in iter_deterministic_batches(
            len(values), config.batch_size, config.epochs, seed=seed
        ):
            if epoch != current_epoch:
                records.append(
                    EpochTrainingRecord(
                        epoch=current_epoch + 1,
                        mean_loss=epoch_loss / epoch_samples,
                        train_accuracy=epoch_correct / epoch_samples,
                        learning_rate_after_epoch=float(optimizer.param_groups[0]["lr"]),
                    )
                )
                current_epoch = epoch
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_samples = 0

            indices = indices_cpu.to(device)
            batch_values = values.index_select(0, indices)
            batch_labels = encoded.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            embeddings = probe(batch_values)
            logits = classifier(embeddings, batch_labels)
            loss = F.cross_entropy(logits, batch_labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite CosFace training loss")
            loss.backward()
            optimizer.step()

            completed_steps += 1
            decay = max(0.0, 1.0 - completed_steps / total_steps)
            learning_rate = config.learning_rate * decay**config.polynomial_power
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            batch_count = int(len(indices_cpu))
            epoch_loss += float(loss.detach().item()) * batch_count
            epoch_correct += int((logits.detach().argmax(dim=1) == batch_labels).sum().item())
            epoch_samples += batch_count

        records.append(
            EpochTrainingRecord(
                epoch=current_epoch + 1,
                mean_loss=epoch_loss / epoch_samples,
                train_accuracy=epoch_correct / epoch_samples,
                learning_rate_after_epoch=float(optimizer.param_groups[0]["lr"]),
            )
        )
    finally:
        if config.deterministic and not deterministic_before:
            torch.use_deterministic_algorithms(False)

    probe = probe.cpu().eval()
    classifier = classifier.cpu().eval()
    return ProbeTrainingResult(
        probe=probe,
        classifier=classifier,
        class_labels=class_labels,
        seed=seed,
        history=tuple(records),
        config=asdict(config),
        optimizer_steps=completed_steps,
    )


def assign_identity_folds(
    identity_ids: Sequence[Any], *, n_splits: int = 5, seed: int
) -> dict[str, int]:
    """Assign unique identities to deterministic, balanced folds."""

    if n_splits < 2:
        raise ValueError("Cross-fitting requires at least two folds")
    identities = sorted({str(value) for value in identity_ids})
    if len(identities) < n_splits:
        raise ValueError("There are fewer identities than folds")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(identities))
    result: dict[str, int] = {}
    for order, identity_index in enumerate(permutation):
        result[identities[int(identity_index)]] = order % n_splits
    return result


def calibrate_far_threshold(negative_scores: np.ndarray, target_far: float) -> float:
    """Choose the least strict observed threshold whose FAR does not exceed target."""

    if not 0 <= target_far <= 1:
        raise ValueError("target_far must be in [0, 1]")
    negative = np.asarray(negative_scores, dtype=np.float64)
    if negative.ndim != 1 or negative.size == 0:
        raise ValueError("At least one one-dimensional negative score is required")
    if not np.isfinite(negative).all():
        raise ValueError("Negative scores contain non-finite values")
    unique = np.unique(negative)
    candidates = np.concatenate(
        [np.asarray([np.nextafter(float(unique.max()), np.inf)]), unique]
    )
    fars = np.asarray([(negative >= threshold).mean() for threshold in candidates])
    eligible = np.flatnonzero(fars <= target_far)
    best_far = fars[eligible].max()
    return float(candidates[eligible[fars[eligible] == best_far]].min())


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _template_query_scores(
    embeddings: np.ndarray,
    identities: np.ndarray,
    roles: np.ndarray,
    included_identities: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isin(identities, sorted(included_identities))
    values = _l2_normalize(embeddings[mask])
    subset_identities = identities[mask]
    subset_roles = roles[mask]
    ordered_identities = sorted(included_identities)
    templates: list[np.ndarray] = []
    for identity in ordered_identities:
        reference_mask = (subset_identities == identity) & (subset_roles == "reference")
        if not reference_mask.any():
            raise ValueError(f"Identity {identity!r} has no reference embedding")
        templates.append(_l2_normalize(values[reference_mask].mean(axis=0)[None, :])[0])
    query_mask = subset_roles == "query"
    query_values = values[query_mask]
    query_identities = subset_identities[query_mask]
    missing_query = sorted(included_identities - set(query_identities.tolist()))
    if missing_query:
        raise ValueError(f"Identities have no usable query embedding: {missing_query}")
    template_array = np.stack(templates)
    similarity = query_values @ template_array.T
    matches = query_identities[:, None] == np.asarray(ordered_identities)[None, :]
    return similarity[matches], similarity[~matches], query_identities


@dataclass(frozen=True)
class FoldOperatingPoint:
    fold: int
    threshold: float
    calibration_negative_pairs: int
    evaluation_positive_pairs: int
    evaluation_negative_pairs: int
    tar: float
    observed_far: float


@dataclass(frozen=True)
class CrossFittedTarResult:
    target_far: float
    tar: float
    observed_far: float
    d_prime: float
    roc_auc: float
    positive_pairs: int
    negative_pairs: int
    identity_tar: dict[str, float]
    folds: tuple[FoldOperatingPoint, ...]


def cross_fitted_template_tar(
    embeddings: np.ndarray | torch.Tensor,
    identity_ids: Sequence[Any],
    protocol_roles: Sequence[str],
    fold_by_identity: Mapping[str, int],
    *,
    target_far: float = 0.01,
) -> CrossFittedTarResult:
    """Estimate dev TAR with identity-disjoint threshold cross-fitting."""

    if isinstance(embeddings, torch.Tensor):
        values = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        values = np.asarray(embeddings, dtype=np.float32)
    identities = np.asarray([str(value) for value in identity_ids])
    roles = np.asarray(protocol_roles)
    if values.ndim != 2 or len(values) != len(identities) or len(values) != len(roles):
        raise ValueError("Embeddings, identities and roles must have matching sample counts")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain non-finite values")
    if not set(roles).issubset({"reference", "query"}):
        raise ValueError("Protocol roles must be 'reference' or 'query'")

    identity_set = set(identities.tolist())
    normalized_folds = {str(key): int(value) for key, value in fold_by_identity.items()}
    missing = identity_set - set(normalized_folds)
    if missing:
        raise ValueError(f"Fold assignment is missing identities: {sorted(missing)}")
    fold_values = sorted({normalized_folds[identity] for identity in identity_set})
    if len(fold_values) < 2:
        raise ValueError("Cross-fitting requires at least two populated folds")

    all_positive: list[np.ndarray] = []
    all_negative: list[np.ndarray] = []
    all_negative_accepted: list[np.ndarray] = []
    identity_acceptance: dict[str, list[float]] = {}
    fold_records: list[FoldOperatingPoint] = []
    for fold in fold_values:
        evaluation_identities = {
            identity for identity in identity_set if normalized_folds[identity] == fold
        }
        calibration_identities = identity_set - evaluation_identities
        _, calibration_negative, _ = _template_query_scores(
            values, identities, roles, calibration_identities
        )
        threshold = calibrate_far_threshold(calibration_negative, target_far)
        positive, negative, query_identities = _template_query_scores(
            values, identities, roles, evaluation_identities
        )
        positive = positive.astype(np.float64, copy=False)
        negative = negative.astype(np.float64, copy=False)
        positive_accepted = positive >= threshold
        negative_accepted = negative >= threshold
        for identity, accepted in zip(query_identities, positive_accepted, strict=True):
            identity_acceptance.setdefault(str(identity), []).append(float(accepted))
        all_positive.append(positive)
        all_negative.append(negative)
        all_negative_accepted.append(negative_accepted)
        fold_records.append(
            FoldOperatingPoint(
                fold=fold,
                threshold=threshold,
                calibration_negative_pairs=int(calibration_negative.size),
                evaluation_positive_pairs=int(positive.size),
                evaluation_negative_pairs=int(negative.size),
                tar=float(positive_accepted.mean()),
                observed_far=float(negative_accepted.mean()),
            )
        )

    positive_scores = np.concatenate(all_positive).astype(np.float64, copy=False)
    negative_scores = np.concatenate(all_negative).astype(np.float64, copy=False)
    negative_accepted = np.concatenate(all_negative_accepted)
    pooled_std = np.sqrt(
        0.5 * (positive_scores.var(ddof=1) + negative_scores.var(ddof=1))
    )
    binary = np.concatenate(
        [
            np.ones(positive_scores.size, dtype=np.int8),
            np.zeros(negative_scores.size, dtype=np.int8),
        ]
    )
    scores = np.concatenate([positive_scores, negative_scores])
    return CrossFittedTarResult(
        target_far=float(target_far),
        tar=float(
            np.mean(
                [accepted for values_ in identity_acceptance.values() for accepted in values_]
            )
        ),
        observed_far=float(negative_accepted.mean()),
        d_prime=float(
            (positive_scores.mean() - negative_scores.mean()) / max(pooled_std, 1e-12)
        ),
        roc_auc=float(roc_auc_score(binary, scores)),
        positive_pairs=int(positive_scores.size),
        negative_pairs=int(negative_scores.size),
        identity_tar={
            identity: float(np.mean(accepted))
            for identity, accepted in sorted(identity_acceptance.items())
        },
        folds=tuple(fold_records),
    )


@dataclass(frozen=True)
class LayerSelectionResult:
    selected_layer: int
    summary: pd.DataFrame
    seed_winners: dict[int, int]
    primary_metric: str
    tiebreak_metrics: tuple[str, ...]


def aggregate_and_select_layer(
    metrics: pd.DataFrame,
    *,
    primary_metric: str = "cross_fitted_tar",
    tiebreak_metrics: Sequence[str] = ("d_prime", "roc_auc"),
    required_seed_count: int = 5,
) -> LayerSelectionResult:
    """Select one global layer from matched seed-level dev measurements."""

    required = {"layer", "seed", primary_metric, *tiebreak_metrics}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Layer metrics are missing columns: {sorted(missing)}")
    if required_seed_count <= 0:
        raise ValueError("required_seed_count must be positive")
    if metrics.duplicated(["layer", "seed"]).any():
        raise ValueError("Each layer/seed combination must appear exactly once")
    score_columns = [primary_metric, *tiebreak_metrics]
    if not np.isfinite(metrics[score_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Layer-selection metrics contain non-finite values")

    expected_seeds: set[int] | None = None
    for layer, group in metrics.groupby("layer", sort=True):
        seeds = {int(seed) for seed in group["seed"]}
        if len(seeds) != required_seed_count:
            raise ValueError(
                f"Layer {layer} has {len(seeds)} seeds; expected {required_seed_count}"
            )
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError("Every layer must use the same matched seed set")

    aggregations: dict[str, list[str]] = {
        column: ["mean", "std"] for column in score_columns
    }
    summary = metrics.groupby("layer", as_index=False).agg(aggregations)
    summary.columns = [
        str(column) if not statistic else f"{column}_{statistic}"
        for column, statistic in summary.columns
    ]
    mean_columns = [f"{column}_mean" for column in score_columns]
    ranking = summary.sort_values(
        [*mean_columns, "layer"],
        ascending=[False] * len(mean_columns) + [True],
        kind="mergesort",
    )
    selected_layer = int(ranking.iloc[0]["layer"])

    seed_winners: dict[int, int] = {}
    for seed, group in metrics.groupby("seed", sort=True):
        ranked_seed = group.sort_values(
            [*score_columns, "layer"],
            ascending=[False] * len(score_columns) + [True],
            kind="mergesort",
        )
        seed_winners[int(seed)] = int(ranked_seed.iloc[0]["layer"])
    return LayerSelectionResult(
        selected_layer=selected_layer,
        summary=summary.sort_values("layer").reset_index(drop=True),
        seed_winners=seed_winners,
        primary_metric=primary_metric,
        tiebreak_metrics=tuple(tiebreak_metrics),
    )


@dataclass(frozen=True)
class PairedIdentityBootstrapResult:
    selected_mean_tar: float
    final_mean_tar: float
    selected_minus_final: float
    ci_lower_95: float
    ci_upper_95: float
    identity_count: int
    seed_count: int
    seed_delta_mean: float
    seed_delta_std: float
    resamples: int
    bootstrap_seed: int


def bootstrap_selected_vs_final_identity_tar(
    selected: pd.DataFrame,
    final: pd.DataFrame,
    *,
    identity_column: str = "identity_id",
    seed_column: str = "seed",
    value_column: str = "tar",
    resamples: int = 1000,
    bootstrap_seed: int,
    required_seed_count: int = 5,
) -> PairedIdentityBootstrapResult:
    """Bootstrap a paired selected-vs-final TAR difference by identity.

    Inputs contain one already-aggregated TAR value per identity and probe seed. Seed
    variation is reported separately; the confidence interval resamples identities,
    never individual verification pairs.
    """

    required = {identity_column, seed_column, value_column}
    for name, frame in (("selected", selected), ("final", final)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} frame is missing columns: {sorted(missing)}")
        if frame.duplicated([identity_column, seed_column]).any():
            raise ValueError(f"{name} has duplicate identity/seed measurements")
        values = frame[value_column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or bool(((values < 0) | (values > 1)).any()):
            raise ValueError(f"{name} TAR values must be finite and in [0, 1]")
    if resamples <= 0 or required_seed_count <= 0:
        raise ValueError("resamples and required_seed_count must be positive")

    selected_indexed = selected.set_index([identity_column, seed_column]).sort_index()
    final_indexed = final.set_index([identity_column, seed_column]).sort_index()
    if not selected_indexed.index.equals(final_indexed.index):
        raise ValueError("Selected and final measurements must have identical paired keys")
    seeds = sorted(selected[seed_column].unique().tolist())
    if len(seeds) != required_seed_count:
        raise ValueError(
            f"Bootstrap received {len(seeds)} seeds; expected {required_seed_count}"
        )
    counts = selected.groupby(identity_column)[seed_column].nunique()
    if not bool((counts == required_seed_count).all()):
        raise ValueError("Every identity must contain every matched probe seed")

    selected_pivot = selected.pivot(
        index=identity_column, columns=seed_column, values=value_column
    ).sort_index()
    final_pivot = final.pivot(
        index=identity_column, columns=seed_column, values=value_column
    ).reindex(index=selected_pivot.index, columns=selected_pivot.columns)
    selected_values = selected_pivot.to_numpy(dtype=np.float64)
    final_values = final_pivot.to_numpy(dtype=np.float64)
    identity_deltas = (selected_values - final_values).mean(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    draw_indices = rng.integers(
        0, len(identity_deltas), size=(resamples, len(identity_deltas))
    )
    draws = identity_deltas[draw_indices].mean(axis=1)
    seed_deltas = (selected_values - final_values).mean(axis=0)
    return PairedIdentityBootstrapResult(
        selected_mean_tar=float(selected_values.mean()),
        final_mean_tar=float(final_values.mean()),
        selected_minus_final=float(identity_deltas.mean()),
        ci_lower_95=float(np.quantile(draws, 0.025)),
        ci_upper_95=float(np.quantile(draws, 0.975)),
        identity_count=int(len(identity_deltas)),
        seed_count=int(len(seeds)),
        seed_delta_mean=float(seed_deltas.mean()),
        seed_delta_std=float(seed_deltas.std(ddof=1)) if len(seeds) > 1 else 0.0,
        resamples=int(resamples),
        bootstrap_seed=int(bootstrap_seed),
    )
