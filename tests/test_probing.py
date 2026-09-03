import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from id_layers.probing import (
    CosFaceClassifier,
    EqualCapacityAngularProbe,
    ProbeTrainingConfig,
    aggregate_and_select_layer,
    assign_identity_folds,
    bootstrap_selected_vs_final_identity_tar,
    build_matched_probe,
    cross_fitted_template_tar,
    iter_deterministic_batches,
    nonaffine_layernorm_token_mean,
    train_equal_capacity_probe,
)


def test_cached_nonaffine_mean_is_exact_fresh_layernorm_factorization() -> None:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(3, 5, 4, generator=generator)
    layer_norm = nn.LayerNorm(4)
    with torch.no_grad():
        layer_norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        layer_norm.bias.copy_(torch.tensor([-0.2, -0.1, 0.1, 0.2]))

    cached = nonaffine_layernorm_token_mean(tokens, eps=layer_norm.eps)
    factorized = cached * layer_norm.weight + layer_norm.bias
    direct = layer_norm(tokens).mean(dim=1)
    torch.testing.assert_close(factorized, direct, atol=1e-6, rtol=1e-6)


def test_equal_capacity_probe_is_bias_free_and_l2_normalized() -> None:
    probe = EqualCapacityAngularProbe(4, 6)
    values = torch.randn(8, 4)
    output = probe(values)
    assert probe.projection.bias is None
    assert output.shape == (8, 6)
    torch.testing.assert_close(output.norm(dim=1), torch.ones(8), atol=1e-6, rtol=1e-6)


def test_cosface_applies_margin_only_to_target_logit() -> None:
    classifier = CosFaceClassifier(2, 2, scale=10.0, margin=0.4)
    with torch.no_grad():
        classifier.weight.copy_(torch.eye(2))
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1], dtype=torch.long)
    logits = classifier(embeddings, labels)
    expected = torch.tensor([[6.0, 0.0], [0.0, 6.0]])
    torch.testing.assert_close(logits, expected)
    assert torch.isfinite(classifier.loss(embeddings, labels))


def test_matched_initialization_and_batch_order_are_deterministic() -> None:
    first_probe, first_classifier = build_matched_probe(4, 6, 3, seed=776)
    second_probe, second_classifier = build_matched_probe(4, 6, 3, seed=776)
    different_probe, _ = build_matched_probe(4, 6, 3, seed=1776)
    torch.testing.assert_close(first_probe.projection.weight, second_probe.projection.weight)
    torch.testing.assert_close(first_classifier.weight, second_classifier.weight)
    assert not torch.equal(first_probe.projection.weight, different_probe.projection.weight)

    first_batches = [
        (epoch, indices.tolist())
        for epoch, indices in iter_deterministic_batches(9, 4, 2, seed=776)
    ]
    second_batches = [
        (epoch, indices.tolist())
        for epoch, indices in iter_deterministic_batches(9, 4, 2, seed=776)
    ]
    assert first_batches == second_batches
    assert [len(indices) for _, indices in first_batches] == [4, 4, 1, 4, 4, 1]


def test_fixed_epoch_training_is_reproducible_and_reaches_zero_final_lr() -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ],
        dtype=np.float32,
    )
    labels = ["a", "a", "b", "b", "c", "c"]
    config = ProbeTrainingConfig(
        output_dim=4,
        epochs=4,
        batch_size=3,
        learning_rate=0.01,
        scale=16.0,
        margin=0.1,
        device="cpu",
    )
    first = train_equal_capacity_probe(features, labels, seed=11, config=config)
    second = train_equal_capacity_probe(features, labels, seed=11, config=config)
    assert first.history == second.history
    assert len(first.history) == config.epochs
    assert first.history[-1].learning_rate_after_epoch == 0.0
    assert first.optimizer_steps == 8
    for name, tensor in first.probe.state_dict().items():
        torch.testing.assert_close(tensor, second.probe.state_dict()[name], rtol=0, atol=0)


def test_identity_fold_assignment_is_balanced_and_deterministic() -> None:
    identities = [f"id_{index:02d}" for index in range(12)]
    first = assign_identity_folds(identities, n_splits=5, seed=20260902)
    second = assign_identity_folds(list(reversed(identities)), n_splits=5, seed=20260902)
    assert first == second
    counts = np.bincount(list(first.values()))
    assert counts.max() - counts.min() <= 1


def test_cross_fitted_tar_calibrates_on_other_identities() -> None:
    identity_count = 10
    rows: list[np.ndarray] = []
    identities: list[str] = []
    roles: list[str] = []
    for index in range(identity_count):
        basis = np.zeros(identity_count, dtype=np.float32)
        basis[index] = 1.0
        rows.extend([basis, basis.copy()])
        identities.extend([f"id_{index}", f"id_{index}"])
        roles.extend(["reference", "query"])
    folds = assign_identity_folds(identities, n_splits=5, seed=3)
    result = cross_fitted_template_tar(
        np.stack(rows), identities, roles, folds, target_far=0.0
    )
    assert result.tar == 1.0
    assert result.observed_far == 0.0
    assert result.roc_auc == 1.0
    assert result.positive_pairs == identity_count
    assert result.negative_pairs == identity_count
    assert set(result.identity_tar.values()) == {1.0}
    assert all(record.calibration_negative_pairs > 0 for record in result.folds)


def test_layer_selection_aggregates_matched_seeds_and_uses_tiebreaker() -> None:
    rows = []
    for seed in range(5):
        rows.extend(
            [
                {
                    "layer": 1,
                    "seed": seed,
                    "cross_fitted_tar": 0.8,
                    "d_prime": 2.0,
                    "roc_auc": 0.9,
                },
                {
                    "layer": 2,
                    "seed": seed,
                    "cross_fitted_tar": 0.8,
                    "d_prime": 2.2,
                    "roc_auc": 0.9,
                },
                {
                    "layer": 3,
                    "seed": seed,
                    "cross_fitted_tar": 0.7,
                    "d_prime": 3.0,
                    "roc_auc": 0.95,
                },
            ]
        )
    result = aggregate_and_select_layer(pd.DataFrame(rows))
    assert result.selected_layer == 2
    assert set(result.seed_winners.values()) == {2}
    assert list(result.summary["layer"]) == [1, 2, 3]


def test_identity_bootstrap_pairs_seeds_and_resamples_only_identities() -> None:
    selected_rows = []
    final_rows = []
    for identity in ["a", "b", "c", "d"]:
        for seed in range(5):
            selected_rows.append({"identity_id": identity, "seed": seed, "tar": 0.8})
            final_rows.append({"identity_id": identity, "seed": seed, "tar": 0.6})
    result = bootstrap_selected_vs_final_identity_tar(
        pd.DataFrame(selected_rows),
        pd.DataFrame(final_rows),
        bootstrap_seed=7,
        resamples=100,
    )
    assert result.identity_count == 4
    assert result.seed_count == 5
    assert result.selected_minus_final == pytest.approx(0.2)
    assert result.ci_lower_95 == pytest.approx(0.2)
    assert result.ci_upper_95 == pytest.approx(0.2)
    assert result.seed_delta_std == pytest.approx(0.0)
