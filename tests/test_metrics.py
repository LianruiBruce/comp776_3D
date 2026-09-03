import numpy as np
import pandas as pd

from id_layers.metrics import (
    bootstrap_selected_template_tar,
    frozen_template_threshold_metrics,
    frozen_threshold_metrics,
    select_layers,
    template_query_metrics,
    verification_metrics,
)


def test_separable_embeddings_have_high_identity_scores() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.01, 0.99],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "a", "b", "b", "c", "c"])
    metrics = verification_metrics(embeddings, labels, far_targets=[0.01])
    assert metrics["positive_cosine_mean"] > metrics["negative_cosine_mean"]
    assert metrics["roc_auc"] == 1.0
    assert metrics["loo_top1"] == 1.0
    assert metrics["tar_at_far_0.01"] == 1.0


def test_threshold_is_calibrated_before_evaluation() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.01, 0.99],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "a", "b", "b", "c", "c"])
    result = frozen_threshold_metrics(embeddings, labels, embeddings, labels, [0.0])
    operating_point = result["operating_points"]["far_0"]
    assert operating_point["calibration_far"] == 0.0
    assert operating_point["evaluation_far"] == 0.0
    assert operating_point["evaluation_tar"] == 1.0


def test_template_query_metrics_use_reference_templates() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.01, 0.99],
        ],
        dtype=np.float32,
    )
    manifest = pd.DataFrame(
        {
            "identity_id": ["a", "a", "b", "b", "c", "c"],
            "protocol_role": ["reference", "query"] * 3,
        }
    )
    metrics = template_query_metrics(embeddings, manifest, far_targets=[0.01])
    assert metrics["template_identities"] == 3
    assert metrics["template_queries"] == 3
    assert metrics["template_positive_pairs"] == 3
    assert metrics["template_negative_pairs"] == 6
    assert metrics["template_roc_auc"] == 1.0
    assert metrics["template_rank1"] == 1.0
    assert metrics["template_recall_at_5"] == 1.0


def test_template_recall_at_5_can_exceed_rank1() -> None:
    references = np.eye(6, dtype=np.float32)
    query = np.asarray(
        [
            [0.6, 0.8, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = np.concatenate([references, query], axis=0)
    manifest = pd.DataFrame(
        {
            "identity_id": ["a", "b", "c", "d", "e", "f", "a", "a"],
            "protocol_role": ["reference"] * 6 + ["query", "query"],
        }
    )
    metrics = template_query_metrics(embeddings, manifest, far_targets=[0.01])
    assert metrics["template_rank1"] == 0.0
    assert metrics["template_recall_at_5"] == 1.0


def test_template_threshold_can_hold_far_below_one_negative_pair() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.01, 0.99],
        ],
        dtype=np.float32,
    )
    manifest = pd.DataFrame(
        {
            "identity_id": ["a", "a", "b", "b", "c", "c"],
            "protocol_role": ["reference", "query"] * 3,
        }
    )
    result = frozen_template_threshold_metrics(
        embeddings, manifest, embeddings, manifest, [0.01]
    )
    point = result["operating_points"]["far_0.01"]
    assert point["calibration_far"] == 0.0
    assert point["evaluation_far"] == 0.0


def test_layer_selection_uses_preregistered_tiebreaker() -> None:
    metrics = pd.DataFrame(
        {
            "split": ["dev", "dev"],
            "representation": ["probe", "probe"],
            "layer": [1, 2],
            "primary": [1.0, 1.0],
            "secondary": [2.0, 3.0],
        }
    )
    result = select_layers(metrics, "dev", "primary", ["secondary"])
    assert result["representations"]["probe"]["selected_layer"] == 2


def test_identity_bootstrap_compares_selected_with_final_layer() -> None:
    manifest = pd.DataFrame(
        {
            "identity_id": ["a", "a", "b", "b", "c", "c"] * 2,
            "protocol_role": ["reference", "query"] * 6,
            "split": ["dev"] * 6 + ["test"] * 6,
        }
    )
    final_values = np.asarray(
        [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1], [0, 0.1, 0.9]]
        * 2,
        dtype=np.float32,
    )
    embeddings = {(1, "probe"): final_values.copy(), (2, "probe"): final_values.copy()}
    selection = {"representations": {"probe": {"selected_layer": 1}}}
    result = bootstrap_selected_template_tar(
        embeddings, manifest, selection, "dev", "test", [0.01], 20, 7
    )
    delta = result[result["comparison"] == "selected_minus_final"].iloc[0]
    assert delta["identity_mean_estimate"] == 0.0
    assert delta["bootstrap_ci_lower_95"] == 0.0
    assert delta["bootstrap_ci_upper_95"] == 0.0
