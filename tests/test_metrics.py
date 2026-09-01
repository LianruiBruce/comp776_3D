import numpy as np

from id_layers.metrics import frozen_threshold_metrics, verification_metrics


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
