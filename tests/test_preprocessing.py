import numpy as np

from id_layers.preprocessing import landmark_yaw_proxy, select_primary_face


def test_primary_face_prefers_area_then_score() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 20, 10], [0, 0, 20, 10]])
    scores = np.asarray([0.99, 0.80, 0.90])
    assert select_primary_face(boxes, scores) == 2


def test_landmark_yaw_proxy_is_centered_for_symmetric_landmarks() -> None:
    landmarks = np.asarray([[0, 0], [2, 0], [1, 1], [0.5, 2], [1.5, 2]])
    assert landmark_yaw_proxy(landmarks) == 0.0
