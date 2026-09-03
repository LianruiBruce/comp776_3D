from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .utils import sha256_file


def _verify_insightface_checkout(path: Path, expected_revision: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing InsightFace checkout {path}. Run the E1 asset bootstrap."
        )
    base = ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path)]
    try:
        revision = subprocess.check_output(
            [*base, "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        dirty = subprocess.check_output(
            [*base, "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Unable to verify InsightFace checkout at {path}") from error
    if revision != expected_revision:
        raise ValueError(f"InsightFace revision mismatch: {revision} != {expected_revision}")
    if dirty:
        raise RuntimeError(f"InsightFace checkout has local changes:\n{dirty}")


def select_primary_face(boxes: np.ndarray, scores: np.ndarray) -> int:
    """Select one face deterministically: largest area, then highest score."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
        raise ValueError("At least one [x1, y1, x2, y2] face box is required")
    if scores.shape != (len(boxes),):
        raise ValueError("scores must have one value per face box")
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    areas = widths * heights
    # np.lexsort uses the final key as the primary key. Negated keys sort descending.
    return int(np.lexsort((np.arange(len(boxes)), -scores, -areas))[0])


def landmark_yaw_proxy(landmarks: np.ndarray) -> float:
    """Return a documented, dimensionless yaw proxy; this is not a pose estimator."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError("Expected five 2-D landmarks")
    eye_midpoint = 0.5 * (points[0] + points[1])
    interocular = float(np.linalg.norm(points[1] - points[0]))
    return float((points[2, 0] - eye_midpoint[0]) / max(interocular, 1e-6))


class InsightFaceFivePointAligner:
    """Pinned SCRFD detection followed by the official ArcFace 112x112 transform."""

    def __init__(self, config: dict[str, Any], locked_model: dict[str, Any]) -> None:
        checkout = Path(config["insightface_checkout_path"])
        _verify_insightface_checkout(checkout, locked_model["code_revision"])
        package_path = str((checkout / "python-package").resolve())
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        try:
            import cv2
            import onnxruntime
            if hasattr(onnxruntime, "preload_dlls"):
                onnxruntime.preload_dlls()
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "Five-point preprocessing dependencies are missing. Install "
                "requirements/face-preprocess.txt."
            ) from error

        self.cv2 = cv2
        self.model_path = Path(config["detector_model_path"])
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Missing detector model {self.model_path}. Run the E1 asset bootstrap."
            )
        observed_hash = sha256_file(self.model_path)
        expected_hash = locked_model["detector_sha256"].lower()
        if observed_hash.lower() != expected_hash:
            raise ValueError(
                f"InsightFace detector checksum mismatch: {observed_hash} != {expected_hash}"
            )

        providers = list(config["providers"])
        available = set(onnxruntime.get_available_providers())
        if bool(config.get("require_cuda", False)) and "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is required but unavailable in ONNX Runtime: "
                f"{sorted(available)}"
            )
        requested_available = [provider for provider in providers if provider in available]
        if not requested_available:
            raise RuntimeError(
                f"None of the requested ONNX Runtime providers are available: {providers}"
            )

        cache_root = Path(config["detector_cache_root"])
        model_name = str(config["detector_model_name"])
        expected_path = cache_root / "models" / model_name / self.model_path.name
        if expected_path.resolve() != self.model_path.resolve():
            raise ValueError(
                "detector_model_path must be inside detector_cache_root/models/model_name"
            )
        self.app = FaceAnalysis(
            name=model_name,
            root=str(cache_root),
            allowed_modules=["detection"],
            providers=requested_available,
        )
        detector_size = tuple(int(value) for value in config["detector_input_size"])
        self.app.prepare(
            ctx_id=int(config["ctx_id"]),
            det_thresh=float(config["detection_threshold"]),
            det_size=detector_size,
        )
        self.output_size = int(config["output_size"])
        self.jpeg_quality = int(config.get("jpeg_quality", 95))
        detector = self.app.models["detection"]
        self.runtime_providers = list(detector.session.get_providers())
        if bool(config.get("require_cuda", False)) and (
            not self.runtime_providers
            or self.runtime_providers[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError(
                "SCRFD session did not activate CUDAExecutionProvider: "
                f"{self.runtime_providers}"
            )

    def align(self, source: Path, target: Path) -> dict[str, Any]:
        from insightface.utils.face_align import estimate_norm

        image = self.cv2.imread(str(source), self.cv2.IMREAD_COLOR)
        if image is None:
            return self._failure("image_decode_failed", 0)

        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean() / 255.0)
        blur_variance = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        faces = self.app.get(image, max_num=0)
        if not faces:
            return {
                **self._failure("no_face_detected", 0),
                "source_width": int(image.shape[1]),
                "source_height": int(image.shape[0]),
                "brightness": brightness,
                "blur_variance": blur_variance,
            }

        boxes = np.stack([np.asarray(face.bbox, dtype=np.float32) for face in faces])
        scores = np.asarray([float(face.det_score) for face in faces], dtype=np.float32)
        selected_index = select_primary_face(boxes, scores)
        selected = faces[selected_index]
        landmarks = np.asarray(selected.kps, dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            return {
                **self._failure("invalid_five_landmarks", len(faces)),
                "source_width": int(image.shape[1]),
                "source_height": int(image.shape[0]),
                "brightness": brightness,
                "blur_variance": blur_variance,
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            transform = estimate_norm(landmarks, image_size=self.output_size, mode="arcface")
        aligned = self.cv2.warpAffine(
            image,
            transform,
            (self.output_size, self.output_size),
            borderValue=0.0,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
        write_options = (
            [self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            if target.suffix.lower() in {".jpg", ".jpeg"}
            else []
        )
        if not self.cv2.imwrite(str(temporary), aligned, write_options):
            raise RuntimeError(f"OpenCV failed to write aligned image: {temporary}")
        os.replace(temporary, target)

        return {
            "face_detection_attempted": True,
            "face_detected": True,
            "detected_face_count": int(len(faces)),
            "selected_face_index": selected_index,
            "detection_score": float(selected.det_score),
            "face_bbox_json": json.dumps(boxes[selected_index].round(4).tolist()),
            "face_landmarks_5_json": json.dumps(landmarks.round(4).tolist()),
            "alignment_transform_json": json.dumps(transform.round(8).tolist()),
            "alignment_template_index": 0,
            "alignment_source": "insightface_scrfd_five_point",
            "alignment_status": "aligned",
            "alignment_failure_reason": None,
            "usable_for_model": True,
            "source_width": int(image.shape[1]),
            "source_height": int(image.shape[0]),
            "brightness": brightness,
            "blur_variance": blur_variance,
            "yaw_proxy": landmark_yaw_proxy(landmarks),
            "relative_path": None,
            "sha256": sha256_file(target),
        }

    @staticmethod
    def _failure(reason: str, face_count: int) -> dict[str, Any]:
        return {
            "face_detection_attempted": True,
            "face_detected": False,
            "detected_face_count": int(face_count),
            "selected_face_index": None,
            "detection_score": None,
            "face_bbox_json": None,
            "face_landmarks_5_json": None,
            "alignment_transform_json": None,
            "alignment_template_index": None,
            "alignment_source": "insightface_scrfd_five_point",
            "alignment_status": "failed",
            "alignment_failure_reason": reason,
            "usable_for_model": False,
            "yaw_proxy": None,
            "relative_path": None,
            "sha256": None,
        }
