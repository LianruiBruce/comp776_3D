from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .modeling import load_lvface_model
from .preprocessing import _verify_insightface_checkout, select_primary_face
from .utils import sha256_file


class InsightFaceGeneratedFaceEncoder:
    """Pinned SCRFD + ArcFace encoder used by PhotoMaker and for diagnostics.

    ArcFace scores from this class are not independent evaluation because PhotoMaker V2
    consumes embeddings from the same model family. The aligned crop is also returned so
    a separate face recognizer can evaluate the identical detected face.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        cache_root: Path,
        model_name: str,
        detector_path: Path,
        recognition_path: Path,
        locked_model: dict[str, Any],
        providers: Sequence[str],
        ctx_id: int = 0,
        detection_threshold: float = 0.5,
        detector_input_size: tuple[int, int] = (640, 640),
        require_cuda: bool = True,
    ) -> None:
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
            from insightface.utils import face_align
        except ImportError as error:
            raise RuntimeError(
                "Generation evaluation requires requirements/face-preprocess.txt"
            ) from error

        expected_paths = {
            "detector": (
                detector_path,
                locked_model["detector_filename"],
                locked_model["detector_sha256"],
            ),
            "recognition": (
                recognition_path,
                locked_model["recognition_filename"],
                locked_model["recognition_sha256"],
            ),
        }
        model_directory = cache_root / "models" / model_name
        for label, (path, filename, expected_hash) in expected_paths.items():
            if path.resolve() != (model_directory / filename).resolve():
                raise ValueError(f"{label} path must be inside the pinned InsightFace pack")
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label} model: {path}")
            observed_hash = sha256_file(path)
            if observed_hash.lower() != str(expected_hash).lower():
                raise ValueError(
                    f"{label} model checksum mismatch: {observed_hash} != {expected_hash}"
                )

        available = set(onnxruntime.get_available_providers())
        if require_cuda and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is required for generation evaluation")
        active_providers = [provider for provider in providers if provider in available]
        if not active_providers:
            raise RuntimeError(f"Requested ONNX providers unavailable: {list(providers)}")
        self.app = FaceAnalysis(
            name=model_name,
            root=str(cache_root),
            allowed_modules=["detection", "recognition"],
            providers=active_providers,
        )
        self.app.prepare(
            ctx_id=ctx_id,
            det_thresh=detection_threshold,
            det_size=detector_input_size,
        )
        self.cv2 = cv2
        self.face_align = face_align
        self.runtime_providers = {
            name: list(model.session.get_providers())
            for name, model in self.app.models.items()
            if hasattr(model, "session")
        }
        if require_cuda:
            non_cuda = {
                name: runtime
                for name, runtime in self.runtime_providers.items()
                if not runtime or runtime[0] != "CUDAExecutionProvider"
            }
            if non_cuda:
                raise RuntimeError(f"InsightFace sessions did not activate CUDA: {non_cuda}")

    def encode_and_align(
        self, source: Path, aligned_target: Path | None = None
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        image = self.cv2.imread(str(source), self.cv2.IMREAD_COLOR)
        if image is None:
            return None, self._failure("image_decode_failed", 0)
        faces = self.app.get(image, max_num=0)
        if not faces:
            return None, self._failure("no_face_detected", 0)
        boxes = np.stack([np.asarray(face.bbox, dtype=np.float32) for face in faces])
        scores = np.asarray([float(face.det_score) for face in faces], dtype=np.float32)
        selected_index = select_primary_face(boxes, scores)
        selected = faces[selected_index]
        landmarks = np.asarray(selected.kps, dtype=np.float32)
        embedding = np.asarray(selected.embedding, dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            return None, self._failure("invalid_five_landmarks", len(faces))
        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            return None, self._failure("invalid_recognition_embedding", len(faces))
        embedding /= max(float(np.linalg.norm(embedding)), 1.0e-12)

        aligned = self.face_align.norm_crop(image, landmark=landmarks, image_size=112)
        if aligned_target is not None:
            aligned_target.parent.mkdir(parents=True, exist_ok=True)
            temporary = aligned_target.with_name(
                f"{aligned_target.stem}.tmp{aligned_target.suffix}"
            )
            if not self.cv2.imwrite(str(temporary), aligned):
                raise RuntimeError(f"OpenCV failed to write {temporary}")
            temporary.replace(aligned_target)
        metadata = {
            "face_detected": True,
            "detected_face_count": int(len(faces)),
            "selected_face_index": selected_index,
            "detection_score": float(selected.det_score),
            "face_bbox": boxes[selected_index].round(4).tolist(),
            "face_landmarks_5": landmarks.round(4).tolist(),
            "alignment_status": "aligned",
            "alignment_failure_reason": None,
            "end_to_end_valid": len(faces) == 1,
        }
        return embedding, metadata

    @staticmethod
    def _failure(reason: str, face_count: int) -> dict[str, Any]:
        return {
            "face_detected": False,
            "detected_face_count": int(face_count),
            "selected_face_index": None,
            "detection_score": None,
            "face_bbox": None,
            "face_landmarks_5": None,
            "alignment_status": "failed",
            "alignment_failure_reason": reason,
            "end_to_end_valid": False,
        }


class LVFaceFinalEncoder:
    """Independent final-head LVFace-T evaluator for 112x112 aligned crops."""

    def __init__(
        self,
        model_config: dict[str, Any],
        locked_model: dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.model = load_lvface_model(model_config, locked_model).to(self.device).eval()

    @staticmethod
    def _load_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5

    def encode_paths(self, paths: Sequence[Path], batch_size: int = 64) -> np.ndarray:
        if not paths:
            raise ValueError("At least one aligned image path is required")
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(paths), batch_size):
                batch_paths = paths[offset : offset + batch_size]
                batch = torch.stack([self._load_tensor(path) for path in batch_paths])
                batch = batch.to(self.device)
                embeddings = F.normalize(self.model(batch).float(), dim=1)
                outputs.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)
