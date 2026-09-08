"""Independent, ID-keyed evaluation of immutable generated-image collections."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import auto_diagnostics as diag
from .auto_research import digest, dump
from .generation_evaluation import InsightFaceGeneratedFaceEncoder, LVFaceFinalEncoder
from .generation_metrics import build_identity_templates, gallery_identity_metrics


def adaface_input(image: Image.Image) -> torch.Tensor:
    """Exact official RGB PIL -> BGR CHW, /255, mean=.5 std=.5 convention."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if rgb.shape != (112, 112, 3):
        raise ValueError("AdaFace requires a shared aligned 112x112 crop")
    return torch.from_numpy(
        np.ascontiguousarray((rgb[:, :, ::-1] / 255 - 0.5).transpose(2, 0, 1) / 0.5)
    )


class AdaFaceEncoder:
    def __init__(self, cache: Path, device="cuda"):
        lock = json.loads((cache / "assets.json").read_text())
        path = cache / "AdaFace/net.py"
        weights = cache / "adaface_ir50_webface4m.ckpt"
        for p, key in ((path, "adaface_net.py"), (weights, "adaface_weights")):
            if digest(p) != lock[key]["sha256"]:
                raise ValueError(f"AdaFace asset mismatch: {p}")
        spec = importlib.util.spec_from_file_location("_pinned_adaface_net", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.device = device
        self.model = module.build_model("ir_50")
        # The official released Lightning checkpoint contains harmless training
        # metadata unsupported by weights_only. This is the pinned official file.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        state = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device).eval().requires_grad_(False)

    def encode_paths(self, paths, batch_size=32):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                tensors = []
                for p in paths[start : start + batch_size]:
                    with Image.open(p) as im:
                        tensors.append(adaface_input(im))
                feature, _ = self.model(torch.stack(tensors).to(self.device))
                outputs.append(torch.nn.functional.normalize(feature.float(), dim=1).cpu().numpy())
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite AdaFace output")
        return result


def make_encoders(root, lock):
    model = lock["models"]["insightface_buffalo_l_scrfd10g"]
    pack = root / "artifacts/cache/models/insightface/models/buffalo_l"
    face = InsightFaceGeneratedFaceEncoder(
        checkout=root / "third_party/InsightFace",
        cache_root=root / "artifacts/cache/models/insightface",
        model_name="buffalo_l",
        detector_path=pack / model["detector_filename"],
        recognition_path=pack / model["recognition_filename"],
        locked_model=model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    lv = LVFaceFinalEncoder(
        {
            "architecture": "vit_t",
            "third_party_path": str(root / "third_party/LVFace"),
            "weights_path": str(
                root / "artifacts/cache/models/LVFace/LVFace-T_Glint360K/LVFace-T_Glint360K.pt"
            ),
        },
        lock["models"]["lvface_t_glint360k"],
    )
    return face, lv


def make_lpips(cache):
    import lpips

    lock = json.loads((cache / "assets.json").read_text())
    weights = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    return diag.LPIPSAlexAdapter(
        alexnet_weights=cache / "torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
        lpips_weights=weights,
        alexnet_sha256=lock["alexnet"]["sha256"],
        device="cuda",
    )


def _jsonable_records(frame):
    return json.loads(frame.to_json(orient="records"))


def _write_table(path, records):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(path, index=False)
    return frame


def evaluate_collection(
    *,
    root,
    source,
    output,
    records,
    dataset,
    gallery_conditions,
    lock,
    expected_seed_pairs,
    frozen_detection=None,
):
    """Recompute recognizer embeddings; never consume anonymous legacy NPZ arrays.

    Dataset includes input and gallery rows; original-image hashes are verified.
    Frozen detector metadata is reusable only after ID and output hash agreement.
    New images are processed by the same pinned SCRFD and five-point transform.
    """
    output.mkdir(parents=True, exist_ok=True)
    cache = root / "artifacts/cache/auto_research"
    images = {}
    metadata = {}
    for row in dataset.to_dict("records"):
        sid = row["sample_id"]
        path = root / "data" / row["source_relative_path"]
        images[sid] = {
            "sample_id": sid,
            "identity_id": row["identity_id"],
            "source_sha256": row["source_sha256"],
            "path": path,
            "kind": "real",
            "condition": row["condition"],
        }
        metadata[sid] = {
            "face_landmarks_5": json.loads(row["face_landmarks_5_json"]),
            "detected_face_count": int(row["detected_face_count"]),
            "end_to_end_valid": int(row["detected_face_count"]) == 1,
            "alignment_status": row["alignment_status"],
        }
    for row in records:
        sid = row["sample_id"]
        if sid in images:
            raise ValueError("Duplicate real/generated sample ID")
        images[sid] = {
            "sample_id": sid,
            "identity_id": row["identity_id"],
            "source_sha256": row.get("output_sha256"),
            "path": source / (row.get("output_relative_path") or f"__missing__/{sid}"),
            "kind": "generated",
            "condition": row["condition"],
        }
    qc = []
    for sid, item in sorted(images.items()):
        if not item["source_sha256"] or not item["path"].exists():
            qc.append({"sample_id": sid, "status": "failed", "error": "generation_output_missing"})
        else:
            if digest(item["path"]) != item["source_sha256"]:
                raise ValueError(f"Image hash mismatch: {sid}")
            qc.append(diag.image_qc(item["path"], sid, expected_sha256=item["source_sha256"]))
    _write_table(output / "image_qc.csv", qc)
    dump(output / "duplicates.json", diag.duplicate_groups(qc))
    face, lv = make_encoders(root, lock)
    align112 = {}
    align256 = {}
    coverage = []
    usable_ids = {row["sample_id"] for row in qc if row["status"] == "complete"}
    for sid, item in sorted(images.items()):
        if sid not in usable_ids:
            coverage.append(
                {
                    "sample_id": sid,
                    "alignment_status": "failed",
                    "end_to_end_valid": False,
                    "alignment_failure_reason": "generation_output_missing",
                }
            )
            continue
        if item["kind"] == "generated":
            if frozen_detection is not None:
                row = frozen_detection[sid]
                if row["output_sha256"] != item["source_sha256"]:
                    raise ValueError("Frozen detection hash mismatch")
                metadata[sid] = {
                    "face_landmarks_5": ast.literal_eval(row["face_landmarks_5"]),
                    "detected_face_count": int(row["detected_face_count"]),
                    "end_to_end_valid": str(row["end_to_end_valid"]).lower() == "true",
                    "alignment_status": row["alignment_status"],
                }
            else:
                _, metadata[sid] = face.encode_and_align(item["path"])
        meta = metadata[sid]
        coverage.append({"sample_id": sid, **meta})
        if meta.get("alignment_status") != "aligned":
            continue
        with Image.open(item["path"]) as im:
            for size, targetmap in ((112, align112), (256, align256)):
                crop, matrix = diag.fixed_face_crop(im, meta["face_landmarks_5"], size=size)
                target = output / f"aligned{size}" / f"{sid}.png"
                target.parent.mkdir(exist_ok=True)
                crop.save(target)
                targetmap[sid] = target
    _write_table(output / "coverage.csv", coverage)
    dump(output / "alignment_metadata.json", metadata)
    encoders = {"lvface": lv}
    failures = {}
    try:
        encoders["adaface"] = AdaFaceEncoder(cache)
    except Exception as e:
        failures["adaface"] = f"{type(e).__name__}: {e}"
    valid_ids = sorted(align112)
    descriptors = [
        {"sample_id": sid, "source_sha256": images[sid]["source_sha256"]} for sid in valid_ids
    ]
    embeddings = {}
    identity_tables = []
    seed_tables = []
    real_tables = []
    repeat = {}
    pairs = diag.build_seed_pairs(records, expected_pair_count=expected_seed_pairs)
    dump(output / "seed_pair_manifest.json", pairs)
    identities = sorted({r["identity_id"] for r in records})
    query_ids = sorted(
        dataset.loc[dataset.condition.isin(gallery_conditions), "sample_id"].tolist()
    )
    common = {
        r["identity_id"]: r["sample_id"]
        for r in dataset.to_dict("records")
        if r["condition"] == "frontal_neutral"
    }
    for name, encoder in encoders.items():
        values = encoder.encode_paths([align112[sid] for sid in valid_ids], batch_size=32)
        again = encoder.encode_paths([align112[sid] for sid in valid_ids[:32]], batch_size=32)
        small = encoder.encode_paths([align112[sid] for sid in valid_ids[:2]], batch_size=32)
        repeat[name] = {
            "max_abs_delta": float(abs(values[: len(again)] - again).max()),
            "tolerance": 1e-5,
            "repeat_batch_size": 32,
            "batch_size2_vs32_max_abs_delta": float(abs(values[:2] - small).max()),
        }
        dump(output / "repeatability.json", repeat)
        if repeat[name]["max_abs_delta"] > 1e-5:
            raise ValueError("Recognizer repeat tolerance exceeded")
        embeddings[name] = dict(zip(valid_ids, values, strict=True))
        diag.save_keyed_array_cache(
            output / f"embeddings_{name}.npz",
            descriptors,
            {"embedding": values},
            preprocessing_version="scrfd_frozen5pt_arcface112_rgbfile_v1",
        )
        queries = [
            sid
            for sid in query_ids
            if sid in embeddings[name] and metadata[sid]["end_to_end_valid"]
        ]
        missing_gallery = sorted(set(identities) - {images[s]["identity_id"] for s in queries})
        if missing_gallery:
            failures[name + "_gallery"] = "No usable gallery for " + ",".join(missing_gallery)
            continue
        templates, gallery_ids = build_identity_templates(
            np.stack([embeddings[name][s] for s in queries]),
            [images[s]["identity_id"] for s in queries],
            identities,
        )
        selected = [r for r in records if r["sample_id"] in embeddings[name]]
        metrics = (
            gallery_identity_metrics(
                np.stack([embeddings[name][r["sample_id"]] for r in selected]),
                [r["identity_id"] for r in selected],
                templates,
                gallery_ids,
                donor_ids=[r.get("donor_identity_id") for r in selected],
                end_to_end_valid=[metadata[r["sample_id"]]["end_to_end_valid"] for r in selected],
            )
            if selected
            else pd.DataFrame()
        )
        for row, metric in zip(selected, _jsonable_records(metrics), strict=True):
            global_patch = None
            if (
                row.get("global_embedding_identity_id")
                and row.get("patch_image_identity_id")
                and row["global_embedding_identity_id"] != row["patch_image_identity_id"]
            ):
                sign = 1 if row["global_embedding_identity_id"] == row["identity_id"] else -1
                global_patch = sign * metric["target_minus_donor"]
            identity_tables.append(
                {
                    **{
                        k: row.get(k)
                        for k in ("sample_id", "identity_id", "condition", "prompt_id", "base_seed")
                    },
                    "evaluator": name,
                    "status": "complete",
                    **metric,
                    "global_minus_patch": global_patch,
                }
            )
        selected_ids = {r["sample_id"] for r in selected}
        for row in records:
            if row["sample_id"] not in selected_ids:
                identity_tables.append(
                    {
                        **{
                            k: row.get(k)
                            for k in (
                                "sample_id",
                                "identity_id",
                                "condition",
                                "prompt_id",
                                "base_seed",
                            )
                        },
                        "evaluator": name,
                        "status": "failed_no_embedding",
                        "target_similarity": None,
                        "target_impostor_margin": None,
                        "aligned_rank1": None,
                        "end_to_end_valid": False,
                        "end_to_end_rank1": False,
                        "global_minus_patch": None,
                    }
                )
        baseline_ids = [common[i] for i in identities if common[i] in embeddings[name]]
        baseline = gallery_identity_metrics(
            np.stack([embeddings[name][s] for s in baseline_ids]),
            [images[s]["identity_id"] for s in baseline_ids],
            templates,
            gallery_ids,
        )
        real_tables.extend(
            [
                {
                    "sample_id": sid,
                    "identity_id": images[sid]["identity_id"],
                    "evaluator": name,
                    **metric,
                }
                for sid, metric in zip(baseline_ids, _jsonable_records(baseline), strict=True)
            ]
        )
        seed_tables.extend(diag.seed_embedding_distances(pairs, embeddings[name], evaluator=name))
    _write_table(output / "identity_metrics.csv", identity_tables)
    _write_table(output / "real_baseline.csv", real_tables)
    _write_table(output / "seed_identity_distances.csv", seed_tables)
    dump(output / "repeatability.json", repeat)
    lpips = make_lpips(cache)
    imagepaths = {sid: item["path"] for sid, item in images.items() if sid in usable_ids}
    refs = {sid: item for sid, item in images.items() if item["kind"] == "real"}
    refpairs = diag.build_reference_pairs(records, common, refs)
    reference_scores = []
    seed_scores = []
    for space, paths in (("whole", imagepaths), ("face", align256)):
        reference_scores.extend(lpips.score_pairs(refpairs, paths, space=space))
        seed_scores.extend(lpips.score_pairs(pairs, paths, space=space))
    _write_table(output / "reference_pair_distances.csv", reference_scores)
    _write_table(
        output / "reference_distances.csv", diag.aggregate_reference_distances(reference_scores)
    )
    _write_table(output / "seed_lpips_distances.csv", seed_scores)
    landmarks = []
    try:
        assets = json.loads((cache / "assets.json").read_text())
        landmarker = diag.MediaPipeLandmarkerAdapter(
            model_path=cache / "face_landmarker.task", model_sha256=assets["landmarker"]["sha256"]
        )
        for sid, item in sorted(images.items()):
            if sid in usable_ids:
                landmarks.append(
                    {
                        **{k: item[k] for k in ("identity_id", "kind", "condition")},
                        **landmarker.analyze(
                            item["path"], sid, expected_sha256=item["source_sha256"]
                        ),
                    }
                )
        landmarker.close()
    except Exception as e:
        failures["landmarker"] = f"{type(e).__name__}: {e}"
    _write_table(output / "landmarks.csv", landmarks)
    summary = {
        "generated_count": len(records),
        "real_count": len(dataset),
        "seed_pairs": len(pairs),
        "aligned_generated": sum(r["sample_id"] in align112 for r in records),
        "failures": failures,
        "scope": "development_only",
        "legacy_npz_reused": False,
        "alignment": "same frozen original SCRFD five points; 112 template scaled for256",
        "metrics_by_condition": _jsonable_records(
            pd.DataFrame(identity_tables)
            .groupby(["evaluator", "condition"])[
                ["target_similarity", "target_impostor_margin", "end_to_end_rank1"]
            ]
            .mean()
            .reset_index()
        ),
    }
    dump(output / "summary.json", summary)
    return summary
