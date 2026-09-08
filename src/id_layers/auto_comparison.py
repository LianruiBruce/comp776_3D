"""Matched whole-system comparison of two completed automatic generation runs.

Same identity/input/seed labels are required; latent hash agreement is checked and
reported rather than assumed. Model-specific prompts, samplers and settings still
differ, so no result identifies an individual architectural module as the cause.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .auto_analysis import ROOT, _bool_column, _group_summary, _project_path, _read


def compare_generation_systems(
    evaluation_a: Path,
    generation_a: Path,
    evaluation_b: Path,
    generation_b: Path,
    output_dir: Path,
    *,
    label_a: str,
    label_b: str,
    expected_cells: int = 48,
    resamples: int = 2000,
    seed: int = 20260905,
) -> dict[str, Any]:
    sources = list(map(_project_path, [evaluation_a, generation_a, evaluation_b, generation_b]))
    evaluation_a, generation_a, evaluation_b, generation_b = sources
    output_dir = _project_path(output_dir)
    protected = ROOT / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot"
    if any(output_dir.is_relative_to(p) for p in [*sources, protected]):
        raise ValueError("Comparison cannot modify source runs")
    if output_dir.exists():
        raise FileExistsError("Comparison output already exists")
    if not label_a or not label_b or label_a == label_b:
        raise ValueError("Need two distinct system labels")
    start = time.perf_counter()
    manifests = []
    for directory in (generation_a, generation_b):
        rows = [
            json.loads(line)
            for line in (directory / "generation_manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate generation sample ID")
        manifests.append({r["sample_id"]: r for r in rows})
    left, right = manifests
    if len(left) != expected_cells or set(left) != set(right):
        raise ValueError("Whole-system comparison must retain the same complete matrix")
    fields = (
        "identity_id",
        "condition",
        "prompt_id",
        "base_seed",
        "effective_seed",
        "reference_sample_ids",
        "width",
        "height",
    )
    pairing = []
    for sample_id in sorted(left):
        a, b = left[sample_id], right[sample_id]
        if any(field not in a or field not in b or a[field] != b[field] for field in fields):
            raise ValueError("Cross-system identity/input/seed matrix mismatch")
        if not a.get("latent_sha256") or not b.get("latent_sha256"):
            raise ValueError("Missing recorded initial latent hash")
        pairing.append(
            {
                "sample_id": sample_id,
                **{k: a[k] for k in fields},
                "status_a": a["status"],
                "status_b": b["status"],
                "prompt_a": a["prompt"],
                "prompt_b": b["prompt"],
                "steps_a": a["num_inference_steps"],
                "steps_b": b["num_inference_steps"],
                "cfg_a": a["guidance_scale"],
                "cfg_b": b["guidance_scale"],
                "latent_sha256_a": a["latent_sha256"],
                "latent_sha256_b": b["latent_sha256"],
                "stored_initial_latent_equal": a["latent_sha256"] == b["latent_sha256"],
            }
        )
    tables = {"cell_pairing": pd.DataFrame(pairing)}
    identity_frames = [
        _read(directory / "identity_metrics.csv", required=True)
        for directory in (evaluation_a, evaluation_b)
    ]
    for frame in identity_frames:
        if frame.duplicated(["sample_id", "evaluator"]).any():
            raise ValueError("Duplicate identity metric cell")
        frame["end_to_end_rank1"] = _bool_column(frame.end_to_end_rank1)
    keys = ["sample_id", "identity_id", "condition", "prompt_id", "base_seed", "evaluator"]
    merged = identity_frames[0].merge(
        identity_frames[1],
        on=keys,
        how="outer",
        suffixes=("_a", "_b"),
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError(
            "Evaluator matrix differs between models; unavailable evaluators "
            "need explicit accounting"
        )
    metrics = ["target_similarity", "target_impostor_margin", "end_to_end_rank1"]
    for metric in metrics:
        merged[metric + "_a_minus_b"] = merged[metric + "_a"] - merged[metric + "_b"]
    tables["identity_per_cell"] = merged.drop(columns="_merge")
    summaries, identity_means = [], []
    for scope in ("each_expression", "both_expressions"):
        groups = ["evaluator", "prompt_id"] if scope == "each_expression" else ["evaluator"]
        means, summary = _group_summary(
            merged, groups, [m + "_a_minus_b" for m in metrics], resamples=resamples, seed=seed
        )
        means["scope"], summary["scope"] = scope, scope
        identity_means.append(means)
        summaries.append(summary)
    tables["identity_means"], tables["identity_summary"] = (
        pd.concat(identity_means),
        pd.concat(summaries),
    )
    references = [
        _read(directory / "reference_distances.csv", required=True)
        for directory in (evaluation_a, evaluation_b)
    ]
    keys = ["sample_id", "identity_id", "condition", "prompt_id", "base_seed", "space"]
    if any(frame.duplicated(keys).any() for frame in references):
        raise ValueError("Duplicate reference-distance cell")
    ref = references[0].merge(
        references[1],
        on=keys,
        how="outer",
        suffixes=("_a", "_b"),
        validate="one_to_one",
        indicator=True,
    )
    if (ref["_merge"] != "both").any():
        raise ValueError("Reference-distance matrix differs between models")
    metric = "common_k1_distance_a_minus_b"
    ref[metric] = ref.common_k1_distance_a - ref.common_k1_distance_b
    tables["reference_per_cell"] = ref.drop(columns="_merge")
    tables["reference_identity"], tables["reference_summary"] = _group_summary(
        ref, ["space", "prompt_id"], [metric], resamples=resamples, seed=seed
    )
    output_dir.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / (name + ".csv"), index=False)
    result = {
        "status": "complete",
        "evidence_type": "DEVELOPMENT_WHOLE_SYSTEM_COMPARISON",
        "contrast_direction": label_a + " minus " + label_b,
        "expected_cells": expected_cells,
        "matched_cell_count": len(pairing),
        "matched_identity_count": len({row["identity_id"] for row in pairing}),
        "stored_initial_latent_equal_count": sum(
            row["stored_initial_latent_equal"] for row in pairing
        ),
        "seed": seed,
        "resamples": resamples,
        "cpu_wall_seconds": time.perf_counter() - start,
        "scope": (
            "Whole systems differ in conditioning, sampler and configuration; differences are "
            "not attributed to an individual module. LPIPS change is proximity to a common "
            "reference, not direct cross-model image distance or copying rate. Identical stored "
            "noise is not an identical denoising trajectory. No human likeness was measured."
        ),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                generation_a / "generation_manifest.jsonl",
                generation_b / "generation_manifest.jsonl",
                evaluation_a / "identity_metrics.csv",
                evaluation_b / "identity_metrics.csv",
                evaluation_a / "reference_distances.csv",
                evaluation_b / "reference_distances.csv",
            ]
        },
    }
    for name in ("summary.json", "status.json", "resolved_config.json"):
        (output_dir / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compare_expression_changes(
    analysis_a: Path,
    analysis_b: Path,
    output_dir: Path,
    *,
    label_a: str,
    label_b: str,
    resamples: int = 2000,
    seed: int = 20260905,
) -> dict[str, Any]:
    """Identity-paired difference of smile-minus-neutral whole-system changes.

    This tests an interaction directly rather than comparing whether two
    separate confidence intervals exclude zero. It does not match actual
    expression intensity, which is itself one of the measured outcomes.
    """
    start = time.perf_counter()
    analysis_a, analysis_b, output_dir = map(_project_path, [analysis_a, analysis_b, output_dir])
    protected = ROOT / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot"
    if any(output_dir.is_relative_to(p) for p in (analysis_a, analysis_b, protected)):
        raise ValueError("Interaction output cannot modify source runs")
    if output_dir.exists():
        raise FileExistsError("Interaction output already exists")
    if not label_a or not label_b or label_a == label_b:
        raise ValueError("Need distinct system labels")
    frames, hashes = [], {}
    keys = ["metric", "evaluator", "identity_id", "condition", "base_seed"]
    for directory in (analysis_a, analysis_b):
        for filename in ("status.json", "summary.json", "expression_paired_cells.csv"):
            path = directory / filename
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete" or not summary.get("new_expression_protocol"):
            raise ValueError("Need completed single-factor expression analyses")
        frame = _read(directory / "expression_paired_cells.csv", required=True)
        frame["evaluator"] = frame.evaluator.fillna("mediapipe")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate expression-pair identity/seed/metric")
        frames.append(frame)
    pairs = frames[0].merge(
        frames[1],
        on=keys,
        how="outer",
        suffixes=("_a", "_b"),
        validate="one_to_one",
        indicator=True,
    )
    if (pairs["_merge"] != "both").any():
        raise ValueError("Expression analyses have different paired matrices")
    pairs["interaction"] = pairs.difference_a - pairs.difference_b
    pairs = pairs.rename(columns={"metric": "outcome"}).drop(columns="_merge")
    means, estimates = _group_summary(
        pairs,
        ["outcome", "evaluator", "condition"],
        ["interaction"],
        resamples=resamples,
        seed=seed,
    )
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source changed during interaction analysis")
    output_dir.mkdir(parents=True)
    for name, frame in (
        ("paired_cells", pairs),
        ("identity_means", means),
        ("interaction_summary", estimates),
    ):
        frame.to_csv(output_dir / (name + ".csv"), index=False)
    result = {
        "status": "complete",
        "evidence_type": "DEVELOPMENT_WHOLE_SYSTEM_EXPRESSION_INTERACTION",
        "contrast_direction": f"({label_a} smile-neutral) minus ({label_b} smile-neutral)",
        "identity_count": int(pairs.identity_id.nunique()),
        "metric_pair_count": len(pairs),
        "resamples": resamples,
        "seed": seed,
        "cpu_wall_seconds": time.perf_counter() - start,
        "source_sha256": hashes,
        "scope": "Fixed instruction interaction between whole systems, not matched actual "
        "expression intensity and not human likeness. Exploratory identity "
        "bootstrap intervals, without multiplicity adjustment.",
    }
    for name in ("summary.json", "status.json", "resolved_config.json"):
        (output_dir / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
