"""Identity-level descriptive analysis of automatic diagnostics; no human ratings.

All generated and real cosine comparisons stay within the same recognizer.
Bootstrap identities, never treat multiple seeds/images as independent people.
This is a development analysis; intervals are unadjusted exploratory intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTRASTS = (
    ("k4_diverse_minus_k1", "k4_diverse_full", "k1_full"),
    ("k4_diverse_minus_k4_repeat", "k4_diverse_full", "k4_repeat_full"),
    ("k1_minus_text", "k1_full", "text_only"),
)
IDENTITY_METRICS = ("target_similarity", "target_impostor_margin", "end_to_end_rank1")
ROOT = Path(__file__).resolve().parents[2]


def _project_path(path: Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else ROOT / value).resolve()


def identity_interval(values: pd.Series, *, resamples: int, seed: int) -> dict[str, Any]:
    """Each Series index must denote one independent identity, values can be missing."""
    if not values.index.is_unique:
        raise ValueError("Identity interval requires unique identity index")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    values = pd.to_numeric(values, errors="raise").sort_index()
    valid = values.dropna().to_numpy(dtype=float)
    if np.isinf(valid).any():
        raise ValueError("Infinite metric")
    result = {
        "identity_count": len(values),
        "valid_identity_count": len(valid),
        "estimate": float(valid.mean()) if len(valid) else None,
        "ci95_lower": None,
        "ci95_upper": None,
    }
    if len(valid) >= 2:
        rng = np.random.default_rng(seed)
        samples = valid[rng.integers(0, len(valid), (resamples, len(valid)))].mean(axis=1)
        result["ci95_lower"], result["ci95_upper"] = np.quantile(samples, [0.025, 0.975]).tolist()
    return result


def paired_identity_contrast(
    frame: pd.DataFrame,
    *,
    metric: str,
    factor: str,
    treatment: str,
    control: str,
    pairing_keys: tuple[str, ...] = ("identity_id", "prompt_id", "base_seed"),
    resamples: int = 2000,
    seed: int = 20260905,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match cells by IDs before averaging within identity; retain incomplete pairs."""
    selected = frame.loc[
        frame[factor].isin([treatment, control]), [*pairing_keys, factor, metric]
    ].copy()
    if selected.duplicated([*pairing_keys, factor]).any():
        raise ValueError("Duplicate cells in paired contrast")
    if "identity_id" not in pairing_keys:
        raise ValueError("Pairing requires identity_id")
    if selected.empty:
        return pd.DataFrame(), {"status": "not_applicable", "reason": "contrast_conditions_absent"}
    paired = selected.pivot(index=list(pairing_keys), columns=factor, values=metric).reindex(
        columns=[treatment, control]
    )
    paired = paired.rename(columns={treatment: "treatment_value", control: "control_value"})
    paired["difference"] = paired["treatment_value"] - paired["control_value"]
    paired["valid_pair"] = paired[["treatment_value", "control_value"]].notna().all(axis=1)
    paired = paired.reset_index().sort_values(list(pairing_keys)).reset_index(drop=True)
    means = paired.groupby("identity_id", sort=True)["difference"].mean()
    summary = {
        "status": "complete",
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "expected_pair_count": len(paired),
        "valid_pair_count": int(paired["valid_pair"].sum()),
        **identity_interval(means, resamples=resamples, seed=seed),
    }
    return paired, summary


def _read(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _bool_column(values: pd.Series) -> pd.Series:
    mapped = (
        values.astype(str)
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0, "nan": np.nan})
    )
    if mapped.isna().sum() > values.isna().sum():
        raise ValueError("Invalid boolean metric representation")
    return mapped


def _group_summary(
    frame: pd.DataFrame, groups: list[str], metrics: list[str], *, resamples: int, seed: int
):
    identity_rows, summaries = [], []
    for key, part in frame.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(groups, key, strict=True))
        for metric in metrics:
            if metric not in part:
                continue
            means = part.groupby("identity_id", sort=True)[metric].mean()
            for identity, value in means.items():
                identity_rows.append(
                    {
                        **labels,
                        "identity_id": identity,
                        "metric": metric,
                        "value": value,
                        "valid_image_or_pair_count": int(
                            part.loc[part.identity_id == identity, metric].notna().sum()
                        ),
                    }
                )
            summaries.append(
                {
                    **labels,
                    "metric": metric,
                    "row_count": len(part),
                    "valid_row_count": int(part[metric].notna().sum()),
                    **identity_interval(means, resamples=resamples, seed=seed),
                }
            )
    return pd.DataFrame(identity_rows), pd.DataFrame(summaries)


def analyze_auto_collection(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    new_expression_protocol: bool = False,
    resamples: int = 2000,
    seed: int = 20260905,
) -> dict[str, Any]:
    """Read a completed independent evaluation; write a NEW analysis directory.

    new_expression_protocol must only be true for the frozen neutral/smile prompts
    differing in expression alone. Old p0_easy/p1_challenging prompts are never
    silently interpreted as a single-expression intervention.
    """
    evaluation_dir, output_dir = _project_path(evaluation_dir), _project_path(output_dir)
    if output_dir == evaluation_dir or output_dir.is_relative_to(evaluation_dir):
        raise ValueError("Analysis output must be separate from immutable evaluation")
    root = ROOT
    frozen = root / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot"
    if output_dir == frozen or output_dir.is_relative_to(frozen):
        raise ValueError("Cannot write inside frozen pilot")
    if output_dir.exists():
        raise FileExistsError("Analysis output already exists; completed outputs are immutable")
    source_status = evaluation_dir / "status.json"
    if source_status.exists():
        status = json.loads(source_status.read_text(encoding="utf-8"))
        if status.get("status", status.get("state")) != "complete":
            raise ValueError("Source evaluation is not complete")
    identity = _read(evaluation_dir / "identity_metrics.csv", required=True)
    if identity.duplicated(["sample_id", "evaluator"]).any():
        raise ValueError("Duplicate recognizer/sample rows")
    for col in ("end_to_end_rank1", "aligned_rank1", "end_to_end_valid"):
        if col in identity:
            identity[col] = _bool_column(identity[col])
    output_dir.mkdir(parents=True)
    start = time.perf_counter()
    tables: dict[str, pd.DataFrame] = {}
    inputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in evaluation_dir.glob("*.csv")
    }
    (output_dir / "source_hashes.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    configuration = {
        "evaluation_dir": str(evaluation_dir),
        "new_expression_protocol": new_expression_protocol,
        "resamples": resamples,
        "seed": seed,
        "evidence_type": "DEVELOPMENT_AUTOMATIC_DIAGNOSTICS",
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    tables["identity_means"], tables["identity_summary"] = _group_summary(
        identity, ["evaluator", "condition"], list(IDENTITY_METRICS), resamples=resamples, seed=seed
    )
    contrast_pairs, contrasts = [], []
    for evaluator, part in identity.groupby("evaluator", sort=True):
        for name, treatment, control in CONTRASTS:
            if not {treatment, control}.issubset(set(part.condition)):
                continue
            for metric in IDENTITY_METRICS:
                pairs, summary = paired_identity_contrast(
                    part,
                    metric=metric,
                    factor="condition",
                    treatment=treatment,
                    control=control,
                    resamples=resamples,
                    seed=seed,
                )
                contrasts.append({"evaluator": evaluator, "contrast": name, **summary})
                contrast_pairs.extend(
                    {"evaluator": evaluator, "contrast": name, "metric": metric, **r}
                    for r in pairs.to_dict("records")
                )
    tables["paired_identity_contrasts"], tables["paired_cells"] = (
        pd.DataFrame(contrasts),
        pd.DataFrame(contrast_pairs),
    )
    baseline = _read(evaluation_dir / "real_baseline.csv")
    if not baseline.empty:
        if baseline.duplicated(["evaluator", "identity_id"]).any():
            raise ValueError("Expected exactly one common real reference per evaluator/identity")
        real = baseline[["evaluator", "identity_id", "target_similarity"]].rename(
            columns={"target_similarity": "real_reference_similarity"}
        )
        gaps = identity.merge(
            real, on=["evaluator", "identity_id"], how="left", validate="many_to_one"
        )
        gaps["real_reference_minus_generated_cosine"] = (
            gaps.real_reference_similarity - gaps.target_similarity
        )
        tables["real_reference_gap_identity"], tables["real_reference_gap_summary"] = (
            _group_summary(
                gaps,
                ["evaluator", "condition"],
                ["real_reference_minus_generated_cosine"],
                resamples=resamples,
                seed=seed,
            )
        )
    conflict = (
        identity.loc[
            identity.condition.isin(["global_target_patch_donor", "global_donor_patch_target"])
        ].copy()
        if "global_minus_patch" in identity
        else pd.DataFrame()
    )
    if not conflict.empty:
        valid_conflict = conflict.global_minus_patch.notna()
        conflict["global_following"] = np.where(
            valid_conflict, (conflict.global_minus_patch > 0).astype(float), np.nan
        )
        conflict["patch_following"] = np.where(
            valid_conflict, (conflict.global_minus_patch < 0).astype(float), np.nan
        )
        conflict["tied"] = np.where(
            valid_conflict, (conflict.global_minus_patch == 0).astype(float), np.nan
        )
        tables["conflict_per_image"] = conflict
        tables["conflict_identity"], tables["conflict_summary"] = _group_summary(
            conflict,
            ["evaluator"],
            ["global_following", "patch_following", "tied", "global_minus_patch"],
            resamples=resamples,
            seed=seed,
        )
    evaluators = sorted(identity.evaluator.unique())
    agreement_rows = []
    if {"adaface", "lvface"}.issubset(evaluators):
        columns = [
            "sample_id",
            "identity_id",
            "condition",
            "target_similarity",
            "end_to_end_rank1",
            "global_minus_patch",
        ]
        columns = [c for c in columns if c in identity]
        agree = identity.loc[identity.evaluator == "lvface", columns].merge(
            identity.loc[identity.evaluator == "adaface", columns],
            on=["sample_id", "identity_id", "condition"],
            how="outer",
            suffixes=("_lvface", "_adaface"),
            validate="one_to_one",
        )
        agree["rank1_agreement"] = np.where(
            agree[["end_to_end_rank1_lvface", "end_to_end_rank1_adaface"]].notna().all(axis=1),
            (agree.end_to_end_rank1_lvface == agree.end_to_end_rank1_adaface).astype(float),
            np.nan,
        )
        if "global_minus_patch_lvface" in agree:
            good = (
                agree[["global_minus_patch_lvface", "global_minus_patch_adaface"]]
                .notna()
                .all(axis=1)
            )
            agree["conflict_direction_agreement"] = np.where(
                good,
                (
                    np.sign(agree.global_minus_patch_lvface)
                    == np.sign(agree.global_minus_patch_adaface)
                ).astype(float),
                np.nan,
            )
        tables["recognizer_agreement_per_image"] = agree
        for condition, part in agree.groupby("condition", sort=True):
            for metric in ("rank1_agreement", "conflict_direction_agreement"):
                if metric in part:
                    agreement_rows.append(
                        {
                            "condition": condition,
                            "metric": metric,
                            **identity_interval(
                                part.groupby("identity_id")[metric].mean(),
                                resamples=resamples,
                                seed=seed,
                            ),
                        }
                    )
        tables["recognizer_agreement_summary"] = pd.DataFrame(agreement_rows)
    references = _read(evaluation_dir / "reference_distances.csv")
    if not references.empty:
        tables["reference_identity"], tables["reference_summary"] = _group_summary(
            references,
            ["space", "condition"],
            [
                "common_k1_distance",
                "actual_reference_mean_distance",
                "actual_reference_min_distance",
            ],
            resamples=resamples,
            seed=seed,
        )
    for filename, grouping, metric in (
        ("seed_lpips_distances", ["space", "condition"], "distance"),
        ("seed_identity_distances", ["evaluator", "condition"], "embedding_cosine_distance"),
    ):
        frame = _read(evaluation_dir / (filename + ".csv"))
        if not frame.empty:
            tables[filename + "_identity"], tables[filename + "_summary"] = _group_summary(
                frame, grouping, [metric], resamples=resamples, seed=seed
            )
    landmarks = _read(evaluation_dir / "landmarks.csv")
    if not landmarks.empty:
        metadata_cols = ["sample_id", "identity_id", "condition", "prompt_id", "base_seed"]
        metadata = identity[metadata_cols].drop_duplicates()
        if metadata.sample_id.duplicated().any():
            raise ValueError("Evaluator metadata disagree for one sample ID")
        generated = landmarks.loc[landmarks.sample_id.isin(metadata.sample_id)].copy()
        generated = generated.drop(
            columns=[c for c in metadata_cols if c != "sample_id" and c in generated]
        ).merge(metadata, on="sample_id", how="right", validate="one_to_one")
        tables["expression_per_image"] = generated
        tables["expression_identity"], tables["expression_summary"] = _group_summary(
            generated,
            ["condition", "prompt_id"],
            ["mouth_smile_mean", "jaw_open"],
            resamples=resamples,
            seed=seed,
        )
        if new_expression_protocol:
            if set(metadata.prompt_id) != {"neutral", "smile"}:
                raise ValueError(
                    "Single-expression analysis requires frozen neutral/smile prompt IDs"
                )
            expression_contrasts, expression_pairs = [], []
            for metric in ("mouth_smile_mean", "jaw_open"):
                pairs, summary = paired_identity_contrast(
                    generated,
                    metric=metric,
                    factor="prompt_id",
                    treatment="smile",
                    control="neutral",
                    pairing_keys=("identity_id", "condition", "base_seed"),
                    resamples=resamples,
                    seed=seed,
                )
                expression_contrasts.append(summary)
                expression_pairs.extend({"metric": metric, **r} for r in pairs.to_dict("records"))
            for evaluator, part in identity.groupby("evaluator", sort=True):
                for metric in IDENTITY_METRICS:
                    pairs, summary = paired_identity_contrast(
                        part,
                        metric=metric,
                        factor="prompt_id",
                        treatment="smile",
                        control="neutral",
                        pairing_keys=("identity_id", "condition", "base_seed"),
                        resamples=resamples,
                        seed=seed,
                    )
                    expression_contrasts.append({"evaluator": evaluator, **summary})
                    expression_pairs.extend(
                        {"evaluator": evaluator, "metric": metric, **r}
                        for r in pairs.to_dict("records")
                    )
            tables["expression_paired_summary"] = pd.DataFrame(expression_contrasts)
            tables["expression_paired_cells"] = pd.DataFrame(expression_pairs)
    for name, table in tables.items():
        table.to_csv(output_dir / (name + ".csv"), index=False)
    _figures(tables, output_dir)
    if inputs != {
        name: hashlib.sha256((evaluation_dir / name).read_bytes()).hexdigest() for name in inputs
    }:
        raise ValueError("Source CSV changed during analysis")
    summary = {
        **configuration,
        "table_rows": {name: len(table) for name, table in tables.items()},
        "cpu_wall_seconds": time.perf_counter() - start,
        "interpretation": [
            "Recognizers measure identity geometry, not human likeness; no human responses used.",
            "Cosine magnitudes are not directly compared between recognizers.",
            "Real reference baseline is one original generator input scored against held-out "
            "real gallery, not a human real-candidate experiment.",
            "Common K1 LPIPS is comparable across reference counts; minimum over all inputs "
            "has a candidate-count advantage and is descriptive only.",
            "Two-seed distances do not characterize a full generation distribution.",
            "MediaPipe coefficients are continuous proxies, not validated expression "
            "success probabilities.",
            "Identity percentile-bootstrap intervals are exploratory "
            "and not multiplicity adjusted.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "status.json").write_text(
        json.dumps({"status": "complete", "cpu_wall_seconds": summary["cpu_wall_seconds"]}),
        encoding="utf-8",
    )
    return summary


def _figures(tables: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "global_donor_patch_target": "G:donor / P:target",
        "global_target_patch_donor": "G:target / P:donor",
        "k1_full": "1 reference",
        "k4_diverse_full": "4 diverse",
        "k4_repeat_full": "4 repeated",
        "text_only": "Text only",
    }

    summary = tables["identity_summary"]
    evaluators = sorted(summary.evaluator.unique())
    figure, axes = plt.subplots(
        1,
        len(evaluators),
        figsize=(6 * len(evaluators), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, evaluator in zip(axes[0], evaluators, strict=True):
        part = summary.loc[
            (summary.evaluator == evaluator) & (summary.metric == "target_similarity")
        ].sort_values("condition")
        axis.bar(range(len(part)), part.estimate, color="#3f6d9b")
        axis.errorbar(
            range(len(part)),
            part.estimate,
            yerr=np.stack([part.estimate - part.ci95_lower, part.ci95_upper - part.estimate]),
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.set_xticks(
            range(len(part)),
            [labels.get(c, c) for c in part.condition],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        axis.set(
            title=f"{evaluator}: own recognition space", ylabel="Target cosine (identity mean)"
        )
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Automatic diagnostics only; bars are not human likeness scores\n"
        "G = global source; P = patch source",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        figure.savefig(output / f"recognizer_identity_context.{suffix}", dpi=160)
    plt.close(figure)
    if "reference_summary" in tables:
        summary = tables["reference_summary"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for axis, space in zip(axes, ("whole", "face"), strict=True):
            part = summary.loc[summary.space == space]
            conditions = sorted(part.condition.unique())
            for metric, label in (
                ("common_k1_distance", "Common single reference"),
                ("actual_reference_mean_distance", "Mean over actual distinct inputs"),
            ):
                rows = part.loc[part.metric == metric].set_index("condition").reindex(conditions)
                offset = -0.06 if metric == "common_k1_distance" else 0.06
                axis.errorbar(
                    np.arange(len(rows)) + offset,
                    rows.estimate,
                    yerr=np.stack(
                        [rows.estimate - rows.ci95_lower, rows.ci95_upper - rows.estimate]
                    ),
                    fmt="o",
                    label=label,
                    capsize=3,
                )
            axis.set_xticks(
                range(len(conditions)),
                [labels.get(c, c) for c in conditions],
                rotation=30,
                ha="right",
                fontsize=8,
            )
            axis.set(
                title=space + " image comparison", ylabel="LPIPS-Alex v0.1 distance", ylim=(0, 0.7)
            )
            axis.legend(fontsize=8)
        figure.suptitle(
            "Reference proximity: smaller means closer, not a measured copying rate\n"
            "Identity bootstrap 95% intervals; G = global source, P = patch source",
            fontsize=11,
        )
        for suffix in ("png", "svg"):
            figure.savefig(output / f"reference_proximity.{suffix}", dpi=160)
        plt.close(figure)


def analyze_vae_collection(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    figure_dir: Path | None = None,
    resamples: int = 2000,
    seed: int = 20260905,
) -> dict[str, Any]:
    """Aggregate codec diagnostics by identity, keeping alignment methods distinct."""
    evaluation_dir, output_dir = _project_path(evaluation_dir), _project_path(output_dir)
    protected = ROOT / ("artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot")
    if output_dir.is_relative_to(evaluation_dir) or output_dir.is_relative_to(protected):
        raise ValueError("VAE analysis must be outside completed source runs")
    if output_dir.exists():
        raise FileExistsError("VAE analysis output already exists")
    status = json.loads((evaluation_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status", status.get("state")) != "complete":
        raise ValueError("VAE source evaluation is not complete")
    reconstruction = _read(evaluation_dir / "reconstruction_metrics.csv", required=True)
    changes = _read(evaluation_dir / "identity_changes.csv", required=True)
    lpips = _read(evaluation_dir / "lpips_distances.csv", required=True)
    if reconstruction.sample_id.duplicated().any():
        raise ValueError("Duplicate VAE reconstruction sample")
    if changes.duplicated(["sample_id", "evaluator", "comparison"]).any():
        raise ValueError("Duplicate VAE identity comparison")
    if lpips.duplicated(["sample_id", "space"]).any():
        raise ValueError("Duplicate VAE LPIPS pair")
    output_dir.mkdir(parents=True)
    start = time.perf_counter()
    input_hashes = {
        name: hashlib.sha256((evaluation_dir / name).read_bytes()).hexdigest()
        for name in ("reconstruction_metrics.csv", "identity_changes.csv", "lpips_distances.csv")
    }
    (output_dir / "source_hashes.json").write_text(
        json.dumps(input_hashes, indent=2), encoding="utf-8"
    )
    reconstruction["comparison"] = "native_resolution_original_vs_reconstruction"
    numeric = ["float__psnr_db", "float__ssim_rgb", "png__psnr_db", "png__ssim_rgb"]
    infinite_counts = {metric: int(np.isinf(reconstruction[metric]).sum()) for metric in numeric}
    reconstruction[numeric] = reconstruction[numeric].replace([np.inf, -np.inf], np.nan)
    tables = {}
    tables["pixel_identity"], tables["pixel_summary"] = _group_summary(
        reconstruction, ["comparison"], numeric, resamples=resamples, seed=seed
    )
    tables["embedding_identity"], tables["embedding_summary"] = _group_summary(
        changes,
        ["evaluator", "comparison"],
        ["identity_cosine", "identity_cosine_distance"],
        resamples=resamples,
        seed=seed,
    )
    tables["lpips_identity"], tables["lpips_summary"] = _group_summary(
        lpips, ["space"], ["distance"], resamples=resamples, seed=seed
    )
    for name, table in tables.items():
        table.to_csv(output_dir / (name + ".csv"), index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    rows = tables["lpips_summary"].set_index("space").reindex(["native", "whole", "face"])
    axes[0].errorbar(
        range(len(rows)),
        rows.estimate,
        yerr=np.stack([rows.estimate - rows.ci95_lower, rows.ci95_upper - rows.estimate]),
        fmt="o",
        capsize=4,
    )
    axes[0].set_xticks(range(len(rows)), ["Native 640 x 480", "Whole at 256", "Aligned face 256"])
    axes[0].set(title="Reference / VAE reconstruction", ylabel="LPIPS distance", ylim=(0, 0.03))
    rows = tables["embedding_summary"]
    evaluators = sorted(rows.evaluator.unique())
    for index, comparison in enumerate(["fixed_original_alignment", "independent_redetection"]):
        values = rows.loc[(rows.metric == "identity_cosine") & (rows.comparison == comparison)]
        values = values.set_index("evaluator").reindex(evaluators)
        axes[1].errorbar(
            np.arange(len(evaluators)) + (index - 0.5) * 0.12,
            values.estimate,
            yerr=np.stack(
                [values.estimate - values.ci95_lower, values.ci95_upper - values.estimate]
            ),
            fmt="o",
            capsize=4,
            label=comparison.replace("_", " "),
        )
    axes[1].set_xticks(range(len(evaluators)), evaluators)
    axes[1].set(
        title="Original / reconstruction in each recognition space",
        ylabel="Cosine",
        ylim=(0, 1.03),
    )
    axes[1].legend(fontsize=8)
    figure.suptitle(
        "VAE-only diagnostics; 56 photos / 8 identities; identity bootstrap 95% intervals\n"
        "Resolution changes LPIPS scale; codec error is not an additive account of generation loss",
        fontsize=10,
    )
    destinations = [output_dir]
    if figure_dir is not None:
        figure_dir = _project_path(figure_dir)
        if figure_dir.is_relative_to(evaluation_dir) or figure_dir.is_relative_to(protected):
            raise ValueError("Curated figures cannot overwrite source experiment")
        figure_dir.mkdir(parents=True, exist_ok=True)
        destinations.append(figure_dir)
    for destination in destinations:
        for suffix in ("png", "svg"):
            figure.savefig(destination / f"vae_reconstruction_summary.{suffix}", dpi=160)
    plt.close(figure)
    summary = {
        "status": "complete",
        "evidence_type": "DEVELOPMENT_CODEC_DIAGNOSTIC",
        "source_directory": str(evaluation_dir),
        "source_photo_count": len(reconstruction),
        "identity_count": int(reconstruction.identity_id.nunique()),
        "infinite_exact_reconstruction_metric_count": infinite_counts,
        "resamples": resamples,
        "seed": seed,
        "cpu_wall_seconds": time.perf_counter() - start,
        "scope": (
            "Within-recognizer reconstruction checks, not an additive decomposition of likeness."
        ),
    }
    for filename in ("summary.json", "status.json", "resolved_config.json"):
        (output_dir / filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_expression_comparison(
    analyses: dict[str, Path], figure_dir: Path, *, stem: str = "neutral_smile_model_comparison"
) -> dict[str, Any]:
    """Render same-cohort automatic expression contrasts; never plot human D/E."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = _project_path(figure_dir)
    protected = ROOT / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot"
    if figure_dir.is_relative_to(protected):
        raise ValueError("Cannot write comparison figures into frozen pilot")
    if not analyses or Path(stem).name != stem:
        raise ValueError("Need named analyses and a plain filename stem")
    summaries, expression_means = [], []
    expected_ids = None
    for model, path in analyses.items():
        path = _project_path(path)
        config = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        if not config["new_expression_protocol"] or status["status"] != "complete":
            raise ValueError("Comparison requires completed single-expression analyses")
        identities = set(_read(path / "expression_identity.csv", required=True).identity_id)
        if expected_ids is not None and identities != expected_ids:
            raise ValueError("Model comparison cohorts differ; do not combine smoke with full runs")
        expected_ids = identities
        frame = _read(path / "expression_paired_summary.csv", required=True)
        frame["model"] = model
        summaries.append(frame)
        frame = _read(path / "expression_summary.csv", required=True)
        frame["model"] = model
        expression_means.append(frame)
    paired, absolute = pd.concat(summaries), pd.concat(expression_means)
    model_order = list(analyses)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axis = axes[0, 0]
    for offset, prompt in ((-0.08, "neutral"), (0.08, "smile")):
        rows = (
            absolute.loc[(absolute.metric == "mouth_smile_mean") & (absolute.prompt_id == prompt)]
            .set_index("model")
            .reindex(model_order)
        )
        axis.errorbar(
            np.arange(len(model_order)) + offset,
            rows.estimate,
            yerr=np.stack([rows.estimate - rows.ci95_lower, rows.ci95_upper - rows.estimate]),
            fmt="o",
            capsize=4,
            label=prompt,
        )
    axis.set(
        title="Continuous smile proxy",
        ylabel="MediaPipe mean mouth-smile coefficient",
        ylim=(0, 1),
    )
    axis.legend()
    for axis, metric, evaluator, title in (
        (axes[0, 1], "mouth_smile_mean", None, "Smile proxy: paired instruction change"),
        (axes[1, 0], "target_similarity", "adaface", "AdaFace: paired identity change"),
        (axes[1, 1], "target_similarity", "lvface", "LVFace: paired identity change"),
    ):
        selected = paired.loc[paired.metric == metric]
        if evaluator is not None:
            selected = selected.loc[selected.evaluator == evaluator]
        rows = selected.set_index("model").reindex(model_order)
        axis.errorbar(
            range(len(model_order)),
            rows.estimate,
            yerr=np.stack([rows.estimate - rows.ci95_lower, rows.ci95_upper - rows.estimate]),
            fmt="o",
            capsize=4,
        )
        axis.axhline(0, color="gray", linewidth=1, linestyle="--")
        axis.set(title=title, ylabel="Smile minus neutral")
        if evaluator is None:
            axis.set_ylim(-0.1, 1)
        else:
            extent = max(0.05, 1.1 * np.nanmax(np.abs(rows[["ci95_lower", "ci95_upper"]])))
            axis.set_ylim(-extent, extent)
    for axis in axes.flat:
        axis.set_xticks(range(len(model_order)), model_order)
        axis.set_xlim(-0.5, len(model_order) - 0.5)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Automatic development diagnostics on {len(expected_ids)} shared FEI identities\n"
        "Identity bootstrap 95% intervals; no self-rated likeness D/E was collected",
        fontsize=11,
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(figure_dir / f"{stem}.{suffix}", dpi=160)
    plt.close(figure)
    paired.to_csv(figure_dir / f"{stem}.csv", index=False)
    return {
        "figure_stem": str(figure_dir / stem),
        "models": model_order,
        "shared_identity_count": len(expected_ids),
    }


def compare_smoke_full_generation(
    smoke_dir: Path, full_dir: Path, output_dir: Path, *, expected_smoke_count: int = 16
) -> dict[str, Any]:
    """Verify fixed matched cells and record file/pixel repeatability without re-generating.

    No mismatch is hidden by a success threshold: exact hashes and observed pixel
    differences are both reported. This comparison uses no recognizer or human vote.
    """
    from PIL import Image

    smoke_dir, full_dir = _project_path(smoke_dir), _project_path(full_dir)
    output_dir = _project_path(output_dir)
    protected = ROOT / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot"
    if any(output_dir.is_relative_to(p) for p in (smoke_dir, full_dir, protected)):
        raise ValueError("Repeatability analysis cannot modify source generations")
    if output_dir.exists():
        raise FileExistsError("Repeatability output already exists")

    def manifest(path):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate generation sample ID")
        return {row["sample_id"]: row for row in rows}

    smoke_path, full_path = (
        smoke_dir / "generation_manifest.jsonl",
        full_dir / "generation_manifest.jsonl",
    )
    smoke, full = manifest(smoke_path), manifest(full_path)
    if len(smoke) != expected_smoke_count or not set(smoke).issubset(full):
        raise ValueError("Smoke/full matrix does not contain the frozen matching cells")
    fields = (
        "identity_id",
        "condition",
        "prompt_id",
        "prompt",
        "base_seed",
        "effective_seed",
        "latent_sha256",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "negative_prompt",
        "reference_sample_ids",
        "global_reference_sample_ids",
        "patch_reference_sample_ids",
        "global_embedding_identity_id",
        "patch_image_identity_id",
    )
    start, result = time.perf_counter(), []
    for sample_id in sorted(smoke):
        left, right = smoke[sample_id], full[sample_id]
        if left["status"] != "complete" or right["status"] != "complete":
            raise ValueError("Repeatability comparison requires completed fixed cells")
        missing_fields = [field for field in fields if field not in left or field not in right]
        mismatches = [field for field in fields if left.get(field) != right.get(field)]
        if missing_fields or mismatches:
            raise ValueError(
                f"Repeatability pairing mismatch for {sample_id}: {missing_fields + mismatches}"
            )
        pixels = []
        for base, row in ((smoke_dir, left), (full_dir, right)):
            path = (base / row["output_relative_path"]).resolve()
            if not path.is_relative_to(base):
                raise ValueError("Generation output escapes source directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["output_sha256"]:
                raise ValueError("Generated output hash mismatch")
            with Image.open(path) as image:
                pixels.append(np.asarray(image.convert("RGB"), dtype=np.int16))
        if pixels[0].shape != pixels[1].shape:
            raise ValueError("Matched output dimensions differ")
        difference = np.abs(pixels[0] - pixels[1])
        result.append(
            {
                "sample_id": sample_id,
                "identity_id": left["identity_id"],
                "prompt_id": left["prompt_id"],
                "base_seed": left["base_seed"],
                "latent_sha256": left["latent_sha256"],
                "smoke_output_sha256": left["output_sha256"],
                "full_output_sha256": right["output_sha256"],
                "file_exact_match": left["output_sha256"] == right["output_sha256"],
                "pixel_exact_match": bool(np.array_equal(pixels[0], pixels[1])),
                "mean_absolute_channel_error_0_255": float(difference.mean()),
                "maximum_absolute_channel_error_0_255": int(difference.max()),
                "changed_pixel_fraction": float((difference != 0).any(axis=2).mean()),
            }
        )
    output_dir.mkdir(parents=True)
    pd.DataFrame(result).to_csv(output_dir / "smoke_full_pixel_comparison.csv", index=False)
    summary = {
        "status": "complete",
        "expected_smoke_count": expected_smoke_count,
        "matched_cell_count": len(result),
        "file_exact_match_count": sum(r["file_exact_match"] for r in result),
        "pixel_exact_match_count": sum(r["pixel_exact_match"] for r in result),
        "maximum_absolute_channel_error_0_255": max(
            r["maximum_absolute_channel_error_0_255"] for r in result
        ),
        "paired_fields": list(fields),
        "cpu_wall_seconds": time.perf_counter() - start,
        "source_manifest_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (smoke_path, full_path)
        },
        "scope": "Fixed-cell numerical repeatability only; not additional independent identities.",
    }
    for name in ("summary.json", "status.json", "resolved_config.json"):
        (output_dir / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-expression-protocol", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_auto_collection(
                args.evaluation, args.output, new_expression_protocol=args.new_expression_protocol
            )
        )
    )


if __name__ == "__main__":
    main()
