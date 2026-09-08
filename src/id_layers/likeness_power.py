"""CPU-only, explicitly synthetic planning scenarios for the SELF D/E endpoints.

Eight rounded/clipped 1..7 responses are generated per identity. Identity-level
likeness and generation-gap variability plus candidate noise induce dependence.
No empirical participant variance is estimated here. Unfamiliar-gallery power
requires a separate crossed-rater design and is intentionally not inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any

import numpy as np
from scipy.stats import t

POWER_SCHEMA_VERSION = "likeness-power-v1"
DEFAULT_POWER_CONFIG = {
    "identity_counts": [12, 24, 40, 60],
    "mean_gaps": [0.0, 0.25, 0.5, 1.0],
    "identity_gap_sds": [0.5, 1.0, 1.5],
    "missing_rates": [0.0, 0.1, 0.25],
    "missing_mechanisms": ["mcar"],
    "expression_extra_gaps": [0.0, 0.5],
    "replicates": 2000,
    "seed": 20260905,
    "population_reference_identities": 200000,
    "real_mean": 5.5,
    "identity_baseline_sd": 0.5,
    "candidate_noise_sd": 0.65,
    "expression_gap_sd": 0.5,
}


def _scores(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    gap: float,
    gap_sd: float,
    extra: float,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    baseline = cfg["real_mean"] + rng.normal(0, cfg["identity_baseline_sd"], shape)
    identity_gap = gap + rng.normal(0, gap_sd, shape)
    expression_gap = extra + rng.normal(0, cfg["expression_gap_sd"], shape)
    # Cell order R0,G0,R1,G1. D's latent expectation is gap; E's is extra.
    means = np.stack(
        [
            baseline,
            baseline - identity_gap + expression_gap / 2,
            baseline,
            baseline - identity_gap - expression_gap / 2,
        ],
        axis=-1,
    )
    scores = means[..., None] + rng.normal(0, cfg["candidate_noise_sd"], shape + (4, 2))
    return np.clip(np.rint(scores), 1, 7)


def _effects(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.isfinite(scores)
    count = available.sum(axis=-1)
    cells = np.divide(
        np.nansum(scores, axis=-1), count, out=np.full(count.shape, np.nan), where=count > 0
    )
    d = (cells[..., 0] - cells[..., 1] + cells[..., 2] - cells[..., 3]) / 2
    e = cells[..., 2] - cells[..., 3] - cells[..., 0] + cells[..., 1]
    return d, e


def _intervals(effects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = np.isfinite(effects).sum(axis=1)
    means = np.divide(
        np.nansum(effects, axis=1), n, out=np.full(n.shape, np.nan, dtype=float), where=n > 0
    )
    ss = np.nansum((effects - means[:, None]) ** 2, axis=1)
    variance = np.divide(ss, n - 1, out=np.full(n.shape, np.nan, dtype=float), where=n > 1)
    half = t.ppf(0.975, np.maximum(n - 1, 1)) * np.sqrt(variance / np.maximum(n, 1))
    return means, means - half, means + half, n


def run_power_grid(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return a frozen-grid table; power targets zero, not an effect-size threshold.

    The planning CI is identity-mean Student-t (not bootstrap), reported explicitly.
    Nominal coverage targets a large independent Monte Carlo population reference
    after rounding/clipping. Its Monte Carlo SE is included. MCAR acts independently
    on each candidate rating. Optional low_score_mnar preferentially hides low
    scores; requested and realized missingness are both reported.
    """
    cfg = {**DEFAULT_POWER_CONFIG, **(dict(config) if config else {})}
    unknown = set(cfg) - set(DEFAULT_POWER_CONFIG)
    if unknown:
        raise ValueError(f"Unknown power configuration keys: {sorted(unknown)}")
    if int(cfg["replicates"]) < 2 or int(cfg["population_reference_identities"]) < 100:
        raise ValueError("Need >=2 replicates and >=100 population reference identities")
    if any(int(n) != n or n < 2 for n in cfg["identity_counts"]):
        raise ValueError("Identity counts must be integers >=2")
    if any(not 0 <= rate < 1 for rate in cfg["missing_rates"]):
        raise ValueError("Missing rates must lie in [0,1)")
    if any(mechanism not in {"mcar", "low_score_mnar"} for mechanism in cfg["missing_mechanisms"]):
        raise ValueError("Unknown missing mechanism")
    for key in ("identity_baseline_sd", "candidate_noise_sd", "expression_gap_sd"):
        if cfg[key] < 0:
            raise ValueError("Standard deviations cannot be negative")
    if any(sd < 0 for sd in cfg["identity_gap_sds"]):
        raise ValueError("Identity gap SDs cannot be negative")
    results = []
    scenarios = list(
        product(cfg["mean_gaps"], cfg["identity_gap_sds"], cfg["expression_extra_gaps"])
    )
    reference_seeds = np.random.SeedSequence(cfg["seed"]).spawn(len(scenarios) * 2)
    for scenario_number, (gap, gap_sd, extra) in enumerate(scenarios):
        ref_rng = np.random.default_rng(reference_seeds[scenario_number * 2])
        reference = _effects(
            _scores(
                ref_rng, (int(cfg["population_reference_identities"]),), gap, gap_sd, extra, cfg
            )
        )
        targets = {
            name: (float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(vals.size)))
            for name, vals in zip(("D", "E"), reference, strict=True)
        }
        combinations = list(
            product(cfg["identity_counts"], cfg["missing_rates"], cfg["missing_mechanisms"])
        )
        children = reference_seeds[scenario_number * 2 + 1].spawn(len(combinations))
        for child, (n, missing, mechanism) in zip(children, combinations, strict=True):
            rng = np.random.default_rng(child)
            raw = _scores(rng, (int(cfg["replicates"]), int(n)), gap, gap_sd, extra, cfg)
            full_effects = _effects(raw)
            probabilities = np.full(raw.shape, missing)
            if mechanism == "low_score_mnar":
                probabilities = np.clip(missing * (1 + (4 - raw) / 4), 0, 0.95)
            absent = rng.random(raw.shape) < probabilities
            observed = np.where(absent, np.nan, raw)
            effects = _effects(observed)
            for endpoint, full, values in zip(("D", "E"), full_effects, effects, strict=True):
                means, low, high, available_n = _intervals(values)
                valid = np.isfinite(low) & np.isfinite(high)
                count = int(valid.sum())
                target, target_se = targets[endpoint]
                reject = valid & ((low > 0) | (high < 0))
                power = float(np.mean(reject))
                coverage = (
                    float(np.mean((low[valid] <= target) & (high[valid] >= target)))
                    if count
                    else None
                )
                results.append(
                    {
                        "schema_version": POWER_SCHEMA_VERSION,
                        "evidence_type": "SIMULATION_ONLY",
                        "endpoint_group": "self",
                        "endpoint": endpoint,
                        "identity_count": int(n),
                        "latent_mean_D": float(gap),
                        "latent_mean_E": float(extra),
                        "identity_gap_sd": float(gap_sd),
                        "requested_missing_rate": float(missing),
                        "missing_mechanism": mechanism,
                        "realized_missing_rate": float(absent.mean()),
                        "population_observed_scale_effect": target,
                        "population_reference_mc_se": target_se,
                        "mean_complete_simulated_effect": float(full.mean()),
                        "mean_analyzed_effect": float(np.mean(means[np.isfinite(means)]))
                        if np.isfinite(means).any()
                        else None,
                        "bias_to_population_effect": float(np.mean(means[np.isfinite(means)]))
                        - target
                        if np.isfinite(means).any()
                        else None,
                        "two_sided_zero_test_power": power,
                        "power_mc_se": float(np.sqrt(power * (1 - power) / cfg["replicates"])),
                        "ci95_population_coverage_given_estimable": coverage,
                        "mean_ci_width_given_estimable": float(np.mean(high[valid] - low[valid]))
                        if count
                        else None,
                        "mean_analyzable_identity_count": float(available_n.mean()),
                        "replicates": int(cfg["replicates"]),
                        "estimable_replicates": count,
                        "seed": int(cfg["seed"]),
                        "scenario_seed_spawn_key": list(child.spawn_key),
                        "interval_method": (
                            "identity-mean Student-t planning approximation; validate final "
                            "frozen analysis separately"
                        ),
                        "scope": (
                            "assumed self-score process, not measured human variance, no "
                            "unfamiliar-rater power or final sample-size decision"
                        ),
                    }
                )
    return results
