"""Prospective single-candidate likeness protocol (independent of human-eval v2).

Record schemas are deliberately explicit: identities own disjoint photo roles;
candidates reference photographs or frozen generation cells; assignments identify
expected responses. Missing responses never become zero. The self endpoint uses
an identity-mean t interval and bootstrap sensitivity; unfamiliar-gallery
intervals use a crossed identity/rater
pigeonhole bootstrap as an exploratory sensitivity analysis, not a fitted ordinal
mixed model or a ready-made confirmatory statistical specification.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

SCHEMA_VERSION = "likeness-study-v1"
ROLES = {"input": 1, "gallery": 3, "real_neutral": 2, "real_smile": 2}
CELLS = ("R0", "G0", "R1", "G1")
COEFFICIENTS = {"D": (0.5, -0.5, 0.5, -0.5), "E": (-1.0, 1.0, 1.0, -1.0)}
Record = Mapping[str, Any]


def _index(rows: Sequence[Record], key: str) -> dict[str, Record]:
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid {key}")
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return dict(sorted(result.items()))


def _check_hash(row: Record, root: Path | None = None) -> None:
    digest = row.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("sha256 must be a lowercase 64-character digest")
    if root is not None:
        path = (root / str(row["path"])).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Asset path escapes verification root")
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Hash mismatch: {path}")


def validate_photo_roles(
    identities: Sequence[Record], photos: Sequence[Record], *, root: Path | None = None
) -> dict[str, Any]:
    """Require eight distinct real captures per identity and identity-disjoint splits.

    identities: identity_id, split, self_rater_id, analysis_consent (bool).
    photos: sample_id, identity_id, role, sha256, optional path (required with root).
    Decoded-pixel duplicates should also be rejected upstream: file hashes alone
    cannot establish that two differently encoded files are distinct captures.
    """
    ids, samples = _index(identities, "identity_id"), _index(photos, "sample_id")
    if not ids:
        raise ValueError("At least one identity is required")
    self_ids = []
    for ident in ids.values():
        if ident.get("split") not in {"development", "confirmation", "intervention_confirmation"}:
            raise ValueError("Unknown or missing identity split")
        if not isinstance(ident.get("analysis_consent"), bool):
            raise ValueError("analysis_consent must be an explicit boolean")
        if not isinstance(ident.get("self_rater_id"), str) or not ident["self_rater_id"]:
            raise ValueError("Missing self_rater_id")
        self_ids.append(ident["self_rater_id"])
    if len(self_ids) != len(set(self_ids)):
        raise ValueError("A self rater cannot own multiple identities")
    counts: dict[str, Counter] = defaultdict(Counter)
    hashes, capture_ids, pixel_hashes = set(), set(), set()
    for row in samples.values():
        identity = row["identity_id"]
        if identity not in ids or row.get("role") not in ROLES:
            raise ValueError("Unknown identity or photo role")
        if "split" in row and row["split"] != ids[identity]["split"]:
            raise ValueError("Photo split differs from identity split")
        _check_hash(row, root)
        if row["sha256"] in hashes:
            raise ValueError("Duplicate photograph hash across roles or identities")
        hashes.add(row["sha256"])
        for field, seen in (("capture_id", capture_ids), ("pixel_sha256", pixel_hashes)):
            if row.get(field) is not None:
                if row[field] in seen:
                    raise ValueError(f"Duplicate photograph {field}")
                seen.add(row[field])
        counts[identity][row["role"]] += 1
    for identity in ids:
        if dict(counts[identity]) != ROLES:
            raise ValueError(
                f"Incorrect eight-photo role counts for {identity}: {counts[identity]}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_count": len(ids),
        "photo_count": len(samples),
    }


def validate_candidates(
    identities: Sequence[Record],
    photos: Sequence[Record],
    candidates: Sequence[Record],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate complete 2-expression x 2-replicate matrices, retaining failed cells.

    candidate fields: candidate_id, identity_id, kind ('real'/'generated'), model
    (null for real), expression ('neutral'/'smile'), replicate (1/2), status
    ('ok'/'generation_failed'/'image_unavailable'), sha256/path for existing images.
    Real rows name photo_sample_id; generated rows name input_sample_ids (one
    frozen input), integer seed. Failure rows require failure_reason and no hash.
    """
    validate_photo_roles(identities, photos, root=root)
    ids, samples = _index(identities, "identity_id"), _index(photos, "sample_id")
    indexed = _index(candidates, "candidate_id")
    cells, models, real_sample_ids = set(), set(), set()
    generation_pairing: dict[tuple[str, str, int], tuple[Any, ...]] = {}
    for row in indexed.values():
        identity, kind = row.get("identity_id"), row.get("kind")
        if identity not in ids or kind not in {"real", "generated"}:
            raise ValueError("Unknown candidate identity or kind")
        if row.get("expression") not in {"neutral", "smile"} or row.get("replicate") not in (1, 2):
            raise ValueError("Invalid candidate expression or replicate")
        if isinstance(row["replicate"], bool):
            raise ValueError("Boolean replicate is not valid")
        model = row.get("model")
        if kind == "real":
            if model is not None:
                raise ValueError("Real candidates are shared across models, model must be null")
            photo = samples.get(row.get("photo_sample_id"))
            if (
                photo is None
                or photo["identity_id"] != identity
                or photo["role"] != "real_" + row["expression"]
            ):
                raise ValueError("Real candidate photo role/identity mismatch")
            if row.get("status") == "ok" and row.get("sha256") != photo["sha256"]:
                raise ValueError("Real candidate hash differs from photo")
            if row["photo_sample_id"] in real_sample_ids:
                raise ValueError("Real candidate photograph reused across replicates")
            real_sample_ids.add(row["photo_sample_id"])
        else:
            if not isinstance(model, str) or not model:
                raise ValueError("Generated candidate requires model")
            models.add(model)
            inputs = row.get("input_sample_ids")
            if not isinstance(inputs, list) or len(inputs) != 1:
                raise ValueError("K=1 generation requires exactly one input sample")
            photo = samples.get(inputs[0])
            if photo is None or photo["identity_id"] != identity or photo["role"] != "input":
                raise ValueError("Input candidate role/identity leakage")
            if not isinstance(row.get("seed"), int) or isinstance(row["seed"], bool):
                raise ValueError("Generated candidate requires integer seed")
            pair_key = (identity, model, row["replicate"])
            pairing = (inputs[0], row["seed"])
            if pair_key in generation_pairing and generation_pairing[pair_key] != pairing:
                raise ValueError("Expression pair must share input and seed within model")
            generation_pairing[pair_key] = pairing
        key = (identity, kind, model, row["expression"], row["replicate"])
        if key in cells:
            raise ValueError("Duplicate candidate cell")
        cells.add(key)
        if row.get("status") == "ok":
            _check_hash(row, root)
        elif row.get("status") in {"generation_failed", "image_unavailable"}:
            if not row.get("failure_reason") or row.get("sha256") is not None:
                raise ValueError("Failed candidate needs reason and no image hash")
            if kind == "real" and row["status"] == "generation_failed":
                raise ValueError("A real photo can be image_unavailable, not generation_failed")
        else:
            raise ValueError("Unknown candidate status")
    if not models:
        raise ValueError("At least one model is required")
    expected = {
        (identity, kind, model, expression, replicate)
        for identity in ids
        for kind, model in [("real", None)] + [("generated", m) for m in sorted(models)]
        for expression in ("neutral", "smile")
        for replicate in (1, 2)
    }
    if cells != expected:
        raise ValueError("Incomplete or unexpected candidate matrix; retain failed rows")
    return {
        "candidate_count": len(indexed),
        "models": sorted(models),
        "failed_candidate_count": sum(r["status"] != "ok" for r in indexed.values()),
    }


def parse_responses(
    identities: Sequence[Record],
    candidates: Sequence[Record],
    raters: Sequence[Record],
    assignments: Sequence[Record],
    responses: Sequence[Record],
) -> list[dict[str, Any]]:
    """Join by assignment ID; reject duplicates, unknown IDs, wrong self and familiarity.

    raters: rater_id, known_identity_ids (explicit list; own identity counts as known).
    assignments: assignment_id, candidate_id, candidate_sha256, rater_id, group
    ('self'/'unfamiliar'). responses: assignment_id, status ('rated'/'unjudgeable'/
    'missing'), score (integer 1..7 only for rated). Unreturned assignments are missing.
    """
    ids = _index(identities, "identity_id")
    candidates_by_id = _index(candidates, "candidate_id")
    raters_by_id = _index(raters, "rater_id")
    assignment_map, response_map = (
        _index(assignments, "assignment_id"),
        _index(responses, "assignment_id"),
    )
    if set(response_map) - set(assignment_map):
        raise ValueError("Response references unknown assignment")
    seen, result = set(), []
    for assignment_id, row in assignment_map.items():
        candidate = candidates_by_id.get(row.get("candidate_id"))
        rater = raters_by_id.get(row.get("rater_id"))
        if candidate is None or rater is None or row.get("group") not in {"self", "unfamiliar"}:
            raise ValueError("Unknown assignment candidate, rater or group")
        if candidate["status"] != "ok":
            raise ValueError("Cannot assign a nonexistent candidate image")
        if row.get("candidate_sha256") != candidate["sha256"]:
            raise ValueError("Assignment candidate hash mismatch")
        key = (row["candidate_id"], row["rater_id"])
        if key in seen:
            raise ValueError("Duplicate candidate/rater assignment")
        seen.add(key)
        identity = candidate["identity_id"]
        if row["group"] == "self" and row["rater_id"] != ids[identity]["self_rater_id"]:
            raise ValueError("Self response must be from that identity's one self rater")
        known = rater.get("known_identity_ids")
        if not isinstance(known, list):
            raise ValueError("Rater must explicitly declare known_identity_ids")
        if row["group"] == "unfamiliar" and (
            identity in known or row["rater_id"] == ids[identity]["self_rater_id"]
        ):
            raise ValueError("Unfamiliar assignment involves a known or own identity")
        response = response_map.get(assignment_id, {"status": "missing", "score": None})
        for field in ("candidate_id", "candidate_sha256", "rater_id", "group"):
            if field in response and response[field] != row[field]:
                raise ValueError(f"Response {field} differs from frozen assignment")
        status, score = response.get("status"), response.get("score")
        if status == "rated":
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 7:
                raise ValueError("Rated score must be an integer from 1 to 7")
        elif status in {"missing", "unjudgeable"}:
            if score is not None:
                raise ValueError("Missing or unjudgeable responses must have null score")
        else:
            raise ValueError("Unknown response status")
        result.append({**row, "identity_id": identity, "status": status, "score": score})
    return result


def _candidate_summary(
    candidate: Record, rows: Sequence[Record], weights: Mapping[str, int] | None = None
) -> dict[str, Any]:
    valid = [r for r in rows if r["status"] == "rated"]
    weighted = [
        (r["score"], 1 if weights is None else weights.get(r["rater_id"], 0)) for r in valid
    ]
    denom = sum(w for _, w in weighted)
    estimate = sum(v * w for v, w in weighted) / denom if denom else None
    total = len(rows)
    values = [r["score"] for r in valid]
    if candidate["status"] != "ok":
        low = high = None
    elif total:
        low = (sum(values) + (total - len(values))) / total
        high = (sum(values) + 7 * (total - len(values))) / total
    else:
        low, high = 1.0, 7.0
    return {
        "estimate": estimate,
        "lower": low,
        "upper": high,
        "assigned_count": total,
        "valid_count": len(valid),
        "missing_count": total - len(valid),
        "unassigned": total == 0,
        "status": candidate["status"],
    }


def _identity_effects(
    identity: str,
    model: str,
    candidates: Sequence[Record],
    ratings: Mapping[str, Sequence[Record]],
    weights: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    cells = {}
    for cell in CELLS:
        kind = "real" if cell[0] == "R" else "generated"
        expression = "neutral" if cell[1] == "0" else "smile"
        selected = [
            c
            for c in candidates
            if c["identity_id"] == identity
            and c["kind"] == kind
            and c["expression"] == expression
            and (kind == "real" or c["model"] == model)
        ]
        values = [
            _candidate_summary(c, ratings.get(c["candidate_id"], []), weights) for c in selected
        ]
        observed = [v["estimate"] for v in values if v["estimate"] is not None]
        all_existing = all(v["status"] == "ok" for v in values)
        cells[cell] = {
            "estimate": float(np.mean(observed)) if observed else None,
            "lower": float(np.mean([v["lower"] for v in values])) if all_existing else None,
            "upper": float(np.mean([v["upper"] for v in values])) if all_existing else None,
            "candidate_count": len(values),
            "valid_candidate_count": len(observed),
            "existing_candidate_count": sum(v["status"] == "ok" for v in values),
            "assigned_rating_count": sum(v["assigned_count"] for v in values),
            "valid_rating_count": sum(v["valid_count"] for v in values),
            "missing_rating_count": sum(v["missing_count"] for v in values),
            "unassigned_candidate_count": sum(v["unassigned"] for v in values),
        }
    estimable = all(cells[c]["estimate"] is not None for c in CELLS)
    boundable = all(cells[c]["lower"] is not None for c in CELLS)
    result = {
        "identity_id": identity,
        "model": model,
        "cells": cells,
        "estimable": estimable,
        "all_eight_candidates_scored": all(cells[c]["valid_candidate_count"] == 2 for c in CELLS),
        "all_expected_ratings_present": all(
            cells[c]["missing_rating_count"] == 0 and cells[c]["unassigned_candidate_count"] == 0
            for c in CELLS
        ),
        "bounds_eligible_all_images_exist": boundable,
    }
    for name, coefficients in COEFFICIENTS.items():
        result[name] = (
            sum(a * cells[c]["estimate"] for a, c in zip(coefficients, CELLS, strict=True))
            if estimable
            else None
        )
        result[name + "_lower"] = (
            sum(
                a * cells[c]["lower" if a > 0 else "upper"]
                for a, c in zip(coefficients, CELLS, strict=True)
            )
            if boundable
            else None
        )
        result[name + "_upper"] = (
            sum(
                a * cells[c]["upper" if a > 0 else "lower"]
                for a, c in zip(coefficients, CELLS, strict=True)
            )
            if boundable
            else None
        )
    return result


def analyze_likeness_study(
    identities: Sequence[Record],
    photos: Sequence[Record],
    candidates: Sequence[Record],
    raters: Sequence[Record],
    assignments: Sequence[Record],
    responses: Sequence[Record],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 20260905,
) -> dict[str, Any]:
    """Return separate endpoints, identity-equal D/E, response counts and bounds.

    Positive D = real minus generated likeness; positive E = extra smile gap.
    Missing bounds include ALL consenting identities with all eight images, even
    identities with no valid responses. Failed image cells remain in the ledger
    and are excluded from numeric bounds (nonexistent likeness is not imputed).
    """
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    validation = validate_candidates(identities, photos, candidates)
    joined = parse_responses(identities, candidates, raters, assignments, responses)
    ids = _index(identities, "identity_id")
    eligible_ids = [i for i, row in ids.items() if row["analysis_consent"]]
    candidate_rows = sorted(candidates, key=lambda c: c["candidate_id"])
    results, summaries = [], []
    rng = np.random.default_rng(seed)
    for group in ("self", "unfamiliar"):
        grouped: dict[str, list[Record]] = defaultdict(list)
        group_rows = [r for r in joined if r["group"] == group and r["identity_id"] in eligible_ids]
        for row in group_rows:
            grouped[row["candidate_id"]].append(row)
        rater_ids = sorted({r["rater_id"] for r in group_rows})
        for model in validation["models"]:
            effects = [_identity_effects(i, model, candidate_rows, grouped) for i in eligible_ids]
            results.extend({**r, "group": group} for r in effects)
            resampled = {"D": [], "E": []}
            if len(eligible_ids) >= 2:
                for _ in range(bootstrap_resamples):
                    identity_counts = Counter(rng.choice(eligible_ids, len(eligible_ids)).tolist())
                    if group == "unfamiliar" and rater_ids:
                        rater_counts = Counter(rng.choice(rater_ids, len(rater_ids)).tolist())
                        replicate = [
                            _identity_effects(i, model, candidate_rows, grouped, rater_counts)
                            for i in identity_counts
                        ]
                    else:
                        replicate = [r for r in effects if r["identity_id"] in identity_counts]
                    for name in ("D", "E"):
                        valid = [r for r in replicate if r[name] is not None]
                        denom = sum(identity_counts[r["identity_id"]] for r in valid)
                        if denom:
                            resampled[name].append(
                                sum(r[name] * identity_counts[r["identity_id"]] for r in valid)
                                / denom
                            )
            for name in ("D", "E"):
                valid = [r[name] for r in effects if r[name] is not None]
                complete = [r[name] for r in effects if r["all_eight_candidates_scored"]]
                bound_rows = [r for r in effects if r["bounds_eligible_all_images_exist"]]
                draws = resampled[name]
                bootstrap_interval = (
                    np.quantile(draws, [0.025, 0.975]).tolist()
                    if len(valid) >= 2 and draws
                    else None
                )
                interval = bootstrap_interval
                if group == "self" and len(valid) >= 2:
                    half = float(
                        t.ppf(0.975, len(valid) - 1) * np.std(valid, ddof=1) / np.sqrt(len(valid))
                    )
                    interval = [float(np.mean(valid) - half), float(np.mean(valid) + half)]
                summaries.append(
                    {
                        "group": group,
                        "model": model,
                        "endpoint": name,
                        "consenting_identity_count": len(eligible_ids),
                        "estimable_identity_count": len(valid),
                        "estimate": float(np.mean(valid)) if valid else None,
                        "ci95": interval,
                        "bootstrap_sensitivity_ci95": bootstrap_interval,
                        "bootstrap_valid_replicates": len(draws),
                        "bootstrap_requested_replicates": bootstrap_resamples,
                        "interval_method": (
                            "identity-mean Student-t, identity percentile bootstrap sensitivity"
                        )
                        if group == "self"
                        else "crossed identity/rater pigeonhole percentile bootstrap (sensitivity)",
                        "complete_eight_candidate_identity_count": len(complete),
                        "complete_eight_candidate_estimate": float(np.mean(complete))
                        if complete
                        else None,
                        "bounds_identity_count": len(bound_rows),
                        "excluded_missing_image_identity_count": len(effects) - len(bound_rows),
                        "missing_response_identification_bounds": [
                            float(np.mean([r[name + "_lower"] for r in bound_rows])),
                            float(np.mean([r[name + "_upper"] for r in bound_rows])),
                        ]
                        if bound_rows
                        else None,
                        "rater_count": len(rater_ids),
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "synthetic"
        if all(i.get("data_kind") == "synthetic" for i in identities)
        else "observed_ratings",
        "validation": validation,
        "seed": seed,
        "identity_results": results,
        "summary": summaries,
        "response_counts": {
            group: dict(Counter(r["status"] for r in joined if r["group"] == group))
            for group in ("self", "unfamiliar")
        },
        "withdrawn_identity_count": sum(not i["analysis_consent"] for i in identities),
        "dependency_notes": {
            "self": (
                "self rater and identity coincide; do not fit separate independent random effects"
            ),
            "unfamiliar": (
                "same canonical rater IDs retained across identities; bootstrap samples "
                "identity and rater dimensions independently"
            ),
            "models": (
                "real candidates shared; separate model summaries do not constitute "
                "independent model contrasts"
            ),
        },
        "confirmatory_limitation": (
            "Ordinal/mixed-model specification, simplification order and multiplicity family "
            "require freezing before real confirmation. Crossed bootstrap is an exploratory "
            "sensitivity analysis."
        ),
    }


def synthetic_study_example(identity_count: int = 12, seed: int = 20260905) -> dict[str, Any]:
    """Generate a labelled schema/analysis demonstration, with NO actual images or people.

    Digests hash synthetic IDs only and must never be presented as real image
    provenance. The final synthetic identity has no self responses to exercise
    the all-identities missingness bounds; strangers share canonical IDs.
    """
    if identity_count < 2:
        raise ValueError("Synthetic demonstration needs at least two identities")
    result: dict[str, Any] = {
        key: []
        for key in ("identities", "photos", "candidates", "raters", "assignments", "responses")
    }
    rng = np.random.default_rng(seed)
    strangers = [f"SYNTHETIC_STRANGER_{k}" for k in range(5)]
    result["raters"].extend({"rater_id": r, "known_identity_ids": []} for r in strangers)
    for index in range(identity_count):
        identity, self_rater = f"SYNTHETIC_ID_{index:03d}", f"SYNTHETIC_SELF_{index:03d}"
        result["identities"].append(
            {
                "identity_id": identity,
                "split": "development",
                "self_rater_id": self_rater,
                "analysis_consent": True,
                "data_kind": "synthetic",
            }
        )
        result["raters"].append({"rater_id": self_rater, "known_identity_ids": [identity]})
        by_role = defaultdict(list)
        for role, count in ROLES.items():
            for replicate in range(1, count + 1):
                sample_id = f"{identity}_{role}_{replicate}"
                photo = {
                    "sample_id": sample_id,
                    "identity_id": identity,
                    "role": role,
                    "capture_id": sample_id,
                    "sha256": sha256(sample_id.encode()).hexdigest(),
                    "synthetic_placeholder": True,
                }
                result["photos"].append(photo)
                by_role[role].append(photo)
        baseline, gap = 5.5 + rng.normal(0, 0.3), 0.5 + rng.normal(0, 0.6)
        for expression in ("neutral", "smile"):
            for kind in ("real", "generated"):
                for replicate in (1, 2):
                    candidate_id = f"{identity}_{kind}_{expression}_{replicate}"
                    candidate = {
                        "candidate_id": candidate_id,
                        "identity_id": identity,
                        "kind": kind,
                        "model": "SYNTHETIC_MODEL" if kind == "generated" else None,
                        "expression": expression,
                        "replicate": replicate,
                        "status": "ok",
                        "synthetic_placeholder": True,
                    }
                    if kind == "real":
                        photo = by_role["real_" + expression][replicate - 1]
                        candidate.update(photo_sample_id=photo["sample_id"], sha256=photo["sha256"])
                    else:
                        candidate.update(
                            input_sample_ids=[by_role["input"][0]["sample_id"]],
                            seed=776 + (replicate - 1) * 1000,
                            sha256=sha256(candidate_id.encode()).hexdigest(),
                        )
                    result["candidates"].append(candidate)
                    candidate_mean = baseline - (
                        gap + (0.25 if expression == "smile" else -0.25)
                        if kind == "generated"
                        else 0
                    )
                    for group, rater in [("self", self_rater)] + [
                        ("unfamiliar", r) for r in strangers
                    ]:
                        assignment_id = candidate_id + "_" + rater
                        result["assignments"].append(
                            {
                                "assignment_id": assignment_id,
                                "candidate_id": candidate_id,
                                "candidate_sha256": candidate["sha256"],
                                "rater_id": rater,
                                "group": group,
                            }
                        )
                        if group == "self" and index == identity_count - 1:
                            continue
                        missing = rng.random() < 0.1
                        score = int(np.clip(np.rint(candidate_mean + rng.normal(0, 0.75)), 1, 7))
                        result["responses"].append(
                            {
                                "assignment_id": assignment_id,
                                "status": "unjudgeable" if missing else "rated",
                                "score": None if missing else score,
                            }
                        )
    return {
        "evidence_type": "SIMULATION_ONLY",
        "seed": seed,
        "warning": (
            "All identities, hashes and responses are invented solely to exercise the schema; "
            "no image assets exist."
        ),
        **result,
    }
