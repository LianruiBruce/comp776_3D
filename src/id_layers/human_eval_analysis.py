from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

ANSWER_KEY_SCHEMA_VERSION = "identity-layers-human-eval/v1"
RESPONSE_SCHEMA_VERSION = "identity-layers-human-response/v1"
ANALYSIS_SCHEMA_VERSION = "identity-layers-human-eval-analysis/v1"

# The first role is the focal condition: selecting it is a "win" in the decoded report.
PAIR_CONTRASTS: dict[str, tuple[str, str]] = {
    "P0": ("k1", "k4_diverse"),
    "P1": ("k4_diverse", "k4_repeat"),
    "P2": ("k4_repeat", "k1"),
    "P3": ("global_target_patch_donor", "global_donor_patch_target"),
    "P4": ("k1", "text_only"),
}


class HumanEvalAnalysisError(ValueError):
    """Raised when human-evaluation inputs cannot be analyzed unambiguously."""


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise HumanEvalAnalysisError(
                    f"Duplicate JSON key {key!r} in {label} {path}"
                )
            result[key] = item
        return result

    def reject_nonfinite(token: str) -> Any:
        raise HumanEvalAnalysisError(
            f"Non-finite JSON number {token!r} in {label} {path}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise HumanEvalAnalysisError(f"Invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise HumanEvalAnalysisError(f"Expected a JSON object in {label} {path}")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanEvalAnalysisError(f"{context} must be a non-empty string")
    return value.strip()


def _string_mapping(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise HumanEvalAnalysisError(f"{context} must be a non-empty JSON object")
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _text(key, f"{context} key")
        result[normalized_key] = _text(item, f"{context}[{normalized_key!r}]")
    if len(result) != len(value):
        raise HumanEvalAnalysisError(f"{context} contains duplicate normalized keys")
    return result


def _validate_answer_key(value: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if value.get("schema_version") != ANSWER_KEY_SCHEMA_VERSION:
        raise HumanEvalAnalysisError(
            f"Unsupported answer-key schema_version: {value.get('schema_version')!r}"
        )
    study_id = _text(value.get("study_id"), "answer-key study_id")
    raw_trials = value.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise HumanEvalAnalysisError("Answer key must contain a non-empty trials array")

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    kinds: set[str] = set()
    for index, raw in enumerate(raw_trials, start=1):
        context = f"answer-key trial {index}"
        if not isinstance(raw, dict):
            raise HumanEvalAnalysisError(f"{context} must be a JSON object")
        trial_id = _text(raw.get("trial_id"), f"{context} trial_id")
        if trial_id in seen:
            raise HumanEvalAnalysisError(f"Duplicate answer-key trial_id: {trial_id!r}")
        seen.add(trial_id)
        kind = _text(raw.get("kind"), f"{context} kind")
        block_code = _text(raw.get("block_code"), f"{context} block_code")
        method = _text(raw.get("method"), f"{context} method")
        option_roles = _string_mapping(raw.get("option_roles"), f"{context} option_roles")

        trial = dict(raw)
        trial.update(
            {
                "trial_id": trial_id,
                "kind": kind,
                "block_code": block_code,
                "method": method,
                "option_roles": option_roles,
            }
        )
        if kind == "pairwise":
            if block_code not in PAIR_CONTRASTS:
                raise HumanEvalAnalysisError(
                    f"Unknown pairwise block_code {block_code!r} in {context}"
                )
            expected_roles = set(PAIR_CONTRASTS[block_code])
            if len(option_roles) != 2 or set(option_roles.values()) != expected_roles:
                raise HumanEvalAnalysisError(
                    f"{context} roles do not match frozen contrast {block_code}: "
                    f"expected {sorted(expected_roles)}"
                )
            trial["identity_id"] = _text(raw.get("identity_id"), f"{context} identity_id")
        elif kind == "four_afc":
            if block_code != "I1":
                raise HumanEvalAnalysisError(
                    f"Identity 4-AFC trial {trial_id!r} must use block_code 'I1'"
                )
            required_roles = {"target_identity", "donor_identity", "foil_1", "foil_2"}
            if len(option_roles) != 4 or set(option_roles.values()) != required_roles:
                raise HumanEvalAnalysisError(
                    f"{context} must have exactly target, donor, and two foil option roles"
                )
            target_identity = _text(
                raw.get("target_identity_id"), f"{context} target_identity_id"
            )
            option_identities = _string_mapping(
                raw.get("option_identity_ids"), f"{context} option_identity_ids"
            )
            if set(option_identities) != set(option_roles):
                raise HumanEvalAnalysisError(
                    f"{context} option_identity_ids keys must equal option_roles keys"
                )
            if len(set(option_identities.values())) != 4:
                raise HumanEvalAnalysisError(
                    f"{context} must map its four options to four distinct identities"
                )
            target_options = [
                option_id
                for option_id, role in option_roles.items()
                if role == "target_identity"
            ]
            if option_identities[target_options[0]] != target_identity:
                raise HumanEvalAnalysisError(
                    f"{context} target option does not map to target_identity_id"
                )
            donor_identity = _text(
                raw.get("donor_identity_id"), f"{context} donor_identity_id"
            )
            donor_option = next(
                option_id
                for option_id, role in option_roles.items()
                if role == "donor_identity"
            )
            if option_identities[donor_option] != donor_identity:
                raise HumanEvalAnalysisError(
                    f"{context} donor option does not map to donor_identity_id"
                )
            trial["target_identity_id"] = target_identity
            trial["donor_identity_id"] = donor_identity
            trial["option_identity_ids"] = option_identities
            trial["variant"] = _text(raw.get("variant"), f"{context} variant")
        else:
            raise HumanEvalAnalysisError(f"Unsupported trial kind {kind!r} in {context}")
        kinds.add(kind)
        trials.append(trial)

    if kinds != {"pairwise", "four_afc"}:
        raise HumanEvalAnalysisError(
            "Answer key must contain both pairwise and identity 4-AFC trials"
        )
    return study_id, trials


def _validate_response(
    value: Mapping[str, Any],
    path: Path,
    *,
    study_id: str,
    trials_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, str]]:
    if value.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise HumanEvalAnalysisError(
            f"Unsupported response schema_version in {path}: {value.get('schema_version')!r}"
        )
    observed_study = _text(value.get("study_id"), f"response study_id in {path}")
    if observed_study != study_id:
        raise HumanEvalAnalysisError(
            f"study_id mismatch in {path}: {observed_study!r} != {study_id!r}"
        )
    participant_id = _text(value.get("participant_id"), f"participant_id in {path}")
    if len(participant_id) > 80:
        raise HumanEvalAnalysisError(f"participant_id in {path} exceeds 80 characters")
    if value.get("completed") is not True:
        raise HumanEvalAnalysisError(f"Response {path} is not marked completed")
    trial_count = value.get("trial_count")
    if isinstance(trial_count, bool) or not isinstance(trial_count, int):
        raise HumanEvalAnalysisError(f"trial_count in {path} must be an integer")
    if trial_count != len(trials_by_id):
        raise HumanEvalAnalysisError(
            f"trial_count mismatch in {path}: {trial_count} != {len(trials_by_id)}"
        )
    raw_responses = value.get("responses")
    if not isinstance(raw_responses, list):
        raise HumanEvalAnalysisError(f"responses in {path} must be an array")

    selections: dict[str, str] = {}
    for index, raw in enumerate(raw_responses, start=1):
        context = f"response {index} in {path}"
        if not isinstance(raw, dict):
            raise HumanEvalAnalysisError(f"{context} must be a JSON object")
        trial_id = _text(raw.get("trial_id"), f"{context} trial_id")
        selection = _text(raw.get("selection"), f"{context} selection")
        if trial_id in selections:
            raise HumanEvalAnalysisError(f"Duplicate response for trial {trial_id!r} in {path}")
        if trial_id not in trials_by_id:
            raise HumanEvalAnalysisError(f"Unknown trial_id {trial_id!r} in {path}")
        trial = trials_by_id[trial_id]
        specials = {"tie", "unjudgeable"} if trial["kind"] == "pairwise" else {
            "none",
            "unjudgeable",
        }
        allowed = set(trial["option_roles"]) | specials
        if selection not in allowed:
            raise HumanEvalAnalysisError(
                f"Invalid selection {selection!r} for trial {trial_id!r} in {path}"
            )
        selections[trial_id] = selection

    expected_ids = set(trials_by_id)
    observed_ids = set(selections)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise HumanEvalAnalysisError(
            f"Incomplete trial coverage in {path}: missing={missing}, extra={extra}"
        )
    return participant_id, selections


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stream_seed(seed: int, label: str) -> int:
    payload = json.dumps([seed, label], ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _cluster_bootstrap(
    values_by_identity: Mapping[str, float | None],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
    label: str,
) -> dict[str, Any]:
    finite = {
        identity: float(value)
        for identity, value in values_by_identity.items()
        if value is not None and math.isfinite(float(value))
    }
    identities = sorted(finite)
    if not identities:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "cluster_count": 0,
        }
    estimate = fmean(finite.values())
    rng = random.Random(_stream_seed(seed, label))
    replicates = [
        fmean(finite[rng.choice(identities)] for _ in identities)
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": estimate,
        "ci_low": _quantile(replicates, tail),
        "ci_high": _quantile(replicates, 1.0 - tail),
        "cluster_count": len(identities),
    }


def _identity_summary(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    four_way = counts["correct"] + counts["incorrect_option"]
    return {
        "rating_count": total,
        "four_way_response_count": four_way,
        "correct_count": counts["correct"],
        "incorrect_option_count": counts["incorrect_option"],
        "none_count": counts["none"],
        "unjudgeable_count": counts["unjudgeable"],
        "accuracy": _rate(counts["correct"], four_way),
        "accuracy_all_ratings": _rate(counts["correct"], total),
        "none_rate": _rate(counts["none"], total),
        "unjudgeable_rate": _rate(counts["unjudgeable"], total),
    }


def _pair_summary(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    judged = counts["win"] + counts["tie"] + counts["loss"]
    return {
        "rating_count": total,
        "judged_count": judged,
        "win_count": counts["win"],
        "tie_count": counts["tie"],
        "loss_count": counts["loss"],
        "unjudgeable_count": counts["unjudgeable"],
        "win_rate": _rate(counts["win"], judged),
        "tie_rate": _rate(counts["tie"], judged),
        "loss_rate": _rate(counts["loss"], judged),
        "unjudgeable_rate": _rate(counts["unjudgeable"], total),
        "preference_score": _rate(counts["win"] + 0.5 * counts["tie"], judged),
    }


def analyze_human_eval(
    answer_key_path: str | Path,
    response_paths: str | Path | Sequence[str | Path],
    *,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 20260903,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Validate and analyze blinded human responses using only the private answer key."""
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise HumanEvalAnalysisError("bootstrap_resamples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise HumanEvalAnalysisError("bootstrap_seed must be an integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float)):
        raise HumanEvalAnalysisError("confidence_level must be numeric")
    if not 0.0 < confidence_level < 1.0:
        raise HumanEvalAnalysisError("confidence_level must be strictly between 0 and 1")

    key_path = Path(answer_key_path).resolve()
    key_value = _read_json_object(key_path, "answer key")
    study_id, trials = _validate_answer_key(key_value)
    trials_by_id = {trial["trial_id"]: trial for trial in trials}

    raw_paths = (
        [response_paths]
        if isinstance(response_paths, (str, Path))
        else list(response_paths)
    )
    if not raw_paths:
        raise HumanEvalAnalysisError("At least one response JSON is required")
    paths = [Path(path).resolve() for path in raw_paths]
    if len(paths) != len(set(paths)):
        raise HumanEvalAnalysisError("The same response file was supplied more than once")

    rater_selections: list[tuple[str, dict[str, str]]] = []
    participant_sources: dict[str, Path] = {}
    for path in paths:
        value = _read_json_object(path, "response")
        participant_id, selections = _validate_response(
            value,
            path,
            study_id=study_id,
            trials_by_id=trials_by_id,
        )
        normalized_participant = participant_id.casefold()
        if normalized_participant in participant_sources:
            raise HumanEvalAnalysisError(
                "Duplicate participant_id "
                f"{participant_id!r} in {participant_sources[normalized_participant]} and {path}"
            )
        participant_sources[normalized_participant] = path
        rater_selections.append((participant_id, selections))

    ratings_by_trial: dict[str, list[str]] = {trial_id: [] for trial_id in trials_by_id}
    for _, selections in rater_selections:
        for trial_id, selection in selections.items():
            ratings_by_trial[trial_id].append(selection)

    identity_counts: Counter[str] = Counter()
    identity_cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pair_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pair_cluster_counts: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    item_rows: list[dict[str, Any]] = []

    for trial in trials:
        trial_id = trial["trial_id"]
        item_counts: Counter[str] = Counter()
        if trial["kind"] == "four_afc":
            identity = trial["target_identity_id"]
            for selection in ratings_by_trial[trial_id]:
                if selection == "none":
                    outcome = "none"
                elif selection == "unjudgeable":
                    outcome = "unjudgeable"
                elif trial["option_roles"][selection] == "target_identity":
                    outcome = "correct"
                else:
                    outcome = "incorrect_option"
                identity_counts[outcome] += 1
                identity_cluster_counts[identity][outcome] += 1
                item_counts[outcome] += 1
            item_rows.append(
                {
                    "trial_id": trial_id,
                    "kind": "four_afc",
                    "block_code": trial["block_code"],
                    "method": trial["method"],
                    "variant": trial["variant"],
                    "target_identity_id": identity,
                    "rating_count": sum(item_counts.values()),
                    "response_counts": dict(sorted(item_counts.items())),
                }
            )
        else:
            identity = trial["identity_id"]
            group = (trial["block_code"], trial["method"])
            focal, comparator = PAIR_CONTRASTS[trial["block_code"]]
            for selection in ratings_by_trial[trial_id]:
                if selection == "tie":
                    outcome = "tie"
                elif selection == "unjudgeable":
                    outcome = "unjudgeable"
                elif trial["option_roles"][selection] == focal:
                    outcome = "win"
                elif trial["option_roles"][selection] == comparator:
                    outcome = "loss"
                else:  # pragma: no cover - answer-key validation makes this unreachable
                    raise AssertionError("Unexpected decoded pairwise role")
                pair_counts[group][outcome] += 1
                pair_cluster_counts[group][identity][outcome] += 1
                item_counts[outcome] += 1
            item_rows.append(
                {
                    "trial_id": trial_id,
                    "kind": "pairwise",
                    "block_code": trial["block_code"],
                    "method": trial["method"],
                    "target_identity_id": identity,
                    "rating_count": sum(item_counts.values()),
                    "response_counts": dict(sorted(item_counts.items())),
                }
            )

    identity_report = {
        "item_count": sum(trial["kind"] == "four_afc" for trial in trials),
        **_identity_summary(identity_counts),
    }
    paired_reports: list[dict[str, Any]] = []
    for group in sorted(pair_counts):
        block_code, method = group
        focal, comparator = PAIR_CONTRASTS[block_code]
        paired_reports.append(
            {
                "block_code": block_code,
                "method": method,
                "focal_condition": focal,
                "comparison_condition": comparator,
                "item_count": sum(
                    trial["kind"] == "pairwise"
                    and trial["block_code"] == block_code
                    and trial["method"] == method
                    for trial in trials
                ),
                **_pair_summary(pair_counts[group]),
            }
        )

    identity_metric_values: dict[str, dict[str, float | None]] = {
        "accuracy": {},
        "none_rate": {},
        "unjudgeable_rate": {},
    }
    for identity, counts in identity_cluster_counts.items():
        summary = _identity_summary(counts)
        for metric in identity_metric_values:
            identity_metric_values[metric][identity] = summary[metric]
    identity_bootstrap = {
        metric: _cluster_bootstrap(
            values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
            label=f"identity-four-afc:{metric}",
        )
        for metric, values in identity_metric_values.items()
    }

    paired_bootstrap: list[dict[str, Any]] = []
    pair_metrics = ("win_rate", "tie_rate", "loss_rate", "unjudgeable_rate", "preference_score")
    for group in sorted(pair_cluster_counts):
        block_code, method = group
        by_metric: dict[str, dict[str, float | None]] = {
            metric: {} for metric in pair_metrics
        }
        for identity, counts in pair_cluster_counts[group].items():
            summary = _pair_summary(counts)
            for metric in pair_metrics:
                by_metric[metric][identity] = summary[metric]
        paired_bootstrap.append(
            {
                "block_code": block_code,
                "method": method,
                "metrics": {
                    metric: _cluster_bootstrap(
                        values,
                        resamples=bootstrap_resamples,
                        seed=bootstrap_seed,
                        confidence_level=confidence_level,
                        label=f"pair:{block_code}:{method}:{metric}",
                    )
                    for metric, values in by_metric.items()
                },
            }
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_id": study_id,
        "inputs": {
            "answer_key": str(key_path),
            "response_files": [str(path) for path in paths],
            "automatic_metrics_read": False,
        },
        "validation": {
            "rater_count": len(rater_selections),
            "participant_ids": sorted(participant for participant, _ in rater_selections),
            "trial_count": len(trials),
            "complete_trial_coverage_required": True,
            "ratings_per_item_min": min(len(values) for values in ratings_by_trial.values()),
            "ratings_per_item_max": max(len(values) for values in ratings_by_trial.values()),
        },
        "identity_four_afc": identity_report,
        "paired_contrasts": paired_reports,
        "per_item_rating_counts": sorted(item_rows, key=lambda row: row["trial_id"]),
        "identity_cluster_bootstrap": {
            "unit": "target_identity_id",
            "aggregation": "metric_within_identity_then_equal_weight_mean_across_identities",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "interval": "percentile",
            "identity_four_afc": identity_bootstrap,
            "paired_contrasts": paired_bootstrap,
        },
    }


def write_human_eval_analysis(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write an analysis JSON atomically."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "HumanEvalAnalysisError",
    "PAIR_CONTRASTS",
    "analyze_human_eval",
    "write_human_eval_analysis",
]
