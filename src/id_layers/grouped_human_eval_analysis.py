from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from statistics import fmean
from typing import Any

ANSWER_KEY_SCHEMA_VERSION = "identity-layers-grouped-human-eval-key/v2"
RESPONSE_SCHEMA_VERSION = "identity-layers-grouped-human-response/v2"
ANALYSIS_SCHEMA_VERSION = "identity-layers-grouped-human-eval-analysis/v2"

EXPECTED_FORM_COUNT = 10
EXPECTED_TRIALS_PER_FORM = 48
EXPECTED_DISPLAY_TRIALS = 480
EXPECTED_RATINGS_PER_ITEM = 5
EXPECTED_UNIQUE_ITEMS = 96
EXPECTED_IDENTITY_COUNT = 8
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260903
CONFIDENCE_LEVEL = 0.95

# A focal win is always scored as one. The direction is frozen here so that a
# changed or accidentally reversed key fails before any result is calculated.
PAIR_CONTRASTS: dict[str, tuple[str, str]] = {
    "P0": ("k4_diverse_full", "k1_full"),
    "P1": ("k4_diverse_full", "k4_repeat_full"),
    "P3": ("global_target_patch_donor", "global_donor_patch_target"),
}

_CONDITION_ALIASES = {
    "k1": "k1_full",
    "k1_full": "k1_full",
    "k4_diverse": "k4_diverse_full",
    "k4_diverse_full": "k4_diverse_full",
    "k4_repeat": "k4_repeat_full",
    "k4_repeat_full": "k4_repeat_full",
    "global_target_patch_donor": "global_target_patch_donor",
    "global_donor_patch_target": "global_donor_patch_target",
}

_SPECIAL_SELECTIONS = {"tie", "unjudgeable", "cannot_judge"}


class GroupedHumanEvalAnalysisError(ValueError):
    """Raised when grouped-v2 inputs are incomplete, inconsistent, or ambiguous."""


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise GroupedHumanEvalAnalysisError(f"Duplicate JSON key {key!r} in {label} {path}")
            result[key] = item
        return result

    def reject_nonfinite(token: str) -> Any:
        raise GroupedHumanEvalAnalysisError(f"Non-finite JSON number {token!r} in {label} {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise GroupedHumanEvalAnalysisError(f"Invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise GroupedHumanEvalAnalysisError(f"Expected a JSON object in {label} {path}")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroupedHumanEvalAnalysisError(f"{context} must be a non-empty string")
    return value.strip()


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GroupedHumanEvalAnalysisError(f"{context} must be an integer")
    return value


def _normalize_condition(value: Any, context: str) -> str:
    condition = _text(value, context)
    try:
        return _CONDITION_ALIASES[condition]
    except KeyError as error:
        raise GroupedHumanEvalAnalysisError(
            f"Unsupported condition {condition!r} in {context}"
        ) from error


def _string_mapping(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) != 2:
        raise GroupedHumanEvalAnalysisError(f"{context} must be an object with exactly two entries")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, f"{context} key")
        result[key] = _text(raw_value, f"{context}[{key!r}]")
    if len(result) != 2:
        raise GroupedHumanEvalAnalysisError(f"{context} contains duplicate keys")
    return result


def _design_form_ids(value: Mapping[str, Any]) -> list[str] | None:
    design = value.get("design")
    if design is None:
        return None
    if not isinstance(design, dict):
        raise GroupedHumanEvalAnalysisError("answer-key design must be an object")
    raw = design.get("form_ids", design.get("forms"))
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise GroupedHumanEvalAnalysisError("answer-key design.form_ids/forms must be an array")
    form_ids = [_text(item, "answer-key design form ID") for item in raw]
    if len(form_ids) != len(set(form_ids)):
        raise GroupedHumanEvalAnalysisError("answer-key design contains duplicate form IDs")
    return form_ids


def _validate_trial(raw: Any, index: int) -> dict[str, Any]:
    context = f"answer-key trial {index}"
    if not isinstance(raw, dict):
        raise GroupedHumanEvalAnalysisError(f"{context} must be a JSON object")

    trial_id = _text(raw.get("trial_id"), f"{context} trial_id")
    pair_id = _text(raw.get("pair_id"), f"{context} pair_id")
    form_id = _text(raw.get("form_id"), f"{context} form_id")
    contrast_code = _text(
        raw.get("contrast_code", raw.get("block_code")),
        f"{context} contrast_code",
    )
    if contrast_code not in PAIR_CONTRASTS:
        raise GroupedHumanEvalAnalysisError(
            f"Unsupported contrast {contrast_code!r}; only P0, P1, and P3 are analyzed"
        )

    expected_focal, expected_comparator = PAIR_CONTRASTS[contrast_code]
    focal_condition = _normalize_condition(raw.get("focal_condition"), f"{context} focal_condition")
    comparator_condition = _normalize_condition(
        raw.get("comparator_condition"), f"{context} comparator_condition"
    )
    if (focal_condition, comparator_condition) != (
        expected_focal,
        expected_comparator,
    ):
        raise GroupedHumanEvalAnalysisError(
            f"{context} reverses or changes frozen contrast {contrast_code}: "
            f"observed {(focal_condition, comparator_condition)!r}, expected "
            f"{(expected_focal, expected_comparator)!r}"
        )

    option_roles = _string_mapping(raw.get("option_roles"), f"{context} option_roles")
    if set(option_roles.values()) != {"focal", "comparator"}:
        raise GroupedHumanEvalAnalysisError(
            f"{context} option_roles must map one option to focal and one to comparator"
        )
    for option_id in option_roles:
        if option_id.casefold() in _SPECIAL_SELECTIONS:
            raise GroupedHumanEvalAnalysisError(
                f"{context} uses reserved response token {option_id!r} as an option ID"
            )

    raw_conditions = _string_mapping(raw.get("option_conditions"), f"{context} option_conditions")
    if set(raw_conditions) != set(option_roles):
        raise GroupedHumanEvalAnalysisError(
            f"{context} option_roles and option_conditions have different option IDs"
        )
    option_conditions = {
        option_id: _normalize_condition(condition, f"{context} option_conditions[{option_id!r}]")
        for option_id, condition in raw_conditions.items()
    }
    expected_by_role = {
        "focal": focal_condition,
        "comparator": comparator_condition,
    }
    for option_id, role in option_roles.items():
        if option_conditions[option_id] != expected_by_role[role]:
            raise GroupedHumanEvalAnalysisError(
                f"{context} assigns option {option_id!r} inconsistent role and condition"
            )

    identity_id = _text(raw.get("identity_id"), f"{context} identity_id")
    prompt_id = _text(raw.get("prompt_id"), f"{context} prompt_id")
    base_seed = _integer(raw.get("base_seed", raw.get("generation_seed")), f"{context} base_seed")
    raw_effective_seed = raw.get("effective_seed")
    effective_seed = (
        None
        if raw_effective_seed is None
        else _integer(raw_effective_seed, f"{context} effective_seed")
    )

    return {
        "trial_id": trial_id,
        "pair_id": pair_id,
        "form_id": form_id,
        "contrast_code": contrast_code,
        "focal_condition": focal_condition,
        "comparator_condition": comparator_condition,
        "identity_id": identity_id,
        "prompt_id": prompt_id,
        "base_seed": base_seed,
        "effective_seed": effective_seed,
        "option_roles": option_roles,
        "option_conditions": option_conditions,
    }


def _same_item_factors(trial: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        trial["contrast_code"],
        trial["focal_condition"],
        trial["comparator_condition"],
        trial["identity_id"],
        trial["prompt_id"],
        trial["base_seed"],
        trial["effective_seed"],
    )


def _validate_factorial(items: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    identities = sorted({str(item["identity_id"]) for item in items})
    prompts = sorted({str(item["prompt_id"]) for item in items})
    base_seeds = sorted({int(item["base_seed"]) for item in items})
    if len(identities) != EXPECTED_IDENTITY_COUNT:
        raise GroupedHumanEvalAnalysisError(
            f"Expected {EXPECTED_IDENTITY_COUNT} identities, found {len(identities)}"
        )
    if len(prompts) != 2:
        raise GroupedHumanEvalAnalysisError(f"Expected two prompt IDs, found {prompts}")
    if len(base_seeds) != 2:
        raise GroupedHumanEvalAnalysisError(f"Expected two base seeds, found {base_seeds}")

    expected_cells = {
        (contrast, identity, prompt, base_seed)
        for contrast in PAIR_CONTRASTS
        for identity in identities
        for prompt in prompts
        for base_seed in base_seeds
    }
    observed_cells = [
        (
            str(item["contrast_code"]),
            str(item["identity_id"]),
            str(item["prompt_id"]),
            int(item["base_seed"]),
        )
        for item in items
    ]
    observed_set = set(observed_cells)
    if observed_set != expected_cells:
        missing = sorted(expected_cells - observed_set)
        extra = sorted(observed_set - expected_cells)
        raise GroupedHumanEvalAnalysisError(
            "Answer key does not contain the complete P0/P1/P3 factorial: "
            f"missing={missing}, extra={extra}"
        )
    if len(observed_cells) != len(observed_set):
        raise GroupedHumanEvalAnalysisError(
            "Multiple pair IDs encode the same contrast/identity/prompt/base-seed cell"
        )
    return {
        "identities": identities,
        "prompts": prompts,
        "base_seeds": base_seeds,
    }


def _validate_answer_key(
    value: Mapping[str, Any],
) -> tuple[
    str,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    dict[str, list[Any]],
]:
    if value.get("schema_version") != ANSWER_KEY_SCHEMA_VERSION:
        raise GroupedHumanEvalAnalysisError(
            f"Unsupported answer-key schema_version: {value.get('schema_version')!r}"
        )
    study_id = _text(value.get("study_id"), "answer-key study_id")
    raw_trials = value.get("trials")
    if not isinstance(raw_trials, list):
        raise GroupedHumanEvalAnalysisError("answer-key trials must be an array")
    if len(raw_trials) != EXPECTED_DISPLAY_TRIALS:
        raise GroupedHumanEvalAnalysisError(
            f"Expected {EXPECTED_DISPLAY_TRIALS} display trials, found {len(raw_trials)}"
        )

    trials: dict[str, dict[str, Any]] = {}
    form_trials: defaultdict[str, list[str]] = defaultdict(list)
    trials_by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw_trial in enumerate(raw_trials, start=1):
        trial = _validate_trial(raw_trial, index)
        trial_id = str(trial["trial_id"])
        if trial_id in trials:
            raise GroupedHumanEvalAnalysisError(f"Duplicate trial_id {trial_id!r}")
        trials[trial_id] = trial
        form_trials[str(trial["form_id"])].append(trial_id)
        trials_by_pair[str(trial["pair_id"])].append(trial)

    if len(form_trials) != EXPECTED_FORM_COUNT:
        raise GroupedHumanEvalAnalysisError(
            f"Expected {EXPECTED_FORM_COUNT} forms, found {len(form_trials)}"
        )
    for form_id, trial_ids in form_trials.items():
        if len(trial_ids) != EXPECTED_TRIALS_PER_FORM:
            raise GroupedHumanEvalAnalysisError(
                f"Form {form_id!r} has {len(trial_ids)} trials; expected {EXPECTED_TRIALS_PER_FORM}"
            )
        pair_ids = [str(trials[trial_id]["pair_id"]) for trial_id in trial_ids]
        if len(pair_ids) != len(set(pair_ids)):
            raise GroupedHumanEvalAnalysisError(f"Form {form_id!r} repeats a unique pair")

    declared_forms = _design_form_ids(value)
    if declared_forms is not None and set(declared_forms) != set(form_trials):
        raise GroupedHumanEvalAnalysisError(
            "answer-key design form IDs do not match trial form IDs"
        )

    if len(trials_by_pair) != EXPECTED_UNIQUE_ITEMS:
        raise GroupedHumanEvalAnalysisError(
            f"Expected {EXPECTED_UNIQUE_ITEMS} unique pairs, found {len(trials_by_pair)}"
        )

    items: dict[str, dict[str, Any]] = {}
    for pair_id, copies in trials_by_pair.items():
        if len(copies) != EXPECTED_RATINGS_PER_ITEM:
            raise GroupedHumanEvalAnalysisError(
                f"Pair {pair_id!r} has {len(copies)} display copies; "
                f"expected {EXPECTED_RATINGS_PER_ITEM}"
            )
        copy_forms = [str(copy["form_id"]) for copy in copies]
        if len(copy_forms) != len(set(copy_forms)):
            raise GroupedHumanEvalAnalysisError(f"Pair {pair_id!r} occurs more than once in a form")
        expected_factors = _same_item_factors(copies[0])
        if any(_same_item_factors(copy) != expected_factors for copy in copies[1:]):
            raise GroupedHumanEvalAnalysisError(
                f"Display copies of pair {pair_id!r} disagree on item factors"
            )
        representative = {
            key: value
            for key, value in copies[0].items()
            if key not in {"trial_id", "form_id", "option_roles", "option_conditions"}
        }
        representative["pair_id"] = pair_id
        representative["display_trial_ids"] = sorted(str(copy["trial_id"]) for copy in copies)
        representative["display_form_ids"] = sorted(copy_forms)
        items[pair_id] = representative

    raw_pairs = value.get("pairs")
    if raw_pairs is not None:
        if not isinstance(raw_pairs, list):
            raise GroupedHumanEvalAnalysisError("answer-key pairs must be an array")
        declared_pair_ids = [
            _text(pair.get("pair_id"), "answer-key pairs pair_id")
            for pair in raw_pairs
            if isinstance(pair, dict)
        ]
        if len(declared_pair_ids) != len(raw_pairs):
            raise GroupedHumanEvalAnalysisError("Every answer-key pairs entry must be an object")
        if len(declared_pair_ids) != len(set(declared_pair_ids)):
            raise GroupedHumanEvalAnalysisError("answer-key pairs contains duplicates")
        if set(declared_pair_ids) != set(items):
            raise GroupedHumanEvalAnalysisError(
                "answer-key pairs and trials contain different pair IDs"
            )

    factorial = _validate_factorial(list(items.values()))
    return study_id, trials, items, dict(form_trials), factorial


def _participant_id(value: Mapping[str, Any], context: str) -> str:
    participant_id = value.get("participant_id")
    participant_code = value.get("participant_code")
    if participant_id is not None and participant_code is not None:
        first = _text(participant_id, f"{context} participant_id")
        second = _text(participant_code, f"{context} participant_code")
        if first != second:
            raise GroupedHumanEvalAnalysisError(
                f"{context} participant_id and participant_code disagree"
            )
        result = first
    else:
        result = _text(
            participant_id if participant_id is not None else participant_code,
            f"{context} participant_id",
        )
    if len(result) > 80:
        raise GroupedHumanEvalAnalysisError(
            f"{context} participant identifier exceeds 80 characters"
        )
    return result


def _normalize_selection(value: Any, allowed_option_ids: set[str], context: str) -> str:
    selection = _text(value, context)
    if selection in allowed_option_ids:
        return selection
    special = selection.casefold()
    if special == "tie":
        return "tie"
    if special in {"unjudgeable", "cannot_judge"}:
        return "cannot_judge"
    raise GroupedHumanEvalAnalysisError(
        f"{context} is neither an opaque option ID nor tie/unjudgeable"
    )


def _validate_response(
    value: Mapping[str, Any],
    path: Path,
    study_id: str,
    trials: Mapping[str, Mapping[str, Any]],
    form_trials: Mapping[str, Sequence[str]],
) -> tuple[str, str, dict[str, str]]:
    context = f"response {path}"
    if value.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise GroupedHumanEvalAnalysisError(
            f"Unsupported response schema_version in {path}: {value.get('schema_version')!r}"
        )
    if _text(value.get("study_id"), f"{context} study_id") != study_id:
        raise GroupedHumanEvalAnalysisError(f"{context} belongs to another study")
    form_id = _text(value.get("form_id"), f"{context} form_id")
    if form_id not in form_trials:
        raise GroupedHumanEvalAnalysisError(f"{context} names unknown form {form_id!r}")
    participant_id = _participant_id(value, context)
    if value.get("completed") is not True:
        raise GroupedHumanEvalAnalysisError(f"{context} is not marked completed=true")

    expected_trial_ids = set(form_trials[form_id])
    trial_count = _integer(value.get("trial_count"), f"{context} trial_count")
    if trial_count != len(expected_trial_ids):
        raise GroupedHumanEvalAnalysisError(
            f"{context} trial_count is {trial_count}; expected {len(expected_trial_ids)}"
        )
    raw_responses = value.get("responses")
    if not isinstance(raw_responses, list):
        raise GroupedHumanEvalAnalysisError(f"{context} responses must be an array")
    if len(raw_responses) != len(expected_trial_ids):
        raise GroupedHumanEvalAnalysisError(
            f"{context} has {len(raw_responses)} response rows; expected {len(expected_trial_ids)}"
        )

    selections: dict[str, str] = {}
    for index, raw_response in enumerate(raw_responses, start=1):
        row_context = f"{context} row {index}"
        if not isinstance(raw_response, dict):
            raise GroupedHumanEvalAnalysisError(f"{row_context} must be an object")
        trial_id = _text(raw_response.get("trial_id"), f"{row_context} trial_id")
        if trial_id not in expected_trial_ids:
            raise GroupedHumanEvalAnalysisError(
                f"{row_context} names a trial outside form {form_id!r}"
            )
        if trial_id in selections:
            raise GroupedHumanEvalAnalysisError(f"{context} repeats trial_id {trial_id!r}")
        allowed = set(trials[trial_id]["option_roles"])
        selections[trial_id] = _normalize_selection(
            raw_response.get("selection"), allowed, f"{row_context} selection"
        )
    if set(selections) != expected_trial_ids:
        missing = sorted(expected_trial_ids - set(selections))
        raise GroupedHumanEvalAnalysisError(f"{context} is incomplete; missing trial IDs {missing}")
    return participant_id, form_id, selections


def _rate(selection: str, trial: Mapping[str, Any]) -> str:
    if selection == "tie":
        return "tie"
    if selection == "cannot_judge":
        return "cannot_judge"
    role = trial["option_roles"].get(selection)
    if role == "focal":
        return "focal_win"
    if role == "comparator":
        return "comparator_win"
    raise AssertionError("Validated selection is absent from option_roles")


def _preference_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    focal = int(counts.get("focal_win", 0))
    tie = int(counts.get("tie", 0))
    comparator = int(counts.get("comparator_win", 0))
    cannot_judge = int(counts.get("cannot_judge", 0))
    judged = focal + tie + comparator
    total = judged + cannot_judge
    return {
        "total_ratings": total,
        "judged_ratings": judged,
        "cannot_judge_count": cannot_judge,
        "cannot_judge_proportion": cannot_judge / total if total else None,
        "focal_win_count": focal,
        "tie_count": tie,
        "comparator_win_count": comparator,
        "focal_win_proportion_judged": focal / judged if judged else None,
        "tie_proportion_judged": tie / judged if judged else None,
        "comparator_win_proportion_judged": comparator / judged if judged else None,
        "focal_preference_score": (focal + 0.5 * tie) / judged if judged else None,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise GroupedHumanEvalAnalysisError("Cannot compute a quantile of no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stream_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _identity_cluster_bootstrap(
    identity_scores: Sequence[float | None], label: str
) -> dict[str, Any]:
    scores = [
        float(score)
        for score in identity_scores
        if score is not None and math.isfinite(float(score))
    ]
    if not scores:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "identity_count_with_score": 0,
        }
    randomizer = random.Random(_stream_seed(label))
    count = len(scores)
    replicates = [
        fmean(scores[randomizer.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "estimate": fmean(scores),
        "ci_low": _quantile(replicates, alpha),
        "ci_high": _quantile(replicates, 1.0 - alpha),
        "identity_count_with_score": count,
    }


def analyze_grouped_human_eval(
    answer_key_path: Path | str,
    response_paths: Sequence[Path | str],
) -> dict[str, Any]:
    """Validate and analyze one completed response for each of the ten v2 forms."""
    key_path = Path(answer_key_path)
    study_id, trials, items, form_trials, factorial = _validate_answer_key(
        _read_json_object(key_path, "answer key")
    )

    paths = [Path(path) for path in response_paths]
    if len(paths) != EXPECTED_FORM_COUNT:
        raise GroupedHumanEvalAnalysisError(
            f"Exactly {EXPECTED_FORM_COUNT} response JSON files are required; received {len(paths)}"
        )
    resolved_paths = [path.resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise GroupedHumanEvalAnalysisError("The response path list contains duplicates")

    participant_keys: dict[str, Path] = {}
    submitted_forms: dict[str, Path] = {}
    ratings_by_trial: dict[str, str] = {}
    participants: list[str] = []
    for path in paths:
        participant_id, form_id, selections = _validate_response(
            _read_json_object(path, "response"),
            path,
            study_id,
            trials,
            form_trials,
        )
        participant_key = participant_id.casefold()
        if participant_key in participant_keys:
            raise GroupedHumanEvalAnalysisError(
                f"Participant identifier {participant_id!r} is not unique across "
                f"{participant_keys[participant_key]} and {path}"
            )
        if form_id in submitted_forms:
            raise GroupedHumanEvalAnalysisError(
                f"Form {form_id!r} has more than one response: "
                f"{submitted_forms[form_id]} and {path}"
            )
        participant_keys[participant_key] = path
        submitted_forms[form_id] = path
        participants.append(participant_id)
        ratings_by_trial.update(selections)

    missing_forms = sorted(set(form_trials) - set(submitted_forms))
    if missing_forms:
        raise GroupedHumanEvalAnalysisError(
            f"No completed response was supplied for forms {missing_forms}"
        )
    if set(ratings_by_trial) != set(trials):
        raise GroupedHumanEvalAnalysisError(
            "The ten form responses do not cover every display trial exactly once"
        )

    counts_by_pair: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for trial_id, selection in ratings_by_trial.items():
        trial = trials[trial_id]
        counts_by_pair[str(trial["pair_id"])][_rate(selection, trial)] += 1
    for pair_id in items:
        total = sum(counts_by_pair[pair_id].values())
        if total != EXPECTED_RATINGS_PER_ITEM:
            raise GroupedHumanEvalAnalysisError(
                f"Pair {pair_id!r} has {total} votes; expected {EXPECTED_RATINGS_PER_ITEM}"
            )

    item_rows: list[dict[str, Any]] = []
    item_results: defaultdict[tuple[str, str], list[tuple[Counter[str], float | None]]] = (
        defaultdict(list)
    )
    ordered_items = sorted(
        items.values(),
        key=lambda item: (
            str(item["contrast_code"]),
            str(item["identity_id"]),
            str(item["prompt_id"]),
            int(item["base_seed"]),
        ),
    )
    for item in ordered_items:
        pair_id = str(item["pair_id"])
        counts = counts_by_pair[pair_id]
        summary = _preference_summary(counts)
        row = {
            "pair_id": pair_id,
            "contrast_code": item["contrast_code"],
            "focal_condition": item["focal_condition"],
            "comparator_condition": item["comparator_condition"],
            "identity_id": item["identity_id"],
            "prompt_id": item["prompt_id"],
            "base_seed": item["base_seed"],
            "effective_seed": item["effective_seed"],
            "display_form_ids": ";".join(item["display_form_ids"]),
            "display_trial_ids": ";".join(item["display_trial_ids"]),
            **summary,
        }
        item_rows.append(row)
        item_results[(str(item["contrast_code"]), str(item["identity_id"]))].append(
            (counts, summary["focal_preference_score"])
        )

    identity_rows: list[dict[str, Any]] = []
    identity_scores_by_contrast: defaultdict[str, list[float | None]] = defaultdict(list)
    for contrast_code in PAIR_CONTRASTS:
        focal_condition, comparator_condition = PAIR_CONTRASTS[contrast_code]
        for identity_id in factorial["identities"]:
            results = item_results[(contrast_code, str(identity_id))]
            combined: Counter[str] = Counter()
            item_scores: list[float] = []
            for counts, score in results:
                combined.update(counts)
                if score is not None:
                    item_scores.append(float(score))
            pooled = _preference_summary(combined)
            identity_score = fmean(item_scores) if item_scores else None
            identity_scores_by_contrast[contrast_code].append(identity_score)
            identity_rows.append(
                {
                    "contrast_code": contrast_code,
                    "focal_condition": focal_condition,
                    "comparator_condition": comparator_condition,
                    "identity_id": identity_id,
                    "unique_item_count": len(results),
                    "scored_item_count": len(item_scores),
                    "focal_preference_score": identity_score,
                    "pooled_vote_focal_preference_score": pooled["focal_preference_score"],
                    **{
                        key: value
                        for key, value in pooled.items()
                        if key != "focal_preference_score"
                    },
                }
            )

    contrast_rows: list[dict[str, Any]] = []
    for contrast_code, (focal_condition, comparator_condition) in PAIR_CONTRASTS.items():
        relevant_items = [row for row in item_rows if row["contrast_code"] == contrast_code]
        combined = Counter(
            {
                "focal_win": sum(row["focal_win_count"] for row in relevant_items),
                "tie": sum(row["tie_count"] for row in relevant_items),
                "comparator_win": sum(row["comparator_win_count"] for row in relevant_items),
                "cannot_judge": sum(row["cannot_judge_count"] for row in relevant_items),
            }
        )
        pooled = _preference_summary(combined)
        bootstrap = _identity_cluster_bootstrap(
            identity_scores_by_contrast[contrast_code], contrast_code
        )
        contrast_rows.append(
            {
                "contrast_code": contrast_code,
                "focal_condition": focal_condition,
                "comparator_condition": comparator_condition,
                "unique_item_count": len(relevant_items),
                "identity_count": EXPECTED_IDENTITY_COUNT,
                "focal_preference_proportion": bootstrap["estimate"],
                "focal_preference_ci_low": bootstrap["ci_low"],
                "focal_preference_ci_high": bootstrap["ci_high"],
                "identity_count_with_score": bootstrap["identity_count_with_score"],
                "pooled_vote_focal_preference_score": pooled["focal_preference_score"],
                **{key: value for key, value in pooled.items() if key != "focal_preference_score"},
            }
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_id": study_id,
        "analysis_scope": {
            "task": "pairwise_identity_likeness",
            "included_contrast_codes": list(PAIR_CONTRASTS),
            "score_mapping": {
                "focal_win": 1.0,
                "tie": 0.5,
                "comparator_win": 0.0,
            },
            "cannot_judge_policy": (
                "Excluded from preference-score denominators and retained in counts/rates"
            ),
            "aggregation": (
                "display votes -> unique pair_id -> equal-weight item mean within "
                "identity -> equal-weight identity mean within contrast"
            ),
            "exploratory_only": True,
            "exploratory_reason": (
                "The identity-cluster interval has only eight identity clusters"
            ),
        },
        "inputs": {
            "answer_key": str(key_path),
            "responses": [str(path) for path in paths],
        },
        "validation": {
            "form_count": len(form_trials),
            "response_file_count": len(paths),
            "completed_response_per_form": True,
            "participant_count": len(participants),
            "participant_ids_unique": True,
            "display_trial_count": len(trials),
            "ratings_per_display_trial_min": 1,
            "ratings_per_display_trial_max": 1,
            "unique_item_count": len(items),
            "ratings_per_item_min": min(sum(counts.values()) for counts in counts_by_pair.values()),
            "ratings_per_item_max": max(sum(counts.values()) for counts in counts_by_pair.values()),
            "trials_per_form": {
                form_id: len(trial_ids) for form_id, trial_ids in sorted(form_trials.items())
            },
            "identity_count": len(factorial["identities"]),
            "prompt_ids": factorial["prompts"],
            "base_seeds": factorial["base_seeds"],
        },
        "contrasts": contrast_rows,
        "item_table": item_rows,
        "identity_table": identity_rows,
        "bootstrap": {
            "unit": "identity",
            "expected_identity_clusters": EXPECTED_IDENTITY_COUNT,
            "resamples": BOOTSTRAP_RESAMPLES,
            "base_seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "interval": "percentile",
            "exploratory_only": True,
        },
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_grouped_human_eval_analysis(
    output_path: Path | str, report: Mapping[str, Any]
) -> dict[str, Path]:
    """Write the full JSON report plus flat item- and identity-level CSV tables."""
    path = Path(output_path)
    item_path = path.with_name(f"{path.stem}_items.csv")
    identity_path = path.with_name(f"{path.stem}_identities.csv")
    _atomic_write_text(
        path,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write_text(item_path, _csv_text(report["item_table"]))
    _atomic_write_text(identity_path, _csv_text(report["identity_table"]))
    return {
        "analysis": path,
        "item_table": item_path,
        "identity_table": identity_path,
    }


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "ANSWER_KEY_SCHEMA_VERSION",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "GroupedHumanEvalAnalysisError",
    "PAIR_CONTRASTS",
    "RESPONSE_SCHEMA_VERSION",
    "analyze_grouped_human_eval",
    "write_grouped_human_eval_analysis",
]
