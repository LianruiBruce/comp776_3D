from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from id_layers.human_eval_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    HumanEvalAnalysisError,
    analyze_human_eval,
    write_human_eval_analysis,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _four_afc_trial(trial_id: str, identity: str, suffix: str) -> dict[str, object]:
    option_roles = {
        f"{suffix}_target": "target_identity",
        f"{suffix}_donor": "donor_identity",
        f"{suffix}_foil1": "foil_1",
        f"{suffix}_foil2": "foil_2",
    }
    return {
        "trial_id": trial_id,
        "block_code": "I1",
        "kind": "four_afc",
        "method": "secret_method",
        "variant": "k4_diverse",
        "prompt_id": "prompt",
        "generation_seed": 7,
        "source_result_id": f"result-{suffix}",
        "target_identity_id": identity,
        "donor_identity_id": f"donor-{suffix}",
        "global_identity_id": identity,
        "patch_identity_id": identity,
        "option_roles": option_roles,
        "option_identity_ids": {
            f"{suffix}_target": identity,
            f"{suffix}_donor": f"donor-{suffix}",
            f"{suffix}_foil1": f"foil1-{suffix}",
            f"{suffix}_foil2": f"foil2-{suffix}",
        },
    }


def _pair_trial(
    trial_id: str,
    identity: str,
    block_code: str,
    focal: str,
    comparator: str,
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "block_code": block_code,
        "kind": "pairwise",
        "method": "secret_method",
        "identity_id": identity,
        "prompt_id": "prompt",
        "generation_seed": 7,
        "option_roles": {
            f"{trial_id}_left": comparator,
            f"{trial_id}_right": focal,
        },
        "source_result_ids": {
            f"{trial_id}_left": f"result-{comparator}",
            f"{trial_id}_right": f"result-{focal}",
        },
    }


def _bundle() -> tuple[dict[str, object], list[dict[str, object]]]:
    trials = [
        _four_afc_trial("identity-a", "id-a", "a"),
        _four_afc_trial("identity-b", "id-b", "b"),
        _pair_trial("p1-a", "id-a", "P1", "k4_diverse", "k4_repeat"),
        _pair_trial("p1-b", "id-b", "P1", "k4_diverse", "k4_repeat"),
        _pair_trial("p4-a", "id-a", "P4", "k1", "text_only"),
        _pair_trial("p4-b", "id-b", "P4", "k1", "text_only"),
    ]
    key: dict[str, object] = {
        "schema_version": "identity-layers-human-eval/v1",
        "study_id": "study-test",
        "randomization_seed": 776,
        "trials": trials,
    }
    selections = [
        {
            "identity-a": "a_target",
            "identity-b": "b_target",
            "p1-a": "p1-a_right",
            "p1-b": "p1-b_right",
            "p4-a": "p4-a_right",
            "p4-b": "tie",
        },
        {
            "identity-a": "a_donor",
            "identity-b": "b_target",
            "p1-a": "tie",
            "p1-b": "p1-b_right",
            "p4-a": "p4-a_right",
            "p4-b": "p4-b_left",
        },
        {
            "identity-a": "none",
            "identity-b": "unjudgeable",
            "p1-a": "p1-a_left",
            "p1-b": "unjudgeable",
            "p4-a": "p4-a_left",
            "p4-b": "unjudgeable",
        },
    ]
    responses: list[dict[str, object]] = []
    for index, participant_selections in enumerate(selections, start=1):
        response_rows = [
            {"trial_id": trial_id, "selection": selection, "updated_at": "ignored"}
            for trial_id, selection in reversed(list(participant_selections.items()))
        ]
        responses.append(
            {
                "schema_version": "identity-layers-human-response/v1",
                "study_id": "study-test",
                "participant_id": f"rater-{index}",
                "exported_at": "ignored",
                "completed": True,
                "trial_count": len(trials),
                "responses": response_rows,
            }
        )
    return key, responses


def _write_bundle(
    tmp_path: Path,
    key: dict[str, object],
    responses: list[dict[str, object]],
) -> tuple[Path, list[Path]]:
    key_path = _write_json(tmp_path / "private-key.json", key)
    response_paths = [
        _write_json(tmp_path / f"response-{index}.json", response)
        for index, response in enumerate(responses, start=1)
    ]
    return key_path, response_paths


def test_analyzes_decoded_counts_and_cluster_bootstrap(tmp_path: Path) -> None:
    key, responses = _bundle()
    key_path, response_paths = _write_bundle(tmp_path, key, responses)

    report = analyze_human_eval(
        key_path,
        response_paths,
        bootstrap_resamples=200,
        bootstrap_seed=19,
    )

    assert report["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert report["study_id"] == "study-test"
    assert report["validation"] == {
        "rater_count": 3,
        "participant_ids": ["rater-1", "rater-2", "rater-3"],
        "trial_count": 6,
        "complete_trial_coverage_required": True,
        "ratings_per_item_min": 3,
        "ratings_per_item_max": 3,
    }
    identity = report["identity_four_afc"]
    assert identity["item_count"] == 2
    assert identity["rating_count"] == 6
    assert identity["correct_count"] == 3
    assert identity["incorrect_option_count"] == 1
    assert identity["none_count"] == 1
    assert identity["unjudgeable_count"] == 1
    assert identity["accuracy"] == pytest.approx(0.75)
    assert identity["accuracy_all_ratings"] == pytest.approx(0.5)
    assert identity["none_rate"] == pytest.approx(1 / 6)
    assert identity["unjudgeable_rate"] == pytest.approx(1 / 6)

    paired = {row["block_code"]: row for row in report["paired_contrasts"]}
    assert paired["P1"]["focal_condition"] == "k4_diverse"
    assert paired["P1"]["comparison_condition"] == "k4_repeat"
    assert paired["P1"]["win_count"] == 3
    assert paired["P1"]["tie_count"] == 1
    assert paired["P1"]["loss_count"] == 1
    assert paired["P1"]["unjudgeable_count"] == 1
    assert paired["P1"]["win_rate"] == pytest.approx(3 / 5)
    assert paired["P4"]["win_count"] == 2
    assert paired["P4"]["tie_count"] == 1
    assert paired["P4"]["loss_count"] == 2
    assert paired["P4"]["unjudgeable_count"] == 1
    assert all(row["rating_count"] == 3 for row in report["per_item_rating_counts"])

    bootstrap = report["identity_cluster_bootstrap"]
    assert bootstrap["unit"] == "target_identity_id"
    assert bootstrap["identity_four_afc"]["accuracy"]["estimate"] == pytest.approx(0.75)
    p1_bootstrap = next(
        row for row in bootstrap["paired_contrasts"] if row["block_code"] == "P1"
    )
    # Equal-weighting the identity win rates gives (1/3 + 1) / 2, not pooled 3/5.
    assert p1_bootstrap["metrics"]["win_rate"]["estimate"] == pytest.approx(2 / 3)
    assert p1_bootstrap["metrics"]["win_rate"]["ci_low"] >= 0
    assert p1_bootstrap["metrics"]["win_rate"]["ci_high"] <= 1

    repeated = analyze_human_eval(
        key_path,
        list(reversed(response_paths)),
        bootstrap_resamples=200,
        bootstrap_seed=19,
    )
    assert repeated["identity_cluster_bootstrap"] == report["identity_cluster_bootstrap"]

    output = write_human_eval_analysis(tmp_path / "analysis.json", report)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_rejects_duplicate_rater_case_insensitively(tmp_path: Path) -> None:
    key, responses = _bundle()
    responses[1]["participant_id"] = "RATER-1"
    key_path, response_paths = _write_bundle(tmp_path, key, responses)

    with pytest.raises(HumanEvalAnalysisError, match="Duplicate participant_id"):
        analyze_human_eval(key_path, response_paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_study", "study_id mismatch"),
        ("incomplete", "Incomplete trial coverage"),
        ("duplicate_trial", "Duplicate response for trial"),
        ("invalid_selection", "Invalid selection"),
        ("false_completed", "not marked completed"),
    ],
)
def test_rejects_invalid_response_bundles(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    key, responses = _bundle()
    target = responses[0]
    response_rows = target["responses"]
    assert isinstance(response_rows, list)
    if mutation == "wrong_study":
        target["study_id"] = "another-study"
    elif mutation == "incomplete":
        response_rows.pop()
    elif mutation == "duplicate_trial":
        response_rows.append(copy.deepcopy(response_rows[0]))
    elif mutation == "invalid_selection":
        response_rows[0]["selection"] = "none"  # type: ignore[index]
    elif mutation == "false_completed":
        target["completed"] = False
    key_path, response_paths = _write_bundle(tmp_path, key, responses)

    with pytest.raises(HumanEvalAnalysisError, match=message):
        analyze_human_eval(key_path, response_paths)


def test_rejects_corrupt_answer_key_mapping(tmp_path: Path) -> None:
    key, responses = _bundle()
    trials = key["trials"]
    assert isinstance(trials, list)
    identity_trial = trials[0]
    assert isinstance(identity_trial, dict)
    option_identities = identity_trial["option_identity_ids"]
    assert isinstance(option_identities, dict)
    option_identities["a_target"] = "not-id-a"
    key_path, response_paths = _write_bundle(tmp_path, key, responses)

    with pytest.raises(HumanEvalAnalysisError, match="target option does not map"):
        analyze_human_eval(key_path, response_paths)
