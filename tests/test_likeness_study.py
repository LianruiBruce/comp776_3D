from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from id_layers.likeness_study import (
    analyze_likeness_study,
    parse_responses,
    validate_candidates,
    validate_photo_roles,
)


def synthetic_study(count: int = 2):
    """This fixture contains no participant responses or real face data."""
    identities, photos, candidates, raters, assignments, responses = [], [], [], [], [], []
    for i in range(count):
        identity, self_id = f"synthetic_{i}", f"self_{i}"
        identities.append(
            {
                "identity_id": identity,
                "split": "development",
                "self_rater_id": self_id,
                "analysis_consent": True,
                "data_kind": "synthetic",
            }
        )
        raters.append({"rater_id": self_id, "known_identity_ids": [identity]})
        role_photos = {}
        for role, number in (("input", 1), ("gallery", 3), ("real_neutral", 2), ("real_smile", 2)):
            role_photos[role] = []
            for k in range(number):
                sample = f"{identity}_{role}_{k}"
                row = {
                    "sample_id": sample,
                    "identity_id": identity,
                    "role": role,
                    "sha256": sha256(sample.encode()).hexdigest(),
                }
                photos.append(row)
                role_photos[role].append(row)
        for expression in ("neutral", "smile"):
            for kind in ("real", "generated"):
                for replicate in (1, 2):
                    cid = f"{identity}_{kind}_{expression}_{replicate}"
                    row = {
                        "candidate_id": cid,
                        "identity_id": identity,
                        "kind": kind,
                        "model": None if kind == "real" else "synthetic_model",
                        "expression": expression,
                        "replicate": replicate,
                        "status": "ok",
                    }
                    if kind == "real":
                        photo = role_photos["real_" + expression][replicate - 1]
                        row.update(photo_sample_id=photo["sample_id"], sha256=photo["sha256"])
                    else:
                        row.update(
                            input_sample_ids=[role_photos["input"][0]["sample_id"]],
                            seed=776 + replicate,
                            sha256=sha256(cid.encode()).hexdigest(),
                        )
                    candidates.append(row)
                    aid = cid + "_self"
                    assignments.append(
                        {
                            "assignment_id": aid,
                            "candidate_id": cid,
                            "candidate_sha256": row["sha256"],
                            "rater_id": self_id,
                            "group": "self",
                        }
                    )
                    # identity 0: R0=6,G0=5,R1=6,G1=4 => D=1.5,E=1.
                    # identity 1: all 5 => D=0,E=0.
                    score = (
                        (6 if kind == "real" else (5 if expression == "neutral" else 4))
                        if i == 0
                        else 5
                    )
                    responses.append({"assignment_id": aid, "status": "rated", "score": score})
    return identities, photos, candidates, raters, assignments, responses


def self_summary(result, endpoint="D"):
    return next(r for r in result["summary"] if r["group"] == "self" and r["endpoint"] == endpoint)


def test_effect_direction_identity_equality_and_order_invariance():
    data = synthetic_study()
    result = analyze_likeness_study(*data, bootstrap_resamples=30, seed=5)
    assert result["evidence_type"] == "synthetic"
    assert self_summary(result)["estimate"] == pytest.approx(0.75)
    assert self_summary(result, "E")["estimate"] == pytest.approx(0.5)
    shuffled = analyze_likeness_study(
        *(list(reversed(x)) for x in data), bootstrap_resamples=30, seed=5
    )
    assert result == shuffled


def test_missing_not_zero_and_bounds_include_no_response_identity():
    data = list(synthetic_study())
    data[-1] = [r for r in data[-1] if "synthetic_0" in r["assignment_id"]]
    result = analyze_likeness_study(*data, bootstrap_resamples=20)
    summary = self_summary(result)
    assert summary["estimate"] == 1.5
    assert summary["estimable_identity_count"] == 1
    assert summary["bounds_identity_count"] == 2
    assert summary["missing_response_identification_bounds"] == pytest.approx([-2.25, 3.75])
    assert result["response_counts"]["self"]["missing"] == 8


def test_one_valid_candidate_per_cell_retains_equal_cell_weights():
    data = list(synthetic_study(1))
    data[-1] = [r for r in data[-1] if "generated_neutral_2" not in r["assignment_id"]]
    result = analyze_likeness_study(*data, bootstrap_resamples=20)
    assert self_summary(result)["estimate"] == 1.5
    assert self_summary(result)["complete_eight_candidate_identity_count"] == 0


def test_generation_failure_is_not_a_likeness_score_or_missing_bound():
    data = list(synthetic_study(1))
    failure = next(c for c in data[2] if c["kind"] == "generated")
    failure.update(status="generation_failed", sha256=None, failure_reason="synthetic OOM")
    removed = {a["assignment_id"] for a in data[4] if a["candidate_id"] == failure["candidate_id"]}
    data[4] = [a for a in data[4] if a["assignment_id"] not in removed]
    data[5] = [r for r in data[5] if r["assignment_id"] not in removed]
    result = analyze_likeness_study(*data, bootstrap_resamples=20)
    assert self_summary(result)["estimate"] == 1.5
    assert self_summary(result)["missing_response_identification_bounds"] is None
    assert self_summary(result)["excluded_missing_image_identity_count"] == 1


def test_reject_role_hash_split_and_real_candidate_reuse():
    data = synthetic_study()
    photos = deepcopy(data[1])
    photos[1]["sha256"] = photos[0]["sha256"]
    with pytest.raises(ValueError, match="Duplicate photograph hash"):
        validate_photo_roles(data[0], photos)
    photos = deepcopy(data[1])
    photos[0]["split"] = "confirmation"
    with pytest.raises(ValueError, match="split differs"):
        validate_photo_roles(data[0], photos)
    candidates = deepcopy(data[2])
    real = [c for c in candidates if c["kind"] == "real"]
    real[1].update(photo_sample_id=real[0]["photo_sample_id"], sha256=real[0]["sha256"])
    with pytest.raises(ValueError, match="reused"):
        validate_candidates(data[0], data[1], candidates)


def test_unavailable_real_candidate_remains_in_operational_ledger():
    data = list(synthetic_study(1))
    candidate = next(c for c in data[2] if c["kind"] == "real")
    candidate.update(status="image_unavailable", sha256=None, failure_reason="synthetic lost image")
    removed = {
        a["assignment_id"] for a in data[4] if a["candidate_id"] == candidate["candidate_id"]
    }
    data[4] = [a for a in data[4] if a["assignment_id"] not in removed]
    data[5] = [r for r in data[5] if r["assignment_id"] not in removed]
    result = analyze_likeness_study(*data, bootstrap_resamples=10)
    assert self_summary(result)["estimate"] == 1.5
    assert self_summary(result)["excluded_missing_image_identity_count"] == 1


def test_reject_input_leakage_hash_and_wrong_expression_pair():
    data = synthetic_study()
    candidates = deepcopy(data[2])
    generated = next(c for c in candidates if c["kind"] == "generated")
    generated["input_sample_ids"] = [
        next(p["sample_id"] for p in data[1] if p["role"] == "gallery")
    ]
    with pytest.raises(ValueError, match="leakage"):
        validate_candidates(data[0], data[1], candidates)
    candidates = deepcopy(data[2])
    generated = next(
        c for c in candidates if c["kind"] == "generated" and c["expression"] == "smile"
    )
    generated["seed"] += 5
    with pytest.raises(ValueError, match="share input and seed"):
        validate_candidates(data[0], data[1], candidates)
    assignments = deepcopy(data[4])
    assignments[0]["candidate_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        parse_responses(data[0], data[2], data[3], assignments, data[5])


def test_reject_duplicate_response_wrong_self_and_invalid_missing():
    data = synthetic_study()
    with pytest.raises(ValueError, match="Duplicate assignment_id"):
        parse_responses(data[0], data[2], data[3], data[4], data[5] + [data[5][0]])
    assignments = deepcopy(data[4])
    assignments[0]["rater_id"] = "self_1"
    with pytest.raises(ValueError, match="one self rater"):
        parse_responses(data[0], data[2], data[3], assignments, data[5])
    responses = deepcopy(data[5])
    responses[0].update(status="unjudgeable", score=0)
    with pytest.raises(ValueError, match="null score"):
        parse_responses(data[0], data[2], data[3], data[4], responses)


def test_unfamiliar_crossed_raters_remain_separate_and_candidate_equal():
    data = list(synthetic_study())
    data[3] += [{"rater_id": f"stranger_{k}", "known_identity_ids": []} for k in range(3)]
    for candidate in data[2]:
        for k in range(3 if candidate["replicate"] == 1 else 1):
            aid = candidate["candidate_id"] + f"_stranger_{k}"
            data[4].append(
                {
                    "assignment_id": aid,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_sha256": candidate["sha256"],
                    "rater_id": f"stranger_{k}",
                    "group": "unfamiliar",
                }
            )
            data[5].append({"assignment_id": aid, "status": "rated", "score": 5})
    result = analyze_likeness_study(*data, bootstrap_resamples=40, seed=7)
    stranger = next(
        s for s in result["summary"] if s["group"] == "unfamiliar" and s["endpoint"] == "D"
    )
    assert stranger["rater_count"] == 3
    assert stranger["estimate"] == 0
    assert stranger["ci95"] == [0, 0]
    assert "crossed" in stranger["interval_method"]
    assert self_summary(result)["estimate"] == 0.75


def test_unfamiliar_cannot_rate_own_or_known_identity():
    data = list(synthetic_study())
    data[4][0]["group"] = "unfamiliar"
    with pytest.raises(ValueError, match="known or own"):
        parse_responses(data[0], data[2], data[3], data[4], data[5])


def test_withdrawn_identity_stays_out_of_estimates_and_bounds():
    data = synthetic_study()
    data[0][0]["analysis_consent"] = False
    result = analyze_likeness_study(*data, bootstrap_resamples=10)
    assert self_summary(result)["estimate"] == 0
    assert self_summary(result)["bounds_identity_count"] == 1
    assert result["withdrawn_identity_count"] == 1


def test_file_hash_verification_rejects_modified_content(tmp_path):
    data = synthetic_study(1)
    for photo in data[1]:
        path = tmp_path / photo["sample_id"]
        path.write_bytes(photo["sample_id"].encode())
        photo["path"] = path.name
    validate_photo_roles(data[0], data[1], root=tmp_path)
    (tmp_path / data[1][0]["path"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="Hash mismatch"):
        validate_photo_roles(data[0], data[1], root=tmp_path)
