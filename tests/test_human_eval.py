from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from id_layers.human_eval import HumanEvalError, build_human_eval


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _study_inputs(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    identities = ["person_alpha", "person_beta", "person_gamma", "person_delta"]
    gallery_conditions = [
        "frontal_smile",
        "left_three_quarter",
        "right_three_quarter_expression",
    ]
    reference_rows: list[dict[str, object]] = []
    for index, identity in enumerate(identities):
        for condition_index, condition in enumerate(gallery_conditions):
            filename = f"secret_reference_{identity}_{condition}.png"
            _write_image(
                tmp_path / "data" / "processed" / "references" / filename,
                (
                    20 + index * 30,
                    40 + condition_index * 30,
                    60 + index * 10 + condition_index,
                ),
            )
            reference_rows.append(
                {
                    "sample_id": f"reference-{index}-{condition_index}",
                    "identity_id": identity,
                    "condition": condition,
                    "protocol_role": "query",
                    "usable_for_model": "true",
                    "relative_path": f"processed/references/{filename}",
                }
            )

    manifest = tmp_path / "data" / "manifests" / "references.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)

    condition_names = [
        "k1_full",
        "k4_diverse_full",
        "k4_repeat_full",
        "global_target_patch_donor",
        "global_donor_patch_target",
        "text_only",
    ]
    result_rows: list[dict[str, object]] = []
    for index, condition in enumerate(condition_names):
        filename = f"secret_generation_{condition}.png"
        _write_image(
            tmp_path / "run" / "images" / filename,
            (80 + index * 20, 100 + index * 15, 120 + index * 10),
        )
        if condition == "global_donor_patch_target":
            global_identity, patch_identity = "person_beta", "person_alpha"
        elif condition == "global_target_patch_donor":
            global_identity, patch_identity = "person_alpha", "person_beta"
        else:
            global_identity = patch_identity = "person_alpha"
        result_rows.append(
            {
                "sample_id": f"generated-{index}",
                "pair_id": "target-donor-prompt-seed",
                "identity_id": "person_alpha",
                "donor_identity_id": "person_beta",
                "method": "PhotoMaker-V2-secret",
                "condition": condition,
                "prompt_id": "private_prompt_name",
                "effective_seed": 12345,
                "output_relative_path": f"images/{filename}",
                "status": "complete",
                "global_embedding_identity_id": global_identity,
                "patch_image_identity_id": patch_identity,
            }
        )
    results = tmp_path / "run" / "generation_manifest.jsonl"
    _write_jsonl(results, result_rows)
    return manifest, results, result_rows


def test_builds_blinded_directory_study_without_embedding_faces(tmp_path: Path) -> None:
    manifest, results, _ = _study_inputs(tmp_path)
    output = tmp_path / "public_study"

    build = build_human_eval(manifest, results, output, seed=776)

    assert build.trial_count == 11
    assert build.html_path == output / "index.html"
    assert build.blinded_json_path == output / "blinded_trials.json"
    assert build.media_dir == output / "media"
    assert build.html_path.is_file()
    assert build.blinded_json_path.is_file()
    assert build.media_dir.is_dir()
    assert all("block_code" not in trial for trial in build.public_study["trials"])
    assert build.public_study["task_counts"] == {"four_afc": 6, "pairwise": 5}
    assert sorted(trial["block_code"] for trial in build.private_answer_key["trials"]) == [
        "I1",
        "I1",
        "I1",
        "I1",
        "I1",
        "I1",
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
    ]

    public_text = json.dumps(build.public_study, ensure_ascii=False, sort_keys=True)
    html = build.html_path.read_text(encoding="utf-8")
    for secret in (
        "PhotoMaker-V2-secret",
        "person_alpha",
        "person_beta",
        "private_prompt_name",
        "k1_full",
        "k4_diverse",
        "k4_repeat",
        "global_target_patch_donor",
        "secret_reference",
        "secret_generation",
    ):
        assert secret not in public_text
        assert secret not in html
    assert "data:image" not in public_text.lower()
    assert "data:image" not in html.lower()
    assert "localStorage" in html
    assert "Download responses JSON" in html
    assert "new Blob" in html
    assert "Tie / equally similar" in html
    assert "Unjudgeable" in html

    pair_trials = [
        trial for trial in build.public_study["trials"] if trial["kind"] == "pairwise"
    ]
    identity_trials = [
        trial for trial in build.public_study["trials"] if trial["kind"] == "four_afc"
    ]
    assert all(len(trial["options"]) == 2 for trial in pair_trials)
    assert all(len(trial["reference_images"]) == 3 for trial in pair_trials)
    assert len(identity_trials) == 6
    assert all(len(trial["options"]) == 4 for trial in identity_trials)
    assert all(
        len(option["images"]) == 3
        for trial in identity_trials
        for option in trial["options"]
    )
    assert all(trial["special_responses"] == ["none", "unjudgeable"] for trial in identity_trials)

    hrefs = set(build.public_study["media"]["sha256"])
    assert hrefs
    for href in hrefs:
        assert re.fullmatch(r"media/img_[0-9a-f]{24}\.png", href)
        assert (output / href).is_file()
    assert set(output.iterdir()) == {
        output / "index.html",
        output / "blinded_trials.json",
        output / "media",
    }

    private_text = json.dumps(build.private_answer_key, ensure_ascii=False, sort_keys=True)
    assert "PhotoMaker-V2-secret" in private_text
    assert "person_alpha" in private_text
    assert "k4_diverse" in private_text
    identity_keys = [
        trial for trial in build.private_answer_key["trials"] if trial["kind"] == "four_afc"
    ]
    assert len(identity_keys) == 6
    assert {trial["variant"] for trial in identity_keys} == {
        "k1",
        "k4_diverse",
        "k4_repeat",
        "global_target_patch_donor",
        "global_donor_patch_target",
        "text_only",
    }
    for trial in identity_keys:
        assert set(trial["option_roles"].values()) == {
            "target_identity",
            "donor_identity",
            "foil_1",
            "foil_2",
        }
        if trial["variant"] in {
            "global_target_patch_donor",
            "global_donor_patch_target",
        }:
            assert {trial["global_identity_id"], trial["patch_identity_id"]} == {
                "person_alpha",
                "person_beta",
            }


def test_randomization_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    manifest, results, _ = _study_inputs(tmp_path)

    first = build_human_eval(manifest, results, tmp_path / "first", seed=20260903)
    second = build_human_eval(manifest, results, tmp_path / "second", seed=20260903)
    different = build_human_eval(manifest, results, tmp_path / "different", seed=20260904)

    assert first.public_study == second.public_study
    assert first.private_answer_key["trials"] == second.private_answer_key["trials"]
    assert first.public_study != different.public_study


def test_html_file_layout_uses_sibling_blinded_json_and_opaque_media(tmp_path: Path) -> None:
    manifest, results, _ = _study_inputs(tmp_path)

    build = build_human_eval(manifest, results, tmp_path / "survey.html", seed=11)

    assert build.html_path == tmp_path / "survey.html"
    assert build.blinded_json_path == tmp_path / "survey.blinded.json"
    assert build.media_dir == tmp_path / "survey_media"
    hrefs = set(build.public_study["media"]["sha256"])
    assert all(href.startswith("survey_media/img_") for href in hrefs)
    assert all((tmp_path / href).is_file() for href in hrefs)


def test_rejects_inline_image_data_and_relevant_generation_failures(tmp_path: Path) -> None:
    manifest, results, rows = _study_inputs(tmp_path)
    inline_rows = [dict(row) for row in rows]
    inline_rows[0]["output_relative_path"] = "data:image/png;base64,AAAA"
    inline_results = tmp_path / "run" / "inline.jsonl"
    _write_jsonl(inline_results, inline_rows)
    with pytest.raises(HumanEvalError, match="Inline data/base64"):
        build_human_eval(manifest, inline_results, tmp_path / "inline-output", seed=1)

    failed_rows = [dict(row) for row in rows]
    failed_rows[0]["status"] = "failed"
    failed_results = tmp_path / "run" / "failed.jsonl"
    _write_jsonl(failed_results, failed_rows)
    with pytest.raises(HumanEvalError, match="must not be silently dropped"):
        build_human_eval(manifest, failed_results, tmp_path / "failed-output", seed=1)


def test_requires_prespecified_k_comparisons(tmp_path: Path) -> None:
    manifest, _, rows = _study_inputs(tmp_path)
    incomplete_rows = [
        row for row in rows if row["condition"] != "k4_diverse_full"
    ]
    incomplete_results = tmp_path / "run" / "incomplete.jsonl"
    _write_jsonl(incomplete_results, incomplete_rows)

    with pytest.raises(HumanEvalError, match=r"missing \['P0', 'P1'\]"):
        build_human_eval(manifest, incomplete_results, tmp_path / "incomplete", seed=776)
