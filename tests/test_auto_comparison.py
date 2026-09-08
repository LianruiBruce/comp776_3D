import json

import pandas as pd
import pytest

from id_layers.auto_comparison import compare_expression_changes, compare_generation_systems


def test_system_comparison_is_paired_and_reports_noise_mismatch(tmp_path):
    directories = {}
    for model in ("a", "b"):
        generation, evaluation = tmp_path / (model + "_gen"), tmp_path / (model + "_eval")
        generation.mkdir()
        evaluation.mkdir()
        directories[model] = (evaluation, generation)
        records, metrics, reference = [], [], []
        for i in range(2):
            identity = f"SYNTHETIC_{i}"
            common = {
                "sample_id": identity,
                "identity_id": identity,
                "condition": "k1_full",
                "prompt_id": "neutral",
                "base_seed": 776,
            }
            records.append(
                {
                    **common,
                    "effective_seed": 776,
                    "reference_sample_ids": [identity + "_ref"],
                    "width": 2,
                    "height": 2,
                    "latent_sha256": ("a" if model == "a" or i == 0 else "b") * 64,
                    "status": "complete",
                    "prompt": "SYNTHETIC " + model,
                    "num_inference_steps": 30,
                    "guidance_scale": 5,
                }
            )
            score = (0.5 + i * 0.2) if model == "a" else 0.4
            metrics.append(
                {
                    **common,
                    "evaluator": "adaface",
                    "target_similarity": score,
                    "target_impostor_margin": score - 0.1,
                    "end_to_end_rank1": True,
                }
            )
            reference.append(
                {**common, "space": "face", "common_k1_distance": 0.2 if model == "a" else 0.3}
            )
        (generation / "generation_manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        pd.DataFrame(metrics).to_csv(evaluation / "identity_metrics.csv", index=False)
        pd.DataFrame(reference).to_csv(evaluation / "reference_distances.csv", index=False)
    summary = compare_generation_systems(
        *directories["a"],
        *directories["b"],
        tmp_path / "result",
        label_a="SYNTHETIC_A",
        label_b="SYNTHETIC_B",
        expected_cells=2,
        resamples=20,
    )
    assert summary["stored_initial_latent_equal_count"] == 1
    frame = pd.read_csv(tmp_path / "result/identity_summary.csv")
    assert frame.loc[
        frame.metric == "target_similarity_a_minus_b", "estimate"
    ].tolist() == pytest.approx([0.2, 0.2])
    path = directories["b"][1] / "generation_manifest.jsonl"
    records[0]["base_seed"] = 7
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix mismatch"):
        compare_generation_systems(
            *directories["a"],
            *directories["b"],
            tmp_path / "bad",
            label_a="SYNTHETIC_A",
            label_b="SYNTHETIC_B",
            expected_cells=2,
            resamples=20,
        )


def test_expression_interaction_pairs_identities_and_preserves_missing(tmp_path):
    directories = []
    for system in ("a", "b"):
        directory = tmp_path / system
        directory.mkdir()
        directories.append(directory)
        (directory / "status.json").write_text('{"status":"complete"}')
        (directory / "summary.json").write_text('{"new_expression_protocol":true}')
        rows = []
        for identity, difference in (("SYNTHETIC_0", 0.3), ("SYNTHETIC_1", 0.7)):
            for seed in (1, 2):
                rows.append(
                    {
                        "metric": "mouth_smile_mean",
                        "evaluator": None,
                        "identity_id": identity,
                        "condition": "k1_full",
                        "base_seed": seed,
                        "difference": difference if system == "a" else 0.1,
                    }
                )
        rows[0]["difference"] = None
        pd.DataFrame(rows[::-1] if system == "b" else rows).to_csv(
            directory / "expression_paired_cells.csv",
            index=False,
        )
    result = compare_expression_changes(
        *directories,
        tmp_path / "result",
        label_a="SYNTHETIC_A",
        label_b="SYNTHETIC_B",
        resamples=20,
    )
    assert result["identity_count"] == 2
    frame = pd.read_csv(tmp_path / "result/interaction_summary.csv")
    assert frame.iloc[0].estimate == pytest.approx(0.4)
    assert frame.iloc[0].valid_row_count == 3
    path = directories[1] / "expression_paired_cells.csv"
    bad = pd.read_csv(path)
    bad.loc[0, "base_seed"] = 99
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different paired matrices"):
        compare_expression_changes(
            *directories,
            tmp_path / "bad",
            label_a="SYNTHETIC_A",
            label_b="SYNTHETIC_B",
            resamples=20,
        )
