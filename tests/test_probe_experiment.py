from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from id_layers.probe_experiment import (
    _condition_wise_shuffled_labels,
    _frozen_evaluation,
    _load_probe,
    _save_probe_checkpoint,
    _state_sha256,
)
from id_layers.probing import build_matched_probe


def test_condition_wise_shuffle_is_not_one_global_identity_renaming() -> None:
    identities = ["a", "b", "c", "d"]
    manifest = pd.DataFrame(
        [
            {"identity_id": identity, "condition": condition}
            for condition in ["left", "frontal", "right"]
            for identity in identities
        ]
    )

    shuffled = _condition_wise_shuffled_labels(manifest, seed=776)
    manifest = manifest.assign(shuffled_identity=shuffled)

    for _, group in manifest.groupby("condition"):
        assert sorted(group["shuffled_identity"]) == sorted(group["identity_id"])
        assert not np.array_equal(
            group["shuffled_identity"].to_numpy(), group["identity_id"].to_numpy()
        )

    # A global category renaming would map every occurrence of an original identity
    # to one fixed output label. Independent condition permutations do not.
    mapped_label_counts = manifest.groupby("identity_id")["shuffled_identity"].nunique()
    assert bool((mapped_label_counts > 1).any())


def _identity_embedding(identity: str) -> np.ndarray:
    index = {"a": 0, "b": 1, "c": 2}[identity]
    value = np.zeros(3, dtype=np.float32)
    value[index] = 1.0
    return value


def _aligned_manifest(split: str, query_conditions: dict[str, str]) -> pd.DataFrame:
    rows = []
    for identity in ["a", "b", "c"]:
        rows.extend(
            [
                {
                    "identity_id": identity,
                    "protocol_role": "reference",
                    "condition": "reference",
                    "split": split,
                    "usable_for_model": True,
                },
                {
                    "identity_id": identity,
                    "protocol_role": "query",
                    "condition": query_conditions[identity],
                    "split": split,
                    "usable_for_model": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def _embeddings_for_manifest(manifest: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [_identity_embedding(str(identity)) for identity in manifest["identity_id"]]
    )


def test_frozen_evaluation_separates_aligned_and_end_to_end_denominators() -> None:
    calibration_manifest = _aligned_manifest(
        "dev", {"a": "bright", "b": "bright", "c": "bright"}
    )
    evaluation_manifest = _aligned_manifest(
        "internal_eval", {"a": "bright", "b": "dark", "c": "dark"}
    )
    full_manifest = pd.concat(
        [
            evaluation_manifest,
            pd.DataFrame(
                [
                    {
                        "identity_id": "a",
                        "protocol_role": "query",
                        "condition": "dark",
                        "split": "internal_eval",
                        "usable_for_model": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    metric_rows, identity_rows, condition_rows = _frozen_evaluation(
        method="synthetic",
        layer=2,
        seed=776,
        calibration_embeddings=_embeddings_for_manifest(calibration_manifest),
        calibration_manifest=calibration_manifest,
        evaluation_embeddings=_embeddings_for_manifest(evaluation_manifest),
        evaluation_manifest=evaluation_manifest,
        full_manifest=full_manifest,
        evaluation_split="internal_eval",
        far_targets=[0.0],
    )

    assert len(metric_rows) == 1
    assert metric_rows[0]["template_queries"] == 3
    identity_frame = pd.DataFrame(identity_rows).set_index("identity_id")
    assert identity_frame.loc["a", "source_query_count"] == 2
    assert identity_frame.loc["a", "usable_query_count"] == 1
    assert identity_frame.loc["a", "scored_query_count"] == 1
    assert identity_frame.loc["a", "accepted_query_count"] == 1
    assert identity_frame.loc["a", "aligned_only_tar"] == 1.0
    assert identity_frame.loc["a", "end_to_end_tar"] == 0.5
    assert identity_frame.loc["a", "tar"] == 0.5

    condition_frame = pd.DataFrame(condition_rows).set_index("query_condition")
    assert condition_frame.loc["dark", "source_query_count"] == 3
    assert condition_frame.loc["dark", "usable_query_count"] == 2
    assert condition_frame.loc["dark", "scored_query_count"] == 2
    assert condition_frame.loc["dark", "query_coverage"] == pytest.approx(2 / 3)
    assert condition_frame.loc["dark", "scored_query_coverage"] == pytest.approx(2 / 3)
    assert condition_frame.loc["dark", "accepted_query_count"] == 2
    assert condition_frame.loc["dark", "aligned_only_tar"] == 1.0
    assert condition_frame.loc["dark", "end_to_end_tar"] == pytest.approx(2 / 3)
    assert condition_frame.loc["bright", "query_coverage"] == 1.0


def test_frozen_evaluation_records_a_condition_with_zero_scored_queries() -> None:
    calibration_manifest = _aligned_manifest(
        "dev", {"a": "bright", "b": "bright", "c": "bright"}
    )
    evaluation_manifest = _aligned_manifest(
        "internal_eval", {"a": "bright", "b": "bright", "c": "bright"}
    )
    full_manifest = pd.concat(
        [
            evaluation_manifest,
            pd.DataFrame(
                [
                    {
                        "identity_id": identity,
                        "protocol_role": "query",
                        "condition": "dark",
                        "split": "internal_eval",
                        "usable_for_model": False,
                    }
                    for identity in ["a", "b", "c"]
                ]
            ),
        ],
        ignore_index=True,
    )

    _, _, condition_rows = _frozen_evaluation(
        method="synthetic",
        layer=2,
        seed=776,
        calibration_embeddings=_embeddings_for_manifest(calibration_manifest),
        calibration_manifest=calibration_manifest,
        evaluation_embeddings=_embeddings_for_manifest(evaluation_manifest),
        evaluation_manifest=evaluation_manifest,
        full_manifest=full_manifest,
        evaluation_split="internal_eval",
        far_targets=[0.0],
    )

    dark = pd.DataFrame(condition_rows).set_index("query_condition").loc["dark"]
    assert dark["source_query_count"] == 3
    assert dark["usable_query_count"] == 0
    assert dark["scored_query_count"] == 0
    assert dark["query_coverage"] == 0.0
    assert dark["scored_query_coverage"] == 0.0
    assert dark["accepted_query_count"] == 0
    assert pd.isna(dark["aligned_only_tar"])
    assert dark["end_to_end_tar"] == 0.0
    assert pd.isna(dark["aligned_observed_far"])


def test_probe_checkpoint_save_load_roundtrip(tmp_path) -> None:
    probe, classifier = build_matched_probe(
        input_dim=3,
        output_dim=4,
        num_classes=3,
        seed=776,
        scale=16.0,
        margin=0.2,
    )
    result = SimpleNamespace(
        probe=probe,
        classifier=classifier,
        class_labels=("a", "b", "c"),
        seed=776,
        config={"output_dim": 4, "epochs": 2},
        optimizer_steps=6,
    )
    path = tmp_path / "probe.pt"

    record = _save_probe_checkpoint(
        path,
        result,
        layer=2,
        control="trained",
    )
    loaded = _load_probe(
        path,
        input_dim=3,
        output_dim=4,
        expected_layer=2,
        expected_seed=776,
        expected_control="trained",
        expected_file_sha256=record["file_sha256"],
        expected_probe_state_sha256=record["probe_state_sha256"],
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert path.is_file()
    assert record["layer"] == 2
    assert record["seed"] == 776
    assert record["control"] == "trained"
    assert record["probe_state_sha256"] == _state_sha256(probe.state_dict())
    assert record["classifier_state_sha256"] == _state_sha256(classifier.state_dict())
    assert payload["class_labels"] == ("a", "b", "c")
    assert payload["training_config"] == {"output_dim": 4, "epochs": 2}
    assert payload["optimizer_steps"] == 6
    assert not loaded.training
    for name, expected in probe.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)

    features = torch.tensor([[0.2, -0.4, 0.8], [1.0, 0.0, -1.0]])
    torch.testing.assert_close(loaded(features), probe.eval()(features), rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="seed mismatch"):
        _load_probe(path, input_dim=3, output_dim=4, expected_seed=1776)
