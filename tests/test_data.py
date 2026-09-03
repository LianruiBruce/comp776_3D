import pandas as pd
import pytest

from id_layers.data import (
    FEI_MEMBER_PATTERN,
    FEI_ORIGINAL_MEMBER_PATTERN,
    FEI_QUERY_SLOTS,
    _validate_cached_fei_full_manifest,
    split_identity_ids,
)


def test_fei_filename_parser() -> None:
    match = FEI_MEMBER_PATTERN.fullmatch("17b.jpg")
    assert match is not None
    assert match.group("identity") == "17"
    assert match.group("condition") == "b"
    assert FEI_MEMBER_PATTERN.fullmatch("../17b.jpg") is None


def test_identity_split_is_complete_disjoint_and_deterministic() -> None:
    identities = list(range(1, 11))
    counts = {"dev": 6, "val": 2, "test": 2}
    first = split_identity_ids(identities, counts, seed=776)
    second = split_identity_ids(identities, counts, seed=776)
    assert first == second
    assert set(first) == set(identities)
    assert {split: list(first.values()).count(split) for split in counts} == counts


def test_fei_original_filename_and_protocol_slots() -> None:
    match = FEI_ORIGINAL_MEMBER_PATTERN.fullmatch("17-04.jpg")
    assert match is not None
    assert match.group("identity") == "17"
    assert match.group("slot") == "04"
    assert FEI_ORIGINAL_MEMBER_PATTERN.fullmatch("../17-04.jpg") is None
    assert {1, 3, 8, 10, 12, 14} == FEI_QUERY_SLOTS


def _fei_full_source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["fei_0001_01", "fei_0001_02"],
            "split": ["train", "train"],
            "protocol_role": ["query", "reference"],
            "condition": ["left_profile_extreme", "left_profile"],
            "acquisition_slot": [1, 2],
            "source_sha256": ["a" * 64, "b" * 64],
        }
    )


def test_cached_fei_full_manifest_is_reordered_and_strictly_validated() -> None:
    expected = _fei_full_source_rows()
    cached = expected.iloc[::-1].copy()
    cached["preprocessing_fingerprint"] = "fingerprint"

    validated = _validate_cached_fei_full_manifest(cached, expected, "fingerprint")
    assert validated["sample_id"].tolist() == expected["sample_id"].tolist()

    corruptions = {
        "split": "dev",
        "protocol_role": "reference",
        "condition": "wrong_condition",
        "acquisition_slot": 14,
        "source_sha256": "c" * 64,
    }
    for field, bad_value in corruptions.items():
        corrupted = cached.copy()
        corrupted.loc[corrupted["sample_id"] == "fei_0001_01", field] = bad_value
        with pytest.raises(ValueError, match=field):
            _validate_cached_fei_full_manifest(corrupted, expected, "fingerprint")


def test_cached_fei_full_manifest_rejects_fingerprint_and_sample_set_changes() -> None:
    expected = _fei_full_source_rows()
    cached = expected.copy()
    cached["preprocessing_fingerprint"] = "stale"
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_cached_fei_full_manifest(cached, expected, "current")

    cached["preprocessing_fingerprint"] = "current"
    cached.loc[1, "sample_id"] = "unexpected"
    with pytest.raises(ValueError, match="sample set"):
        _validate_cached_fei_full_manifest(cached, expected, "current")
