from id_layers.data import FEI_MEMBER_PATTERN, split_identity_ids


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
