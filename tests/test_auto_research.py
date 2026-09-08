import numpy as np
import pytest
from PIL import Image

from id_layers.auto_evaluation import adaface_input
from id_layers.auto_research import Budget, BudgetStop, outside_source, tree_hashes


def test_output_cannot_enter_source(tmp_path):
    source = tmp_path / "old"
    for target in (source, source / "new", source / "a" / ".." / "b"):
        with pytest.raises(ValueError):
            outside_source(target, source)
    assert outside_source(tmp_path / "new", source) == (tmp_path / "new").resolve()


def test_source_digest_detects_same_size_change(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(b"ab")
    old = tree_hashes(tmp_path)
    path.write_bytes(b"ba")
    assert old != tree_hashes(tmp_path)


def test_adaface_exact_official_bgr_normalization():
    pixels = np.zeros((112, 112, 3), dtype=np.uint8)
    pixels[:] = [255, 128, 0]
    actual = adaface_input(Image.fromarray(pixels)).numpy()
    official = ((pixels[:, :, ::-1] / 255.0 - 0.5) / 0.5).transpose(2, 0, 1).astype(np.float32)
    np.testing.assert_allclose(actual, official, atol=6e-8)
    np.testing.assert_allclose(actual[:, 0, 0], [-1, 1 / 255, 1], atol=6e-8)
    with pytest.raises(ValueError):
        adaface_input(Image.new("RGB", (256, 256)))


def test_budget_reservations_and_cached_hash(tmp_path):
    budget = Budget(tmp_path / "ledger.json", download_limit=10, gpu_limit=60)
    budget.state["gpu_seconds"] = 30
    assert budget.can_gpu(30) and not budget.can_gpu(31)
    budget.state["active_gpu"] = {"reservation_seconds": 20}
    assert not budget.can_gpu(11)
    with pytest.raises(BudgetStop):
        budget.download("https://example.invalid/unused", tmp_path / "model", max_bytes=11)
    asset = tmp_path / "cached"
    asset.write_bytes(b"x")
    with pytest.raises(ValueError):
        budget.download("https://example.invalid/unused", asset, max_bytes=1, expected_sha="0" * 64)


def test_cpu_download_save_cannot_erase_gpu_account(tmp_path):
    ledger = tmp_path / "ledger.json"
    download = Budget(ledger)
    download.kind = "download"
    gpu = Budget(ledger)
    gpu.kind = "gpu"
    gpu.state["gpu_seconds"] = 123
    gpu.save()
    download.state["download_bytes"] = 456
    download.save()
    gpu.state["gpu_seconds"] = 124
    gpu.save()
    actual = Budget(ledger).state
    assert actual["gpu_seconds"] == 124 and actual["download_bytes"] == 456
