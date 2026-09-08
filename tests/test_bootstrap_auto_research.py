from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bootstrap_auto_research_under_test", ROOT / "scripts/bootstrap_auto_research.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_versioned_lock_is_portable_and_all_diagnostic_sources_are_pinned() -> None:
    lock = bootstrap.load_frozen_lock(ROOT / "configs/auto_research.assets.lock.yaml")
    assert (
        lock["diagnostic_source_revisions"]["adaface"] == "c60eaa786a42c03444f3df7096dbaf9d57ae010d"
    )
    assert lock["diagnostic_packages"]["lpips"]["version"] == "0.1.4"
    assert lock["pulid"]["code_revision"] == "1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa"
    for group in (
        lock["diagnostic_packages"].values(),
        lock["diagnostic_assets"].values(),
        lock["pulid"]["assets"],
    ):
        for record in group:
            assert not Path(record["path"]).is_absolute()
            assert "resolved_url" not in record
            assert "reused" not in record
            assert "status" not in record
    assert (
        lock["packaged_resources"]["lpips_alex_v01_calibration"]["sha256"]
        == "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
    )


def test_missing_versioned_lock_has_no_network_or_live_branch_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no mutable-source fallback"):
        bootstrap.load_frozen_lock(tmp_path / "missing.yaml", tmp_path)


def test_loader_rejects_mutable_adaface_source_url(tmp_path: Path) -> None:
    frozen = yaml.safe_load((ROOT / "configs/auto_research.assets.lock.yaml").read_text())
    frozen["diagnostic_assets"]["adaface_net.py"]["url"] = (
        "https://raw.githubusercontent.com/mk-minchul/AdaFace/master/net.py"
    )
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="not pinned"):
        bootstrap.load_frozen_lock(path, tmp_path)


def test_corrupt_cache_is_rejected_before_any_download_attempt(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"wrong")
    record = {
        "path": path.name,
        "sha256": hashlib.sha256(b"right").hexdigest(),
        "bytes": 5,
        "url": "https://example.invalid/weights.bin",
    }

    class NeverDownload:
        def download(self, *args, **kwargs):
            raise AssertionError(
                "Corrupt cached assets must be rejected before invoking network code"
            )

    with pytest.raises(ValueError, match="cached asset mismatch"):
        bootstrap.acquire_locked_asset(NeverDownload(), tmp_path, "weights", record)
    assert path.read_bytes() == b"wrong"


def test_frozen_download_receives_exact_sha_and_byte_limit(tmp_path: Path) -> None:
    content = b"exact"
    record = {
        "path": "weights.bin",
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "url": "https://example.invalid/pinned.bin",
    }

    class FakeBudget:
        def download(self, url, target, *, max_bytes, expected_sha):
            assert url == record["url"]
            assert max_bytes == len(content)
            assert expected_sha == record["sha256"]
            target.write_bytes(content)
            return {"reused": False}

    result = bootstrap.acquire_locked_asset(FakeBudget(), tmp_path, "weights", record)
    assert result["sha256"] == record["sha256"]


def test_lock_paths_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative to repository"):
        bootstrap.locked_path(tmp_path, "../weights.bin")
    with pytest.raises(ValueError, match="relative to repository"):
        bootstrap.locked_path(tmp_path, str(tmp_path / "weights.bin"))
