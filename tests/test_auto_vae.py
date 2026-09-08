from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from id_layers.auto_research import dump
from id_layers.auto_vae import (
    _read_terminal_unit,
    decode_posterior_mode,
    prepare_vae_input,
    reconstruction_metrics,
    restore_vae_output,
)


class _ModeOnlyDistribution:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def mode(self) -> torch.Tensor:
        return self.tensor

    def sample(self) -> torch.Tensor:
        raise AssertionError("A deterministic VAE reconstruction must never sample")


class _IdentityModeVAE:
    config = SimpleNamespace(scaling_factor=0.13025)

    def encode(self, value: torch.Tensor, *, return_dict: bool):
        return SimpleNamespace(latent_dist=_ModeOnlyDistribution(value))

    def decode(self, value: torch.Tensor, *, return_dict: bool):
        self.decoder_input = value.clone()
        return SimpleNamespace(sample=value)


def test_native_fei_dimensions_have_no_padding_or_resizing() -> None:
    pixels = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3).astype(np.uint8)
    tensor, geometry = prepare_vae_input(Image.fromarray(pixels))
    assert tensor.dtype == torch.float32
    assert tensor.shape == (1, 3, 480, 640)
    assert geometry["padding_bottom"] == geometry["padding_right"] == 0
    restored = restore_vae_output(tensor, geometry)
    assert np.array_equal(np.rint(restored * 255).astype(np.uint8), pixels)


def test_padding_is_bottom_right_reflection_and_geometry_restores_exactly() -> None:
    pixels = np.arange(13 * 17 * 3, dtype=np.uint32).reshape(13, 17, 3).astype(np.uint8)
    tensor, geometry = prepare_vae_input(Image.fromarray(pixels))
    assert tensor.shape == (1, 3, 16, 24)
    assert geometry["padding_bottom"] == 3
    assert geometry["padding_right"] == 7
    assert geometry["padding_top"] == geometry["padding_left"] == 0
    np.testing.assert_allclose(tensor[0, :, 13, 0], pixels[11, 0].astype(np.float32) / 127.5 - 1)
    assert np.array_equal(
        np.rint(restore_vae_output(tensor, geometry) * 255).astype(np.uint8), pixels
    )


def test_mode_roundtrip_decodes_raw_latent_without_unilateral_scaling() -> None:
    vae = _IdentityModeVAE()
    tensor = torch.linspace(-1, 1, 3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    latent, decoded = decode_posterior_mode(vae, tensor)
    assert torch.equal(latent, tensor)
    assert torch.equal(vae.decoder_input, latent)
    assert torch.equal(decoded, tensor)
    diffusion_latent = latent * vae.config.scaling_factor
    decoder_latent = diffusion_latent / vae.config.scaling_factor
    torch.testing.assert_close(decoder_latent, latent, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(diffusion_latent, latent)
    assert not torch.allclose(latent / vae.config.scaling_factor, latent)
    # The same input/mode is deterministic; the fake distribution rejects sample().
    _, repeated = decode_posterior_mode(vae, tensor)
    assert torch.equal(decoded, repeated)


def test_vae_half_precision_and_nonfinite_latents_are_rejected() -> None:
    with pytest.raises(ValueError, match="float32"):
        decode_posterior_mode(_IdentityModeVAE(), torch.zeros(1, 3, 16, 16, dtype=torch.float16))
    tensor = torch.zeros(1, 3, 16, 16)
    tensor[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite float32"):
        decode_posterior_mode(_IdentityModeVAE(), tensor)


def test_decoder_shape_and_nonfinite_outputs_do_not_become_valid_images() -> None:
    tensor, geometry = prepare_vae_input(Image.new("RGB", (16, 16)))
    with pytest.raises(ValueError, match="spatial"):
        restore_vae_output(tensor[:, :, :8], geometry)
    tensor[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        restore_vae_output(tensor, geometry)


def test_identity_reconstruction_has_explicit_infinite_psnr_and_ssim_one() -> None:
    pixels = np.zeros((32, 32, 3))
    result = reconstruction_metrics(pixels, pixels)
    assert result["mse_rgb_0_1"] == 0
    assert result["psnr_db"] is None
    assert result["psnr_is_infinite"]
    assert result["ssim_rgb"] == pytest.approx(1)


def test_constant_offset_psnr_has_correct_units() -> None:
    pixels = np.zeros((32, 32, 3))
    result = reconstruction_metrics(pixels, pixels + 0.1)
    assert result["mse_rgb_0_1"] == pytest.approx(0.01)
    assert result["psnr_db"] == pytest.approx(20)
    assert not result["psnr_is_infinite"]
    assert 0 <= result["ssim_rgb"] < 1


@pytest.mark.parametrize("defect", ["shape", "nan", "range"])
def test_invalid_reconstruction_arrays_are_rejected(defect: str) -> None:
    original = np.zeros((32, 32, 3))
    changed = original.copy()
    if defect == "shape":
        changed = changed[:16]
    elif defect == "nan":
        changed[0, 0, 0] = np.nan
    else:
        changed[0, 0, 0] = 255
    with pytest.raises(ValueError):
        reconstruction_metrics(original, changed)


def test_failed_vae_units_remain_terminal_and_provenance_bound(tmp_path: Path) -> None:
    path = tmp_path / "failed.json"
    record = {
        "sample_id": "a",
        "source_sha256": "a" * 64,
        "config_sha256": "c" * 64,
        "status": "failed",
        "error": "recorded failure",
    }
    dump(path, record)
    assert _read_terminal_unit(path, "a" * 64, "c" * 64, tmp_path) == record
    with pytest.raises(ValueError, match="provenance"):
        _read_terminal_unit(path, "b" * 64, "c" * 64, tmp_path)


def test_completed_vae_units_verify_output_file_hashes(tmp_path: Path) -> None:
    from id_layers.auto_research import digest

    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(image)
    record = {
        "sample_id": "a",
        "source_sha256": "a" * 64,
        "config_sha256": "c" * 64,
        "status": "complete",
        "control_relative_path": image.name,
        "control_sha256": digest(image),
        "reconstruction_relative_path": image.name,
        "reconstruction_sha256": digest(image),
    }
    path = tmp_path / "unit.json"
    dump(path, record)
    assert _read_terminal_unit(path, "a" * 64, "c" * 64, tmp_path)["status"] == "complete"
    Image.new("RGB", (16, 16), "white").save(image)
    with pytest.raises(ValueError, match="output hash"):
        _read_terminal_unit(path, "a" * 64, "c" * 64, tmp_path)
