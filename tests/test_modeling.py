import numpy as np
import pytest
import torch
from torch.nn import functional as F

from id_layers.modeling import extract_layer_embeddings, non_affine_ln_token_mean


class _AddBlock(torch.nn.Module):
    def __init__(self, offset: list[float]) -> None:
        super().__init__()
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float32))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.offset


class _ToyTokenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [_AddBlock([0.2, -0.1, 0.4]), _AddBlock([-0.3, 0.5, 0.1])]
        )
        self.norm = torch.nn.LayerNorm(3)
        self.feature = torch.nn.Linear(12, 2, bias=False)
        with torch.no_grad():
            self.norm.weight.copy_(torch.tensor([2.0, 0.5, 3.0]))
            self.norm.bias.copy_(torch.tensor([0.7, -0.2, 0.4]))
            self.feature.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(2, 12))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = images.reshape(images.shape[0], 3, 4).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        return self.feature(self.norm(tokens).flatten(1))


def test_cached_non_affine_ln_mean_matches_direct_affine_layer_norm() -> None:
    generator = torch.Generator().manual_seed(776)
    tokens = torch.randn(3, 7, 5, generator=generator, dtype=torch.float64)
    weight = torch.randn(5, generator=generator, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(5, generator=generator, dtype=torch.float64, requires_grad=True)

    direct = F.layer_norm(tokens, (5,), weight=weight, bias=bias, eps=1e-5).mean(dim=1)
    cached = non_affine_ln_token_mean(tokens, eps=1e-5) * weight + bias
    torch.testing.assert_close(cached, direct, rtol=1e-12, atol=1e-12)

    direct_gradients = torch.autograd.grad(direct.square().sum(), (weight, bias))
    cached_gradients = torch.autograd.grad(cached.square().sum(), (weight, bias))
    for cached_gradient, direct_gradient in zip(cached_gradients, direct_gradients, strict=True):
        torch.testing.assert_close(cached_gradient, direct_gradient, rtol=1e-11, atol=1e-11)


def test_non_affine_ln_token_mean_validates_shape_and_epsilon() -> None:
    with pytest.raises(ValueError, match="shape"):
        non_affine_ln_token_mean(torch.ones(2, 3))
    with pytest.raises(ValueError, match="epsilon"):
        non_affine_ln_token_mean(torch.ones(2, 3, 4), eps=0.0)


def test_probe_representation_uses_raw_post_block_tokens_without_l2() -> None:
    model = _ToyTokenModel()
    images = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 10.0
    result = extract_layer_embeddings(
        model=model,
        loader=[{"image": images}],
        device_name="cpu",
        representations=["probe_ln0_mean"],
        image_size=2,
    )

    first_layer_tokens = model.blocks[0](images.reshape(2, 3, 4).transpose(1, 2))
    expected = non_affine_ln_token_mean(first_layer_tokens).numpy()
    np.testing.assert_allclose(
        result.embeddings[(1, "probe_ln0_mean")], expected, rtol=1e-6, atol=1e-6
    )
    assert result.metadata["final_head_max_abs_difference"] is None
    assert result.metadata["probe_layer_norm_eps"] == 1e-5


def test_extraction_returns_only_requested_layers_and_records_metadata() -> None:
    model = _ToyTokenModel()
    images = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 10.0
    result = extract_layer_embeddings(
        model=model,
        loader=[{"image": images}],
        device_name="cpu",
        representations=["probe_ln0_mean", "head_projected"],
        image_size=2,
        layers=[2],
    )

    assert set(result.embeddings) == {
        (2, "probe_ln0_mean"),
        (2, "head_projected"),
    }
    assert result.embeddings[(2, "probe_ln0_mean")].shape == (2, 3)
    assert result.embeddings[(2, "head_projected")].shape == (2, 2)
    assert result.metadata["sample_count"] == 2
    assert result.metadata["sample_order_verified"] is False
    assert result.metadata["layer_count"] == 2
    assert result.metadata["extracted_layers"] == [2]
    assert result.metadata["representations"] == ["probe_ln0_mean", "head_projected"]
    assert result.metadata["final_head_max_abs_difference"] == 0.0
    assert result.metadata["probe_layer_norm_eps"] == 1e-5


@pytest.mark.parametrize("layers", [[], [0], [3], [-1, 1]])
def test_extraction_rejects_empty_or_out_of_range_layers(layers: list[int]) -> None:
    model = _ToyTokenModel()
    images = torch.zeros(1, 3, 2, 2)
    with pytest.raises(ValueError, match="layer|Layer"):
        extract_layer_embeddings(
            model=model,
            loader=[{"image": images}],
            device_name="cpu",
            representations=["probe_ln0_mean"],
            image_size=2,
            layers=layers,
        )


def test_head_parity_is_unavailable_when_final_layer_is_not_extracted() -> None:
    model = _ToyTokenModel()
    images = torch.zeros(1, 3, 2, 2)
    result = extract_layer_embeddings(
        model=model,
        loader=[{"image": images}],
        device_name="cpu",
        representations=["head_projected"],
        image_size=2,
        layers=[1],
    )
    assert result.metadata["final_head_max_abs_difference"] is None


def test_extraction_verifies_manifest_order_when_loader_provides_indices() -> None:
    model = _ToyTokenModel()
    images = torch.zeros(2, 3, 2, 2)
    result = extract_layer_embeddings(
        model=model,
        loader=[{"image": images, "index": torch.tensor([0, 1])}],
        device_name="cpu",
        representations=["probe_ln0_mean"],
        image_size=2,
    )
    assert result.metadata["sample_order_verified"] is True

    with pytest.raises(RuntimeError, match="sample order"):
        extract_layer_embeddings(
            model=model,
            loader=[{"image": images, "index": torch.tensor([1, 0])}],
            device_name="cpu",
            representations=["probe_ln0_mean"],
            image_size=2,
        )
