import pytest
import torch

from pamiq_torch_modules.models.one_hot import (
    MaskedOneHotCategoricalStraightThrough,
    MultiOneHots,
    OneHotToEmbedding,
)
from tests.helpers import MPS_DEVICE, parametrize_device


class TestMaskedOneHotCategoricalStraightThrough:
    """Test suite for MaskedOneHotCategoricalStraightThrough class."""

    @parametrize_device
    @pytest.mark.parametrize("batch_size", [1, 32])
    @pytest.mark.parametrize("num_categories", [5])
    @pytest.mark.parametrize("num_choices", [3])
    def test_initialization(self, batch_size, num_categories, num_choices, device):
        """Test initialization of MaskedOneHotCategoricalStraightThrough."""
        if device == MPS_DEVICE:
            pytest.skip("Some host occurs segmentation fault.")
        logits = torch.randn(batch_size, num_categories, num_choices, device=device)
        mask = torch.zeros(num_categories, num_choices, dtype=torch.bool, device=device)
        mask[:, num_choices // 2 :] = True

        distribution = MaskedOneHotCategoricalStraightThrough(logits=logits, mask=mask)

        assert distribution.logits.shape == (batch_size, num_categories, num_choices)
        assert torch.all(distribution.logits[..., mask] == -torch.inf)
        assert distribution.probs.shape == (batch_size, num_categories, num_choices)
        assert torch.all(distribution.probs[..., mask] == 0)
        assert distribution.logits.device == device
        assert distribution.probs.device == device

    @parametrize_device
    def test_entropy(self, device):
        """Test entropy calculation."""
        if device == MPS_DEVICE:
            pytest.skip("Some host occurs segmentation fault.")
        logits = torch.randn(1, 3, 4, device=device)
        # fmt: off
        mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, True],
                [False, True, True, True],
            ],
            device=device
        )
        # fmt: on

        distribution = MaskedOneHotCategoricalStraightThrough(logits=logits, mask=mask)
        entropy = distribution.entropy()

        assert entropy.shape == (1, 3)
        assert torch.all(entropy >= 0)  # Entropy should be non-negative
        assert torch.all(torch.isfinite(entropy))
        assert entropy.device == device

    @parametrize_device
    def test_log_prob(self, device):
        """Test log probability calculation."""
        logits = torch.randn(1, 3, 4, device=device)
        # fmt: off
        mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, True],
                [False, True, True, True],
            ],
            device=device
        )
        # fmt: on

        distribution = MaskedOneHotCategoricalStraightThrough(logits=logits, mask=mask)

        # Test with valid one-hot vectors
        # fmt: off
        valid_sample = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                ]
            ],
            device=device
        )
        # fmt: on
        log_prob = distribution.log_prob(valid_sample)

        assert log_prob.shape == (1, 3)
        assert torch.all(torch.isfinite(log_prob))
        assert log_prob.device == device

        # Test with invalid one-hot vectors (selecting masked values)
        # fmt: off
        invalid_sample = torch.tensor(
            [
                [
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                ]
            ],
            device=device
        )
        # fmt: on
        log_prob_invalid = distribution.log_prob(invalid_sample)

        assert torch.all(log_prob_invalid == -torch.inf)
        assert log_prob_invalid.device == device


class TestMultiOneHots:
    """Test suite for MultiOneHots class."""

    @parametrize_device
    @pytest.mark.parametrize("in_features", [10])
    @pytest.mark.parametrize("choices_per_category", [[2, 3, 4], [3, 3, 3, 2, 2]])
    @pytest.mark.parametrize("batch_size", [1, 32])
    def test_multi_one_hots(
        self, in_features, choices_per_category, batch_size, device
    ):
        if device == MPS_DEVICE:
            pytest.skip("Some host occurs segmentation fault.")
        """Test MultiOneHots functionality."""
        # Create MultiOneHots instance
        multi_one_hots = MultiOneHots(in_features, choices_per_category).to(device)

        # Create input tensor
        x = torch.randn(batch_size, in_features, device=device)

        # Forward pass
        output = multi_one_hots(x)

        # Check output type
        assert isinstance(output, MaskedOneHotCategoricalStraightThrough)

        # Check output shapes
        num_categories = len(choices_per_category)
        max_choices = max(choices_per_category)
        assert output.logits.shape == (batch_size, num_categories, max_choices)

        # Check that logits for invalid choices are set to -inf
        # Check that probs for invalid choices are set to 0
        for i, choices in enumerate(choices_per_category):
            assert torch.all(output.logits[:, i, choices:] == -torch.inf)
            assert torch.all(output.probs[:, i, choices:] == 0)

        # Check device
        assert output.logits.device == device
        assert output.probs.device == device

        # Test sampling
        sample = output.sample()
        assert sample.shape == (batch_size, num_categories, max_choices)
        assert torch.all(sample.sum(dim=-1) == 1)  # One-hot property
        assert sample.device == device

        # Test sample gradient
        rsample = output.rsample()
        assert rsample.requires_grad is True
        assert rsample.device == device

        sample = output.sample()
        assert sample.requires_grad is False
        assert sample.device == device

    def test_multi_one_hots_invalid_args(self):
        """Test MultiOneHots with invalid arguments."""
        with pytest.raises(AssertionError):
            MultiOneHots(10, [0, 1, 2])  # Category with no choices


class TestOneHotToEmbedding:
    """Test suite for OneHotToEmbedding class."""

    @parametrize_device
    @pytest.mark.parametrize("num_embeddings", [10])
    @pytest.mark.parametrize("embedding_dim", [5])
    @pytest.mark.parametrize("batch_size", [1, 32])
    def test_one_hot_to_embedding(
        self, num_embeddings, embedding_dim, batch_size, device
    ):
        """Test OneHotToEmbedding functionality."""
        if device == MPS_DEVICE:
            pytest.skip("Some host occurs segmentation fault.")
        # Create OneHotEmbedding instance
        one_hot_embedding = OneHotToEmbedding(num_embeddings, embedding_dim).to(device)

        # Create input tensor (one-hot vectors)
        x = torch.eye(num_embeddings, device=device).repeat(batch_size, 1, 1)

        # Forward pass
        output = one_hot_embedding(x)

        # Check output shape
        assert output.shape == (batch_size, num_embeddings, embedding_dim)
        assert output.device == device

        # Check that the embedding for each one-hot vector is correct
        for i in range(num_embeddings):
            assert torch.allclose(output[:, i], one_hot_embedding._weight[i])  # type: ignore

    @parametrize_device
    def test_one_hot_to_embedding_gradients(self, device):
        """Test gradients for OneHotToEmbedding."""
        if device == MPS_DEVICE:
            pytest.skip("Some host occurs segmentation fault.")
        num_embeddings, embedding_dim = 10, 5
        one_hot_embedding = OneHotToEmbedding(num_embeddings, embedding_dim).to(device)
        x = torch.eye(num_embeddings, requires_grad=True, device=device)

        output = one_hot_embedding(x)
        output.sum().backward()

        assert one_hot_embedding._weight.grad is not None  # type: ignore
        assert x.grad is not None
        assert one_hot_embedding._weight.grad.device == device  # type: ignore
        assert x.grad.device == device
