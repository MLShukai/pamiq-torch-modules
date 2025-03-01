import pytest
import torch
import torch.nn as nn

from pamiq_torch_modules.models.resnet import ResNetFF
from tests.helpers import parametrize_device


class TestResNetFF:
    """Test suite for ResNetFF module."""

    @pytest.mark.parametrize(
        "dim, dim_hidden, depth, activation",
        [
            (32, 64, 2, nn.GELU()),
            (32, 128, 3, nn.ReLU()),
            (32, 64, 1, nn.Tanh()),
        ],
    )
    def test_initialization(
        self, dim: int, dim_hidden: int, depth: int, activation: nn.Module
    ) -> None:
        """Test the initialization of ResNetFF.

        Args:
            dim: Input and output dimension
            dim_hidden: Hidden layer dimension
            depth: Number of feed-forward blocks
            activation: Activation function to use
        """
        model = ResNetFF(dim, dim_hidden, depth, activation)

        # Check if the model has correct number of feed-forward blocks
        assert len(model.ff_list) == depth

    @parametrize_device
    def test_forward(self, device: torch.device) -> None:
        """Test if the forward pass preserves the input shape.

        Args:
            input_tensor: Sample input tensor
        """
        input_tensor = torch.randn(2, 32, device=device)
        model = ResNetFF(dim=32, dim_hidden=64, depth=2).to(device)

        output = model(input_tensor)

        # Check if output has the same shape as input
        assert output.shape == input_tensor.shape
        # Check if output dtype matches input dtype
        assert output.dtype == input_tensor.dtype
        # Check if output device matches input device
        assert output.device == input_tensor.device
