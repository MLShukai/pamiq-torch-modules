import pytest
import torch

from pamiq_torch_modules.models.scalar_output import ScalarOutput
from tests.helpers import parametrize_device


class TestScalarOutput:
    """Test suite for ScalarOutput module."""

    @parametrize_device
    @pytest.mark.parametrize(
        "dim_in, batch_size, squeeze_output",
        [(16, 8, False), (32, 1, False), (64, 4, True)],
    )
    def test_forward(
        self, dim_in: int, batch_size: int, squeeze_output: bool, device: torch.device
    ) -> None:
        """Test if the forward pass produces correctly shaped outputs.

        Args:
            dim_in: Input dimension
            batch_size: Batch size
            squeeze_output: Whether to squeeze the output dimension
            device: Device to run the test on
        """
        model = ScalarOutput(dim_in, squeeze_output).to(device)
        x = torch.randn(batch_size, dim_in, device=device)

        output = model(x)

        # Check output shape
        expected_shape = (batch_size,) if squeeze_output else (batch_size, 1)
        assert output.shape == expected_shape

        # Check device
        assert output.device == device

        # Check dtype
        assert output.dtype == x.dtype
