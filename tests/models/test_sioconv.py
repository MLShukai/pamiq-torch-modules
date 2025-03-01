import pytest
import torch

from pamiq_torch_modules.models.sioconv import (
    RMSNorm,
    SioConvPS,
    SioConvPSBlock,
    SioConvPSLayer,
    SwiGLU,
)
from tests.helpers import MPS_DEVICE, parametrize_device


class TestRMSNorm:
    """Test suite for RMSNorm module."""

    @parametrize_device
    def test_forward(self, device: torch.device) -> None:
        """Test if RMSNorm works correctly.

        Args:
            device: Device to run the test on
        """
        dim = 64
        batch_size = 2
        seq_len = 10

        # With affine parameters
        norm = RMSNorm(dim, elementwise_affine=True).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        out = norm(x)

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # Check normalization property - mean of squares should be close to 1
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-2)

        # Without affine parameters
        norm = RMSNorm(dim, elementwise_affine=False).to(device)
        out = norm(x)

        assert out.shape == x.shape


class TestSwiGLU:
    """Test suite for SwiGLU module."""

    @parametrize_device
    def test_forward(self, device: torch.device) -> None:
        """Test if SwiGLU works correctly.

        Args:
            device: Device to run the test on
        """
        dim = 64
        dim_ff = 128
        batch_size = 2
        seq_len = 10

        model = SwiGLU(dim, dim_ff).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        out = model(x)

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device


class TestSioConvPSLayer:
    """Test suite for SioConvPSLayer module."""

    @parametrize_device
    def test_forward(self, device: torch.device) -> None:
        """Test if SioConvPSLayer works correctly.

        Args:
            device: Device to run the test on
        """
        if device == MPS_DEVICE:
            pytest.skip(
                "Some operators are not currently implemented for the MPS device."
            )
        dim = 64
        batch_size = 2
        seq_len = 10

        model = SioConvPSLayer(dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        hidden = torch.randn(batch_size, dim, device=device)

        out, new_hidden = model(x, hidden)

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        if seq_len > 1:
            assert new_hidden.shape == (batch_size, seq_len, dim)
        else:
            assert new_hidden.shape == (batch_size, dim)


class TestSioConvPSBlock:
    """Test suite for SioConvPSBlock module."""

    @parametrize_device
    def test_forward(self, device: torch.device) -> None:
        """Test if SioConvPSBlock works correctly.

        Args:
            device: Device to run the test on
        """
        if device == MPS_DEVICE:
            pytest.skip(
                "Some operators are not currently implemented for the MPS device."
            )
        dim = 64
        dim_ff = 128
        batch_size = 2
        seq_len = 10

        model = SioConvPSBlock(dim, dim_ff).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        hidden = torch.randn(batch_size, dim, device=device)

        out, _ = model(x, hidden)

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device


class TestSioConvPS:
    """Test suite for SioConvPS module."""

    @parametrize_device
    @pytest.mark.parametrize(
        "shape, hidden_shape, expected_hidden_shape",
        [
            (
                (2, 10, 64),
                (2, 3, 64),
                (2, 3, 10, 64),
            ),  # Batched sequence, batched hidden
            ((2, 64), (2, 3, 64), (2, 3, 64)),  # Batched single token, batched hidden
            ((10, 64), (3, 64), (3, 10, 64)),  # Unbatched sequence, unbatched hidden
            ((64,), (3, 64), (3, 64)),  # Single token, unbatched hidden
        ],
    )
    def test_forward(
        self,
        shape: tuple,
        hidden_shape: tuple,
        expected_hidden_shape: tuple,
        device: torch.device,
    ) -> None:
        """Test if SioConvPS works correctly with various input shapes.

        Args:
            shape: Input tensor shape
            hidden_shape: Hidden state tensor shape
            expected_hidden_shape: Expected output hidden state shape
            device: Device to run the test on
        """
        if device == MPS_DEVICE:
            pytest.skip(
                "Some operators are not currently implemented for the MPS device."
            )

        depth = hidden_shape[-2]  # Extract depth from hidden_shape
        dim = shape[-1]
        dim_ff = 128

        model = SioConvPS(depth, dim, dim_ff).to(device)
        x = torch.randn(*shape, device=device)
        hidden = torch.randn(*hidden_shape, device=device)

        out, new_hidden = model(x, hidden)

        # Check output shape matches input shape
        assert out.shape == shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # Check hidden state shape matches expected shape
        assert new_hidden.shape == expected_hidden_shape
