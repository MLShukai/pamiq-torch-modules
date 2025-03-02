import pytest
import torch
from torch.distributions import Normal

from pamiq_torch_modules.models.fc_normal import (
    SCALE_ONE,
    SHIFT_ZERO,
    DeterministicNormal,
    FCFixedStdNormal,
    FCNormal,
)
from tests.helpers import parametrize_device


class TestDeterministicNormal:
    """Test suite for DeterministicNormal class."""

    @parametrize_device
    def test_sample(self, device: torch.device) -> None:
        """Test that sample methods always return mean.

        Args:
            device: Device to run the test on.
        """
        mean = torch.randn(10, device=device)
        std = torch.ones(10, device=device)
        dn = DeterministicNormal(mean, std)

        # Check that sample() and rsample() both return mean
        assert torch.equal(dn.sample(), mean)
        assert torch.equal(dn.sample(), mean)
        assert torch.equal(dn.rsample(), mean)
        assert torch.equal(dn.rsample(), mean)

        # Check with sample shape
        samples = dn.sample((5,))
        assert samples.shape == (5, 10)
        assert torch.allclose(samples, mean.expand(5, 10))


class TestFCFixedStdNormal:
    """Test suite for FCFixedStdNormal module."""

    @parametrize_device
    def test_forward(self, device: torch.device) -> None:
        """Test forward pass returns correct Normal distribution.

        Args:
            device: Device to run the test on.
        """
        layer = FCFixedStdNormal(10, 20).to(device)
        x = torch.randn(10, device=device)
        out = layer(x)

        # Check output type and properties
        assert isinstance(out, Normal)
        assert out.sample().shape == (20,)
        assert out.loc.device == device
        assert out.scale.device == device

        # Check batch handling
        batched_x = torch.randn(3, 4, 10, device=device)
        batched_out = layer(batched_x)
        assert batched_out.sample().shape == (3, 4, 20)

    @parametrize_device
    def test_squeeze_feature_dim(self, device: torch.device) -> None:
        """Test squeezing feature dimension functionality.

        Args:
            device: Device to run the test on.
        """
        with pytest.raises(AssertionError):
            # dim_out must be 1 when squeeze_feature_dim is True
            FCFixedStdNormal(10, 2, squeeze_feature_dim=True)

        # When squeeze_feature_dim is True with dim_out=1
        net = FCFixedStdNormal(10, 1, squeeze_feature_dim=True).to(device)
        x = torch.randn(10, device=device)
        out = net(x)
        assert out.sample().shape == ()

        # With batch dimension
        batched_x = torch.randn(3, 10, device=device)
        batched_out = net(batched_x)
        assert batched_out.sample().shape == (3,)

    @parametrize_device
    def test_deterministic_normal(self, device: torch.device) -> None:
        """Test using DeterministicNormal as distribution class.

        Args:
            device: Device to run the test on.
        """
        layer = FCFixedStdNormal(10, 5, normal_cls="Deterministic").to(device)
        x = torch.randn(10, device=device)
        out = layer(x)

        # Check that multiple samples are identical
        sample1 = out.sample()
        sample2 = out.sample()
        assert torch.equal(sample1, sample2)
        assert torch.equal(sample1, out.loc)

    def test_logprob_constants(self) -> None:
        """Test the mathematical properties of the constant values."""
        # Test SHIFT_ZERO: log_prob should be zero at mean
        mean = torch.zeros(10)
        std = torch.full_like(mean, SHIFT_ZERO)
        normal = Normal(mean, std)
        expected = torch.zeros_like(mean)
        torch.testing.assert_close(normal.log_prob(mean), expected)

        # Test SCALE_ONE: difference in negative log prob should equal
        # difference in squared error
        mean = torch.zeros(3)
        std = torch.full_like(mean, SCALE_ONE)
        normal = Normal(mean, std)

        t1 = torch.full_like(mean, 1)
        t2 = torch.full_like(mean, 3)

        nlp1 = -normal.log_prob(t1)
        nlp2 = -normal.log_prob(t2)

        expected = (t1 - mean) ** 2 - (t2 - mean) ** 2
        actual = nlp1 - nlp2

        torch.testing.assert_close(actual, expected)


class TestFCNormal:
    """Test suite for FCNormal module."""

    @parametrize_device
    @pytest.mark.parametrize(
        ["dim_in", "dim_out", "batch_shape"],
        [
            (8, 16, ()),  # No batch
            (8, 16, (3,)),  # 1D batch
            (8, 16, (2, 3)),  # 2D batch
        ],
    )
    def test_forward(
        self, dim_in: int, dim_out: int, batch_shape: tuple, device: torch.device
    ) -> None:
        """Test forward pass returns correct Normal distribution.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            batch_shape: Shape of batch dimensions.
            device: Device to run the test on.
        """
        model = FCNormal(dim_in, dim_out).to(device)
        x_shape = batch_shape + (dim_in,)
        x = torch.randn(*x_shape, device=device)
        out = model(x)

        # Check output type and properties
        assert isinstance(out, Normal)
        expected_output_shape = batch_shape + (dim_out,)
        assert out.sample().shape == expected_output_shape
        assert out.loc.device == device
        assert out.scale.device == device

        # Initial std should be near 1.0 due to initialization
        assert torch.allclose(out.scale, torch.ones_like(out.scale), atol=0.1)

    @parametrize_device
    def test_squeeze_feature_dim(self, device: torch.device) -> None:
        """Test squeezing feature dimension functionality.

        Args:
            device: Device to run the test on.
        """
        with pytest.raises(AssertionError):
            # dim_out must be 1 when squeeze_feature_dim is True
            FCNormal(10, 2, squeeze_feature_dim=True)

        # When squeeze_feature_dim is True with dim_out=1
        net = FCNormal(10, 1, squeeze_feature_dim=True).to(device)
        x = torch.randn(10, device=device)
        out = net(x)
        assert out.sample().shape == ()

        # With batch dimension
        batched_x = torch.randn(3, 10, device=device)
        batched_out = net(batched_x)
        assert batched_out.sample().shape == (3,)
