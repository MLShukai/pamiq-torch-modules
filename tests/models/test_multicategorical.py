import pytest
import torch
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution

from pamiq_torch_modules.models.multicategorical import (
    Multicategorical,
    MulticategoricalOutput,
)
from tests.helpers import parametrize_device


class TestMulticategorical:
    """Test suite for MultiCategoricals distribution."""

    @pytest.fixture
    def distributions(self) -> list[Categorical]:
        """Create sample distributions for testing.

        Returns:
            List of categorical distributions
        """
        choices_per_dist = [3, 2, 5]
        batch_size = 8
        return [
            Categorical(logits=torch.zeros(batch_size, c)) for c in choices_per_dist
        ]

    @pytest.fixture
    def multi_categoricals(self, distributions) -> Multicategorical:
        """Create MultiCategoricals instance for testing.

        Args:
            distributions: List of categorical distributions

        Returns:
            MultiCategoricals instance
        """
        return Multicategorical(distributions)

    def test_init(self, multi_categoricals: Multicategorical) -> None:
        """Test initialization of MultiCategoricals."""
        assert multi_categoricals.batch_shape == (8, 3)

    def test_sample(self, multi_categoricals: Multicategorical) -> None:
        """Test sampling from MultiCategoricals."""
        assert multi_categoricals.sample().shape == (8, 3)
        assert multi_categoricals.sample((1, 2)).shape == (1, 2, 8, 3)

    def test_log_prob(self, multi_categoricals: Multicategorical) -> None:
        """Test log probability calculation."""
        sampled = multi_categoricals.sample()
        assert multi_categoricals.log_prob(sampled).shape == sampled.shape

    def test_entropy(self, multi_categoricals: Multicategorical) -> None:
        """Test entropy calculation."""
        assert multi_categoricals.entropy().shape == (8, 3)


class TestMulticategoricalOutput:
    """Test suite for MulticategoricalOutput module."""

    @parametrize_device
    @pytest.mark.parametrize(
        "batch_size, dim_in, action_choices_per_category",
        [
            (8, 256, [3, 3, 3, 2, 2]),
            (1, 16, [1, 2, 3, 4, 5]),
        ],
    )
    def test_forward(
        self,
        batch_size: int,
        dim_in: int,
        action_choices_per_category: list[int],
        device: torch.device,
    ) -> None:
        """Test if the forward pass produces correct distributions.

        Args:
            batch_size: Batch size for test input
            dim_in: Input dimension
            action_choices_per_category: List of action choice counts
            device: Device to run the test on
        """
        model = MulticategoricalOutput(dim_in, action_choices_per_category).to(device)
        x = torch.randn(batch_size, dim_in, device=device)

        dist = model(x)

        # Check basic properties
        assert isinstance(dist, Distribution)

        # Check shapes
        sampled = dist.sample()
        assert sampled.shape == (batch_size, len(action_choices_per_category))
        assert sampled.device == device

        log_probs = dist.log_prob(sampled)
        assert log_probs.shape == (batch_size, len(action_choices_per_category))
        assert log_probs.device == device

        entropy = dist.entropy()
        assert entropy.shape == (batch_size, len(action_choices_per_category))
        assert entropy.device == device
