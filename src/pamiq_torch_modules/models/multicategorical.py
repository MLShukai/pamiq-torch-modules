import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Distribution
from typing_extensions import override

from ._types import SizeType

__all__ = ["Multicategorical", "MulticategoricalOutput"]


class Multicategorical(Distribution):
    """Set of categorical distributions for multiple action dimensions.

    This distribution represents multiple independent categorical
    distributions, each potentially having a different number of
    categories.
    """

    def __init__(self, distributions: list[Categorical]) -> None:
        """Initialize the MultiCategoricals distribution.

        Args:
            distributions: A list of Categorical distributions, where each
                           distribution may have a different size of action choices
        """
        assert len(distributions) > 0, "At least one distribution is required"
        first_dist = distributions[0]
        assert all(
            first_dist.batch_shape == d.batch_shape for d in distributions
        ), "All batch shapes must be same"
        batch_shape = torch.Size((*first_dist.batch_shape, len(distributions)))
        super().__init__(
            batch_shape=batch_shape, event_shape=torch.Size(), validate_args=False
        )  # type: ignore

        self.dists = distributions

    @override
    def sample(self, sample_shape: SizeType = torch.Size()) -> Tensor:
        """Sample from each distribution and stack the outputs.

        Args:
            sample_shape: Shape of the samples

        Returns:
            Stacked samples from all distributions

        Shapes:
            return: (*sample_shape, *batch_shape)
        """
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """Compute log probability of given values.

        Args:
            value: Tensor of action values

        Returns:
            Log probabilities for each action value

        Shapes:
            value: (*, num_dists)
            return: (*, num_dists)
        """
        return torch.stack(
            [d.log_prob(v) for d, v in zip(self.dists, value.movedim(-1, 0))], dim=-1
        )

    @override
    def entropy(self) -> Tensor:
        """Compute entropy for each distribution.

        Returns:
            Entropy values for each distribution

        Shapes:
            return: (*batch_shape)
        """
        return torch.stack([d.entropy() for d in self.dists], dim=-1)


class MulticategoricalOutput(nn.Module):
    """Neural network head for outputting distributions over multiple
    categorical actions.

    This module generates independent categorical distributions for each
    action dimension, where each dimension can have a different number
    of possible values.
    """

    def __init__(self, dim_in: int, action_choices_per_category: list[int]) -> None:
        """Initialize the MulticategoricalOutput module.

        Args:
            dim_in: Input dimension size of tensor
            action_choices_per_category: List of action choice count per category
        """
        super().__init__()  # type: ignore

        self.heads = nn.ModuleList()
        for choice in action_choices_per_category:
            self.heads.append(nn.Linear(dim_in, choice, bias=False))

    @override
    def forward(self, x: Tensor) -> Multicategorical:
        """Process input through the policy heads to generate action
        distributions.

        Args:
            x: Input tensor of shape (*, dim_in)

        Returns:
            Distribution object representing multiple categorical distributions
        """
        categoricals: list[Categorical] = []
        for head in self.heads:
            logits = head(x)
            categoricals.append(Categorical(logits=logits))

        return Multicategorical(categoricals)
