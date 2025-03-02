from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import OneHotCategoricalStraightThrough
from typing_extensions import override


class MaskedOneHotCategoricalStraightThrough(OneHotCategoricalStraightThrough):
    """Masked OneHotCategoricalStraightThrough class.

    This class extends OneHotCategoricalStraightThrough by adding
    functionality to apply a mask, disabling specific choices.
    """

    def __init__(self, logits: Tensor, mask: Tensor, validate_args: Any = None) -> None:
        """Initialize the MaskedOneHotCategoricalStraightThrough.

        Args:
            logits: Tensor of logits.
            mask: Mask tensor. True elements indicate invalid choices.
            validate_args: Arguments for distribution validation.
        """
        assert logits.shape[-mask.ndim :] == mask.shape
        logits = logits.masked_fill(mask, -torch.inf)
        self.mask = mask
        self.logits: Tensor
        self.probs: Tensor
        super().__init__(logits=logits, validate_args=validate_args)  # type: ignore

    @override
    def entropy(self) -> Tensor:
        """Calculate the entropy of the distribution.

        Returns:
            Entropy tensor.
        """
        log_prob: Tensor = torch.log_softmax(self.logits, dim=-1)
        log_prob = log_prob.masked_fill(self.mask, 0.0)
        return -torch.sum(self.probs * log_prob, dim=-1)


class MultiOneHots(nn.Module):
    """Class for handling multiple OneHot distributions.

    This class generates OneHot distributions for multiple categories.
    Each category can have a different number of choices.
    """

    def __init__(self, in_features: int, choices_per_category: list[int]) -> None:
        """Initialize the MultiOneHots module.

        Args:
            in_features: Number of input features.
            choices_per_category: List of number of choices for each category.
        """
        super().__init__()  # type: ignore

        out_features = max(choices_per_category)
        num_categories = len(choices_per_category)
        mask = torch.zeros((num_categories, out_features), dtype=torch.bool)
        for i, c in enumerate(choices_per_category):
            assert c > 0, f"Category index {i} has no choices!"
            mask[i, c:] = True

        self.mask: Tensor
        self.register_buffer("mask", mask)

        self._layers = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=False)
            for _ in choices_per_category
        )

    @override
    def forward(self, x: Tensor) -> MaskedOneHotCategoricalStraightThrough:
        """Process input through the multiple OneHot layers.

        Args:
            x: Input tensor.

        Returns:
            A MaskedOneHotCategoricalStraightThrough distribution.

        Shapes:
            x: (*, in_features)
            return: (*, num_categories, max_choice)
        """
        out = torch.stack([lyr(x) for lyr in self._layers], dim=-2)
        return MaskedOneHotCategoricalStraightThrough(logits=out, mask=self.mask)


class OneHotToEmbedding(nn.Module):
    """Make the embedding from one hot vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        """Initialize the OneHotToEmbedding module.

        Args:
            num_embeddings: Number of embeddings.
            embedding_dim: Dimension of the embedding vectors.
        """
        super().__init__()  # type: ignore
        self._weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim), True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Convert one-hot vectors to embeddings.

        Args:
            x: One-hot input tensor.

        Returns:
            Embedded tensor.

        Shapes:
            x: (*, num_embeddings)
            return: (*, embedding_dim)
        """
        return x @ self._weight
