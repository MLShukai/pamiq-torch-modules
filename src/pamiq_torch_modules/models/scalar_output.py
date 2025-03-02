import torch.nn as nn
from torch import Tensor
from typing_extensions import override


class ScalarOutput(nn.Module):
    """Neural network head that outputs a scalar value.

    This module applies a linear transformation to convert input
    features into a single scalar value per sample.
    """

    def __init__(self, dim_in: int, squeeze_output: bool = False) -> None:
        """Initialize the ScalarOutput module.

        Args:
            dim_in: Input dimension size of tensor
            squeeze_output: Whether to squeeze the output dimension
        """
        super().__init__()  # type: ignore
        self.fc = nn.Linear(dim_in, 1)
        self.squeeze_output = squeeze_output

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Process input through the scalar output layer.

        Args:
            x: Input tensor of shape (*, dim_in)

        Returns:
            Output tensor of shape (*, 1) or (*) if squeeze_output is True
        """
        out: Tensor = self.fc(x)
        if self.squeeze_output:
            out = out.squeeze(-1)
        return out
