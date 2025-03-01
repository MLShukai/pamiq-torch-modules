import torch.nn as nn
from torch import Tensor
from typing_extensions import override


class ResNetFF(nn.Module):
    """Residual network feed-forward module.

    A feed-forward network with residual connections that processes
    input tensors.
    """

    @override
    def __init__(
        self, dim: int, dim_hidden: int, depth: int, activation: nn.Module = nn.GELU()
    ) -> None:
        """Initialize the ResNetFF module.

        Args:
            dim: Input and output dimension
            dim_hidden: Hidden layer dimension
            depth: Number of feed-forward blocks
            activation: Activation function to use
        """
        super().__init__()  # type: ignore
        self.ff_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim_hidden),
                    activation,
                    nn.Linear(dim_hidden, dim),
                )
                for _ in range(depth)
            ]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Process input through residual feed-forward blocks.

        Args:
            x: Input tensor

        Returns:
            Processed tensor with residual connections
        """
        for ff in self.ff_list:
            x_ = x
            x = ff(x)
            x = x + x_
        return x
