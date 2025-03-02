import math
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Normal
from typing_extensions import override

_size: TypeAlias = Size | list[int] | tuple[int, ...]

# Constants for FixedStdNormal
SHIFT_ZERO = 1.0 / math.sqrt(2.0 * math.pi)
SCALE_ONE = 1.0 / math.sqrt(2.0)

__all__ = [
    "FCNormal",
    "FCFixedStdNormal",
    "DeterministicNormal",
    "SHIFT_ZERO",
    "SCALE_ONE",
]


class DeterministicNormal(Normal):
    """Always samples only mean without randomness.

    This distribution behaves like a Normal distribution for log
    probability calculations but always returns the mean when sampling.
    """

    loc: Tensor

    @override
    def sample(self, sample_shape: _size = Size()) -> Tensor:
        """Return the mean instead of sampling.

        Args:
            sample_shape: Sample shape to expand mean by.

        Returns:
            The mean tensor expanded to the sample shape.
        """
        return self.rsample(sample_shape)

    @override
    def rsample(self, sample_shape: _size = Size()) -> Tensor:
        """Return the mean instead of sampling with reparameterization.

        Args:
            sample_shape: Sample shape to expand mean by.

        Returns:
            The mean tensor expanded to the sample shape.
        """
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)


class FCFixedStdNormal(nn.Module):
    """A fully connected layer that outputs a Normal distribution with fixed
    standard deviation.

    This module applies a linear transformation to the input and returns
    a Normal distribution with the transformed output as mean and a
    fixed standard deviation.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        std: float = SHIFT_ZERO,
        normal_cls: type[Normal] | Literal["Normal", "Deterministic"] = Normal,
        squeeze_feature_dim: bool = False,
    ) -> None:
        """Initialize the FCFixedStdNormal module.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            std: Fixed standard deviation value for all outputs.
            normal_cls: Class to use for the normal distribution or a string specifier.
            squeeze_feature_dim: If True, squeezes the feature dimension (requires dim_out=1).
        """
        super().__init__()  # type: ignore
        if not isinstance(normal_cls, type):
            match normal_cls:
                case "Normal":
                    normal_cls = Normal
                case "Deterministic":
                    normal_cls = DeterministicNormal
                case _:
                    raise ValueError(
                        "Normal class must be 'Normal' or 'Deterministic'!"
                    )

        if squeeze_feature_dim:
            assert dim_out == 1, "Cannot squeeze feature dimension when dim_out > 1!"
        self.fc = nn.Linear(dim_in, dim_out)
        self.std = std
        self.normal_cls = normal_cls
        self.squeeze_feature_dim = squeeze_feature_dim

    @override
    def forward(self, x: Tensor) -> Normal:
        """Apply linear transformation and create Normal distribution.

        Args:
            x: Input tensor of shape (..., dim_in).

        Returns:
            Normal distribution with mean from linear transformation and fixed std.
        """
        mean: Tensor = self.fc(x)
        if self.squeeze_feature_dim:
            mean = mean.squeeze(-1)
        std = torch.full_like(mean, self.std)
        return self.normal_cls(mean, std)


class FCNormal(nn.Module):
    """A fully connected layer that outputs a Normal distribution.

    This module applies two separate linear transformations to the input
    to produce both mean and standard deviation for a Normal
    distribution.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        eps: float = 1e-6,
        squeeze_feature_dim: bool = False,
    ) -> None:
        """Initialize the FCNormal module.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            eps: Small constant added to standard deviation for numerical stability.
            squeeze_feature_dim: If True, squeezes the feature dimension (requires dim_out=1).
        """
        super().__init__()  # type: ignore
        if squeeze_feature_dim:
            assert dim_out == 1, "Cannot squeeze feature dimension when dim_out > 1!"

        self.fc_mean = nn.Linear(dim_in, dim_out)
        self.fc_logvar = nn.Linear(dim_in, dim_out)
        self.eps = eps
        self.squeeze_feature_dim = squeeze_feature_dim

        # Initialize parameters
        nn.init.normal_(self.fc_logvar.weight, 0, 0.01)
        nn.init.constant_(self.fc_logvar.bias, 0)

    @override
    def forward(self, x: Tensor) -> Normal:
        """Apply linear transformations and create Normal distribution.

        Args:
            x: Input tensor of shape (..., dim_in).

        Returns:
            Normal distribution with learned mean and standard deviation.
        """
        mean: Tensor = self.fc_mean(x)
        std: Tensor = torch.exp(0.5 * self.fc_logvar(x)) + self.eps
        if self.squeeze_feature_dim:
            mean = mean.squeeze(-1)
            std = std.squeeze(-1)
        return Normal(mean, std)
