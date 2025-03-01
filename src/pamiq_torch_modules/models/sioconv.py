import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes input by dividing by RMS (root mean square) and
    optionally applies learned scaling parameters.
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True
    ) -> None:
        """Initialize the RMSNorm module.

        Args:
            dim: Feature dimension to normalize.
            eps: Small constant for numerical stability.
            elementwise_affine: Whether to learn scaling parameters.
        """
        super().__init__()  # type: ignore
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape.
        """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            output = output * self.weight
        return output


class SwiGLU(nn.Module):
    """SwiGLU activation function as described in the paper "GLU Variants
    Improve Transformer".

    Combines the Swish activation function with a Gated Linear Unit.
    """

    def __init__(self, dim: int, dim_ff_hidden: int) -> None:
        """Initialize the SwiGLU module.

        Args:
            dim: Input dimension.
            dim_ff_hidden: Hidden dimension for feedforward network.
        """
        super().__init__()  # type: ignore
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_gate = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU activation function.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        return self.fc_out(self.fc(x) * self.act(self.fc_gate(x)))


class SioConvPSLayer(nn.Module):
    """Single layer of SioConv with Parallel Scan algorithm.

    This layer implements State-space model with Input-Output
    Convolution using a parallel scan algorithm for efficient sequence
    processing.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the SioConvPSLayer.

        Args:
            dim: Model dimension.
        """
        super().__init__()  # type: ignore
        self.dim = dim
        self.fc_ln_z = nn.Linear(dim, dim)  # Input projection
        self.fc_y = nn.Linear(dim, dim)  # Output projection
        self.fc_y_gate = nn.Linear(dim, dim)  # Output gate projection
        self.act = nn.SiLU()  # Activation function
        self.fc_dt = nn.Linear(dim, dim)  # Time-decay projection

    @override
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Process sequence with SioConv.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            hidden: Hidden state tensor of shape (batch_size, dim).

        Returns:
            Tuple of (output tensor, new hidden state).
        """
        # Input projection with log-domain computation for numerical stability
        ln_z = -F.softplus(-self.fc_ln_z(x))  # (batch, seq_len, dim)

        # Time-decay parameters
        ln_da = -F.softplus(-self.fc_dt(x))  # (batch, seq_len, dim)
        ln_z_da = ln_z + ln_da
        ln_o_da = -F.softplus(self.fc_dt(x))  # (batch, seq_len, dim)
        ln_o_da_cumsum = torch.cumsum(ln_o_da, dim=1)

        # Parallel scan algorithm for efficient sequence processing
        ln_z_da_ln_o_da_cumsum = ln_z_da - ln_o_da_cumsum  # (batch, seq_len, dim)
        logcumsumexp_ln_z_da_ln_o_da_cumsum = torch.logcumsumexp(
            ln_z_da_ln_o_da_cumsum, dim=1
        )

        # Internal hidden state computation
        h_inner = torch.exp(
            logcumsumexp_ln_z_da_ln_o_da_cumsum + ln_o_da_cumsum
        )  # (batch, seq_len, dim)

        # Include cross-chunk hidden state effects
        h_cross = torch.einsum(
            "bld,bd->bld", torch.exp(ln_o_da_cumsum), hidden
        )  # (batch, seq_len, dim)

        # Combine internal and cross-chunk effects
        h = h_inner + h_cross

        # Output projection with gating
        y = self.fc_y(h) * self.act(self.fc_y_gate(x))

        return y, h


class SioConvPSBlock(nn.Module):
    """SioConv block with normalization and feed-forward network.

    Combines SioConvPSLayer with normalization layers and a feed-forward
    network in a residual block structure.
    """

    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float = 0.0) -> None:
        """Initialize the SioConvPSBlock.

        Args:
            dim: Model dimension.
            dim_ff_hidden: Hidden dimension for feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()  # type: ignore
        self.sio_conv = SioConvPSLayer(dim)
        self.ffn = SwiGLU(dim, dim_ff_hidden)
        self.norm_sio_conv = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Process sequence with SioConvPS block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            hidden: Hidden state tensor of shape (batch_size, dim).

        Returns:
            Tuple of (output tensor, new hidden state).
        """
        # First sub-block: SioConv with residual connection
        residual = x
        x_norm = self.norm_sio_conv(x)
        x_sio, hidden_out = self.sio_conv(x_norm, hidden)
        x_sio = self.dropout(x_sio)
        x = x_sio + residual

        # Second sub-block: Feed-forward with residual connection
        residual = x
        x_norm = self.norm_ffn(x)
        x_ffn = self.ffn(x_norm)
        x_ffn = self.dropout(x_ffn)
        x = x_ffn + residual

        return x, hidden_out


class SioConvPS(nn.Module):
    """SioConv model with Parallel Scan algorithm.

    Stacks multiple SioConvPS blocks for sequence processing.
    """

    def __init__(
        self, depth: int, dim: int, dim_ff_hidden: int, dropout: float = 0.0
    ) -> None:
        """Initialize the SioConvPS model.

        Args:
            depth: Number of SioConvPS blocks.
            dim: Model dimension.
            dim_ff_hidden: Hidden dimension for feed-forward networks.
            dropout: Dropout probability.
        """
        super().__init__()  # type: ignore
        self.blocks = nn.ModuleList(
            [SioConvPSBlock(dim, dim_ff_hidden, dropout) for _ in range(depth)]
        )
        self.depth = depth
        self.dim = dim

    @override
    def forward(self, x: Tensor, hidden_stack: Tensor) -> tuple[Tensor, Tensor]:
        """Process sequence through stacked SioConvPS blocks.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim) or (batch_size, dim)
               if processing a single token.
            hidden: Hidden state tensor of shape (batch_size, depth, dim).
                   If None, initializes with zeros.

        I/O Shapes:
            (*batch, len, dim), (*batch, depth, dim) -> (*batch, len, dim), (*batch, depth, len, dim) or
            (len, dim), (depth, dim) -> (len, dim), (depth, len, dim) or
            (*batch, dim), (*batch, depth, dim) -> (*batch, dim), (*batch, depth, dim) or
            (dim), (depth, dim) -> (dim), (depth, dim)

        Returns:
            Tuple of (output tensor of the same shape as input,
                     new hidden state tensor of shape (batch_size, depth, dim)).
        """
        no_batch = hidden_stack.ndim < 3
        if no_batch:
            x = x.unsqueeze(0)
            hidden_stack = hidden_stack.unsqueeze(0)

        no_len = x.ndim < 3
        if no_len:
            x = x.unsqueeze(1)

        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[len(batch_shape) :])
        hidden_stack = hidden_stack.reshape(-1, *hidden_stack.shape[len(batch_shape) :])

        hidden_out_list: list[Tensor] = []
        for i, module in enumerate(self.blocks):
            x, hidden_out = module(x, hidden_stack[:, i, :])
            hidden_out_list.append(hidden_out)

        hidden_out_stack = torch.stack(hidden_out_list).transpose(1, 0)

        x = x.view(*batch_shape, *x.shape[1:])
        hidden_out_stack = hidden_out_stack.view(
            *batch_shape, *hidden_out_stack.shape[1:]
        )

        if no_len:
            x = x.squeeze(1)
            hidden_out_stack = hidden_out_stack.squeeze(2)

        if no_batch:
            x = x.squeeze(0)
            hidden_out_stack = hidden_out_stack.squeeze(0)

        return x, hidden_out_stack
