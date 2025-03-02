import math
from functools import partial
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

FloatNDArray: TypeAlias = npt.NDArray[np.floating]


def get_1d_sincos_pos_embed(embed_dim: int, positions: FloatNDArray) -> FloatNDArray:
    """Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        positions: Positions to encode, shape [length].

    Returns:
        Positional embeddings, shape [length, embed_dim].
    """
    assert embed_dim % 2 == 0, "Embed dimension must be even"
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [embed_dim//2]

    outer = np.outer(positions, omega)  # [length, embed_dim//2]
    emb_sin = np.sin(outer)  # [length, embed_dim//2]
    emb_cos = np.cos(outer)  # [length, embed_dim//2]

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # [length, embed_dim]
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]) -> FloatNDArray:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid size as (height, width).

    Returns:
        Positional embeddings of shape [grid_size[0] * grid_size[1], embed_dim].
    """
    assert embed_dim % 2 == 0, "Embed dimension must be even"

    # Use half of dimensions for each spatial dimension
    embed_dim_half = embed_dim // 2
    grid_h, grid_w = grid_size

    # Create position grid
    grid_h_pos = np.arange(grid_h, dtype=float)
    grid_w_pos = np.arange(grid_w, dtype=float)
    grid = np.meshgrid(grid_w_pos, grid_h_pos)
    grid = np.stack(grid, axis=0)  # [2, grid_h, grid_w]

    # Flatten grid
    grid = grid.reshape([2, -1])  # [2, grid_h*grid_w]

    # Create positional embeddings
    pos_emb_h = get_1d_sincos_pos_embed(embed_dim_half, grid[1])  # height embeddings
    pos_emb_w = get_1d_sincos_pos_embed(embed_dim_half, grid[0])  # width embeddings

    # Combine embeddings
    pos_emb = np.concatenate(
        [pos_emb_h, pos_emb_w], axis=1
    )  # [grid_h*grid_w, embed_dim]
    return pos_emb


class JEPAEncoder(nn.Module):
    """Base class for JEPA encoders with mask support."""

    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        positional_encodings: Tensor | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        init_std: float = 0.02,
    ) -> None:
        """Initialize the JEPAEncoder.

        Args:
            embed_dim: Embedding dimension per patch.
            out_dim: Output dimension per patch.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            positional_encodings: Pre-computed positional encodings to add to embeddings.
                Shape: [1, n_patches, embed_dim]. Defaults to None.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            dropout: Dropout rate.
            activation: Activation function to use.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()  # type: ignore
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std

        # Mask token vector
        self.mask_token_vector = nn.Parameter(torch.empty(embed_dim))

        # Store positional encodings if provided

        self.positional_encodings: Tensor | None
        self.register_buffer("positional_encodings", positional_encodings)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
        )

        # Output projection
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for the model."""
        nn.init.trunc_normal_(self.mask_token_vector, std=self.init_std)
        self.apply(partial(_init_weights, init_std=self.init_std))
        fix_init_weight(self.transformer_encoder)

    @override
    def forward(self, x: Tensor, masks: Tensor | None = None) -> Tensor:
        """Encode input into latents, applying boolean masks if provided.

        Args:
            x: Input embeddings. Shape: [batch_size, n_patches, embed_dim]
            masks: Boolean masks. Shape: [batch_size, n_patches].
                   True values indicate masked patches. Defaults to None.

        Returns:
            Encoded latents. Shape: [batch_size, n_patches, out_dim]
        """
        # Apply mask if provided
        if masks is not None:
            assert (
                x.shape[:-1] == masks.shape
            ), "Mask shape must match batch and patch dimensions"
            x = x.clone()  # Avoid breaking gradient graph
            x[masks] = self.mask_token_vector

        # Add positional embeddings if provided
        if self.positional_encodings is not None:
            x = x + self.positional_encodings

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Apply normalization and output projection
        x = self.norm(x)
        x = self.out_proj(x)

        return x


class JEPAPredictor(nn.Module):
    """Base class for JEPA predictors with boolean target support."""

    def __init__(
        self,
        context_encoder_out_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        positional_encodings: Tensor | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        init_std: float = 0.02,
    ) -> None:
        """Initialize the JEPAPredictor.

        Args:
            context_encoder_out_dim: Output dimension of the context encoder.
            hidden_dim: Hidden dimension for prediction.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            positional_encodings: Pre-computed positional encodings to add to embeddings.
                Shape: [1, n_patches, hidden_dim]. Defaults to None.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            dropout: Dropout rate.
            activation: Activation function to use.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()  # type: ignore
        self.init_std = init_std

        # Input projection
        self.input_proj = nn.Linear(context_encoder_out_dim, hidden_dim, bias=True)

        # Prediction token
        self.prediction_token_vector = nn.Parameter(torch.empty(hidden_dim))

        # Store positional encodings if provided
        self.positional_encodings: Tensor | None
        self.register_buffer("positional_encodings", positional_encodings)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
        )

        # Output projections
        self.predictor_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(hidden_dim, context_encoder_out_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for the model."""
        nn.init.trunc_normal_(self.prediction_token_vector, std=self.init_std)
        self.apply(partial(_init_weights, init_std=self.init_std))
        fix_init_weight(self.transformer_encoder)

    @override
    def forward(
        self,
        latents: Tensor,
        predictor_targets: Tensor,
    ) -> Tensor:
        """Predict latents of target patches based on input latents and boolean
        targets.

        Args:
            latents: Input latents from context_encoder.
                Shape: [batch, n_patches, context_encoder_out_dim]
            predictor_targets: Boolean targets for patches.
                Shape: [batch, n_patches]. True values indicate target patches to be predicted.

        Returns:
            Prediction results for target patches.
                Shape: [batch, n_patches, context_encoder_out_dim]
        """
        # Project input to hidden dimension
        x = self.input_proj(latents)

        # Add prediction tokens
        x = x.clone()  # Avoid breaking gradient graph
        x[predictor_targets] += self.prediction_token_vector

        # Add positional embeddings if provided
        if self.positional_encodings is not None:
            x = x + self.positional_encodings

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Apply normalization and output projection
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def _init_weights(m: nn.Module, init_std: float) -> None:
    """Initialize weights for the model.

    Args:
        m: Module to initialize.
        init_std: Standard deviation for weight initialization.
    """
    match m:
        case nn.Linear():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case nn.LayerNorm():
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        case nn.Conv2d() | nn.Conv1d():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case _:
            pass


def fix_init_weight(transformer_encoder: nn.TransformerEncoder) -> None:
    """Fix initialization weights for transformer encoder.

    Args:
        transformer_encoder: Transformer encoder to fix.
    """
    for i, layer in enumerate(transformer_encoder.layers, start=1):
        layer = cast(nn.TransformerEncoderLayer, layer)
        # Scale the weights based on layer depth
        scale_factor = 1.0 / math.sqrt(2.0 * i)

        # Scale the projection in multi-head attention
        layer.self_attn.out_proj.weight.data.mul_(scale_factor)

        # Scale the feedforward's second linear layer
        layer.linear2.weight.data.mul_(scale_factor)
