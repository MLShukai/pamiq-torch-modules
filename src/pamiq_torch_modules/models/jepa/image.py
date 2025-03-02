import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from ..utils import size_2d, size_2d_to_int_tuple
from .base import JEPAEncoder, JEPAPredictor, get_2d_sincos_pos_embed


class PatchEmbedding(nn.Module):
    """Convert input images into patch embeddings."""

    def __init__(
        self,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Initialize the PatchEmbedding module.

        Args:
            patch_size: Pixel size per patch.
            in_channels: Number of input image channels.
            embed_dim: Embedding dimension per patch.
        """
        super().__init__()  # type: ignore
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Convert input images to patch embeddings.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Patch embeddings of shape [batch_size, n_patches, embed_dim]
        """
        x = self.proj(x).flatten(-2).transpose(-2, -1)
        return x


class ImageJEPAEncoder(JEPAEncoder):
    """JEPA encoder for images with boolean mask support."""

    def __init__(
        self,
        img_size: size_2d = 224,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        out_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        init_std: float = 0.02,
    ) -> None:
        """Initialize the ImageJEPAEncoder.

        Args:
            img_size: Input image size.
            patch_size: Pixel size per patch.
            in_channels: Input image channels.
            embed_dim: Embedding dimension per patch.
            out_dim: Output dimension per patch.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            dropout: Dropout rate.
            activation: Activation function to use.
            init_std: Standard deviation for weight initialization.
        """
        # Calculate number of patches and create positional encodings
        img_size = size_2d_to_int_tuple(img_size)
        patch_size = size_2d_to_int_tuple(patch_size)
        img_height, img_width = img_size
        patch_height, patch_width = patch_size

        assert (
            img_height % patch_height == 0
        ), "Image height must be divisible by patch height"
        assert (
            img_width % patch_width == 0
        ), "Image width must be divisible by patch width"

        n_patches_h = img_height // patch_height
        n_patches_w = img_width // patch_width
        n_patches = n_patches_h * n_patches_w

        # Create positional encodings
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            (n_patches_h, n_patches_w),
        ).reshape(1, n_patches, embed_dim)
        pos_embed_tensor = Tensor(pos_embed).float()

        # Initialize parent class with positional encodings
        super().__init__(
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
            positional_encodings=pos_embed_tensor,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
            init_std=init_std,
        )

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

    @override
    def forward(self, x: Tensor, masks: Tensor | None = None) -> Tensor:
        """Encode input images into latents, applying boolean masks if
        provided.

        Args:
            x: Input images. Shape: [batch_size, in_channels, height, width]
            masks: Boolean masks for patches. Shape: [batch_size, n_patches].
                   True values indicate masked patches. Defaults to None.

        Returns:
            Encoded latents. Shape: [batch_size, n_patches, out_dim]
        """
        # Convert images to patch embeddings
        x = self.patch_embed(x)

        # Apply the parent class forward which handles masking and transformer encoding
        x = super().forward(x, masks)

        return x


class ImageJEPAPredictor(JEPAPredictor):
    """JEPA predictor for images with boolean target support."""

    def __init__(
        self,
        n_patches: size_2d,
        context_encoder_out_dim: int = 384,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        init_std: float = 0.02,
    ) -> None:
        """Initialize the ImageJEPAPredictor.

        Args:
            n_patches: Number of patches along vertical and horizontal axes.
            context_encoder_out_dim: Output dimension of the context encoder.
            hidden_dim: Hidden dimension for prediction.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            dropout: Dropout rate.
            activation: Activation function to use.
            init_std: Standard deviation for weight initialization.
        """
        # Calculate number of patches and create positional encodings
        n_patches_h, n_patches_w = size_2d_to_int_tuple(n_patches)
        total_patches = n_patches_h * n_patches_w

        # Create positional encodings
        pos_embed = get_2d_sincos_pos_embed(
            hidden_dim,
            (n_patches_h, n_patches_w),
        ).reshape(1, total_patches, hidden_dim)
        pos_embed_tensor = Tensor(pos_embed).float()

        # Initialize parent class with positional encodings
        super().__init__(
            context_encoder_out_dim=context_encoder_out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            positional_encodings=pos_embed_tensor,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
            init_std=init_std,
        )
