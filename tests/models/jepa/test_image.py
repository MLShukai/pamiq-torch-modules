import pytest
import torch

from pamiq_torch_modules.models.jepa.image import (
    ImageJEPAEncoder,
    ImageJEPAPredictor,
    PatchEmbedding,
)
from tests.helpers import parametrize_device

from .helpers import make_bool_masks_randomly


class TestPatchEmbedding:
    """Test suite for PatchEmbedding module."""

    @parametrize_device
    @pytest.mark.parametrize(
        "patch_size, in_channels, embed_dim",
        [
            ((16, 16), 3, 768),
            ((8, 8), 1, 512),
            ((32, 32), 3, 1024),
        ],
    )
    def test_patch_embedding(
        self, patch_size, in_channels, embed_dim, device: torch.device
    ):
        """Test PatchEmbedding module with various configurations.

        Args:
            patch_size: Size of patches.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
            device: Device to run the test on.
        """
        # Initialize patch embedding
        patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        ).to(device)

        # Create sample input
        batch_size = 2
        img_size = (patch_size[0] * 4, patch_size[1] * 4)  # 4x4 grid of patches
        images = torch.randn(batch_size, in_channels, *img_size, device=device)

        # Forward pass
        embeddings = patch_embed(images)

        # Check output shape
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        assert embeddings.shape == (batch_size, n_patches, embed_dim)
        assert embeddings.device == device


class TestImageJEPAEncoder:
    """Test suite for ImageJEPAEncoder."""

    @parametrize_device
    @pytest.mark.parametrize(
        "img_size, patch_size, in_channels, embed_dim, out_dim, depth, num_heads",
        [
            ((224, 224), (16, 16), 3, 768, 384, 12, 12),
            ((192, 192), (8, 8), 3, 512, 256, 8, 8),
            ((128, 128), (32, 32), 1, 384, 192, 6, 6),
        ],
    )
    def test_initialization(
        self,
        img_size: tuple[int, int],
        patch_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        device: torch.device,
    ) -> None:
        """Test initialization of ImageJEPAEncoder.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
            out_dim: Output dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            device: Device to run the test on.
        """
        # Check prerequisite
        assert (
            img_size[0] % patch_size[0] == 0
        ), "Image height must be divisible by patch height"
        assert (
            img_size[1] % patch_size[1] == 0
        ), "Image width must be divisible by patch width"

        # Initialize encoder
        encoder = ImageJEPAEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
        ).to(device)

        # Check attributes
        assert encoder.embed_dim == embed_dim
        assert encoder.num_heads == num_heads
        assert len(encoder.transformer_encoder.layers) == depth
        assert encoder.positional_encodings is not None
        assert encoder.positional_encodings.device == device

    @parametrize_device
    @pytest.mark.parametrize(
        "img_size, patch_size, in_channels, embed_dim, out_dim, depth, num_heads",
        [
            ((64, 64), (16, 16), 3, 96, 48, 2, 3),  # Use smaller values for testing
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_forward(
        self,
        img_size: tuple[int, int],
        patch_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        batch_size: int,
        use_mask: bool,
        device: torch.device,
    ) -> None:
        """Test forward pass of ImageJEPAEncoder.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
            out_dim: Output dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            batch_size: Batch size for test input.
            use_mask: Whether to use mask in the forward pass.
            device: Device to run the test on.
        """
        # Check prerequisite
        assert (
            img_size[0] % patch_size[0] == 0
        ), "Image height must be divisible by patch height"
        assert (
            img_size[1] % patch_size[1] == 0
        ), "Image width must be divisible by patch width"

        # Initialize encoder
        encoder = ImageJEPAEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
        ).to(device)

        # Create sample input
        images = torch.randn(batch_size, in_channels, *img_size, device=device)

        # Calculate expected number of patches
        n_patches_h, n_patches_w = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        n_patches = n_patches_h * n_patches_w

        # Create mask if needed
        masks = None
        if use_mask:
            masks = make_bool_masks_randomly(batch_size, n_patches).to(device)

        # Forward pass
        output = encoder(images, masks)

        # Check output shape and properties
        assert output.shape == (batch_size, n_patches, out_dim)
        assert output.device == device
        assert output.dtype == images.dtype

        # Check that the output changes when using a mask
        if use_mask:
            output_no_mask = encoder(images)
            assert not torch.allclose(output, output_no_mask)


class TestImageJEPAPredictor:
    """Test suite for ImageJEPAPredictor."""

    @parametrize_device
    @pytest.mark.parametrize(
        "n_patches, context_encoder_out_dim, hidden_dim, depth, num_heads",
        [
            ((14, 14), 384, 384, 6, 12),
            ((16, 16), 256, 256, 4, 8),
            ((8, 8), 192, 192, 3, 6),
        ],
    )
    def test_initialization(
        self,
        n_patches: tuple[int, int],
        context_encoder_out_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        device: torch.device,
    ) -> None:
        """Test initialization of ImageJEPAPredictor.

        Args:
            n_patches: Number of patches.
            context_encoder_out_dim: Context encoder output dimension.
            hidden_dim: Hidden dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            device: Device to run the test on.
        """
        # Initialize predictor
        predictor = ImageJEPAPredictor(
            n_patches=n_patches,
            context_encoder_out_dim=context_encoder_out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
        ).to(device)

        # Check attributes
        assert len(predictor.transformer_encoder.layers) == depth
        assert predictor.positional_encodings is not None
        assert predictor.positional_encodings.device == device

    @parametrize_device
    @pytest.mark.parametrize(
        "n_patches, context_encoder_out_dim, hidden_dim, depth, num_heads",
        [
            ((4, 4), 48, 48, 2, 3),  # Use smaller values for testing
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_forward(
        self,
        n_patches: tuple[int, int],
        context_encoder_out_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Test forward pass of ImageJEPAPredictor.

        Args:
            n_patches: Number of patches.
            context_encoder_out_dim: Context encoder output dimension.
            hidden_dim: Hidden dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            batch_size: Batch size for test input.
            device: Device to run the test on.
        """
        # Initialize predictor
        predictor = ImageJEPAPredictor(
            n_patches=n_patches,
            context_encoder_out_dim=context_encoder_out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
        ).to(device)

        # Calculate total number of patches
        total_patches = n_patches[0] * n_patches[1]

        # Create sample input
        latents = torch.randn(
            batch_size, total_patches, context_encoder_out_dim, device=device
        )
        predictor_targets = make_bool_masks_randomly(batch_size, total_patches).to(
            device
        )

        # Forward pass
        output = predictor(latents, predictor_targets)

        # Check output shape and properties
        assert output.shape == (batch_size, total_patches, context_encoder_out_dim)
        assert output.device == device
        assert output.dtype == latents.dtype
