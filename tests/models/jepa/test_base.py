import numpy as np
import pytest
import torch

from pamiq_torch_modules.models.jepa.base import (
    JEPAEncoder,
    JEPAPredictor,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
)
from tests.helpers import parametrize_device

from .helpers import make_bool_masks_randomly


class TestPositionalEncodings:
    """Test suite for positional encoding functions."""

    def test_1d_sincos_pos_embed(self):
        """Test 1D sinusoidal positional embedding generation."""
        # Test with different dimensions
        for embed_dim in [32, 64, 128]:
            # Test with different sequence lengths
            for seq_len in [10, 20, 50]:
                positions = np.arange(seq_len, dtype=float)
                pos_embed = get_1d_sincos_pos_embed(embed_dim, positions)

                # Check shape
                assert pos_embed.shape == (seq_len, embed_dim)

                # Check dtype
                assert pos_embed.dtype == np.float64

                # Check values are within expected range
                assert np.all(pos_embed >= -1.0)
                assert np.all(pos_embed <= 1.0)

    def test_2d_sincos_pos_embed(self):
        """Test 2D sinusoidal positional embedding generation."""
        # Test with different dimensions
        for embed_dim in [32, 64, 128]:
            # Test with different grid sizes
            for grid_h, grid_w in [(4, 4), (7, 8), (16, 16)]:
                pos_embed = get_2d_sincos_pos_embed(embed_dim, (grid_h, grid_w))

                # Check shape
                assert pos_embed.shape == (grid_h * grid_w, embed_dim)

                # Check dtype
                assert pos_embed.dtype == np.float64

                # Check values are within expected range
                assert np.all(pos_embed >= -1.0)
                assert np.all(pos_embed <= 1.0)


class TestJEPAEncoderBase:
    """Test suite for JEPAEncoder base class."""

    @parametrize_device
    def test_forward_without_positional_encodings(self, device: torch.device):
        """Test forward pass without positional encodings."""
        # Model parameters
        embed_dim = 64
        out_dim = 32
        depth = 2
        num_heads = 4

        # Initialize encoder
        encoder = JEPAEncoder(
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
            positional_encodings=None,
        ).to(device)

        # Input parameters
        batch_size = 2
        n_patches = 16

        # Create input tensor
        x = torch.randn(batch_size, n_patches, embed_dim, device=device)

        # Test forward without mask
        output = encoder(x)
        assert output.shape == (batch_size, n_patches, out_dim)
        assert output.device == device

        # Test forward with mask
        masks = make_bool_masks_randomly(batch_size, n_patches).to(device)
        output_masked = encoder(x, masks)
        assert output_masked.shape == (batch_size, n_patches, out_dim)
        assert output_masked.device == device

        # Check if outputs are different when using mask
        assert not torch.allclose(output, output_masked)

    @parametrize_device
    def test_forward_with_positional_encodings(self, device: torch.device):
        """Test forward pass with positional encodings."""
        # Model parameters
        embed_dim = 64
        out_dim = 32
        depth = 2
        num_heads = 4

        # Create positional encodings
        n_patches = 16
        pos_embed = get_1d_sincos_pos_embed(
            embed_dim, np.arange(n_patches, dtype=float)
        ).reshape(1, n_patches, embed_dim)
        pos_embed_tensor = torch.from_numpy(pos_embed).float().to(device)

        # Initialize encoder
        encoder = JEPAEncoder(
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
            positional_encodings=pos_embed_tensor,
        ).to(device)

        # Input parameters
        batch_size = 2

        # Create input tensor
        x = torch.randn(batch_size, n_patches, embed_dim, device=device)

        # Test forward without mask
        output = encoder(x)
        assert output.shape == (batch_size, n_patches, out_dim)
        assert output.device == device


class TestJEPAPredictorBase:
    """Test suite for JEPAPredictor base class."""

    @parametrize_device
    def test_forward_without_positional_encodings(self, device: torch.device):
        """Test forward pass without positional encodings."""
        # Model parameters
        context_encoder_out_dim = 32
        hidden_dim = 64
        depth = 2
        num_heads = 4

        # Initialize predictor
        predictor = JEPAPredictor(
            context_encoder_out_dim=context_encoder_out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            positional_encodings=None,
        ).to(device)

        # Input parameters
        batch_size = 2
        n_patches = 16

        # Create input tensors
        latents = torch.randn(
            batch_size, n_patches, context_encoder_out_dim, device=device
        )
        predictor_targets = make_bool_masks_randomly(batch_size, n_patches).to(device)

        # Test forward
        output = predictor(latents, predictor_targets)
        assert output.shape == (batch_size, n_patches, context_encoder_out_dim)
        assert output.device == device

    @parametrize_device
    def test_forward_with_positional_encodings(self, device: torch.device):
        """Test forward pass with positional encodings."""
        # Model parameters
        context_encoder_out_dim = 32
        hidden_dim = 64
        depth = 2
        num_heads = 4

        # Create positional encodings
        n_patches = 16
        pos_embed = get_1d_sincos_pos_embed(
            hidden_dim, np.arange(n_patches, dtype=float)
        ).reshape(1, n_patches, hidden_dim)
        pos_embed_tensor = torch.from_numpy(pos_embed).float().to(device)

        # Initialize predictor
        predictor = JEPAPredictor(
            context_encoder_out_dim=context_encoder_out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            positional_encodings=pos_embed_tensor,
        ).to(device)

        # Input parameters
        batch_size = 2

        # Create input tensors
        latents = torch.randn(
            batch_size, n_patches, context_encoder_out_dim, device=device
        )
        predictor_targets = make_bool_masks_randomly(batch_size, n_patches).to(device)

        # Test forward
        output = predictor(latents, predictor_targets)
        assert output.shape == (batch_size, n_patches, context_encoder_out_dim)
        assert output.device == device
