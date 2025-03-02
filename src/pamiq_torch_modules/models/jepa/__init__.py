"""Joint-Embedding Predictive Architecture (JEPA) implementation.

This module provides a PyTorch implementation of the Joint-Embedding
Predictive Architecture (JEPA) for self-supervised learning, with
support for both image and audio modalities.
"""

from .base import (
    JEPAEncoder,
    JEPAPredictor,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
)
from .image import (
    ImageJEPAEncoder,
    ImageJEPAPredictor,
    PatchEmbedding,
)

__all__ = [
    # Base classes
    "JEPAEncoder",
    "JEPAPredictor",
    "get_1d_sincos_pos_embed",
    "get_2d_sincos_pos_embed",
    # Image JEPA
    "ImageJEPAEncoder",
    "ImageJEPAPredictor",
    "PatchEmbedding",
]
