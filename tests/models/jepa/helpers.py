import torch


def make_bool_masks_randomly(
    batch_size: int,
    n_patches: int,
    mask_ratio: float = 0.75,
) -> torch.Tensor:
    """Boolean mask maker for the tests.

    Args:
        batch_size: Batch size.
        n_patches: Total number of patches.
        mask_ratio: Ratio of patches to be masked.

    Returns:
        Boolean mask tensor. Shape: [batch_size, n_patches].
        True values indicate masked patches.
    """
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
    n_masked = int(n_patches * mask_ratio)
    for i in range(batch_size):
        mask[i, torch.randperm(n_patches)[:n_masked]] = True
    return mask
