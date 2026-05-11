import torch


def extract_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split a batch of images into non-overlapping patches.

    Args:
        images: Tensor of shape (B, C, H, W).
        patch_size: Side length of each square patch in pixels.

    Returns:
        Tensor of shape (B, num_patches, patch_dim) where
        num_patches = (H // patch_size) * (W // patch_size) and
        patch_dim = C * patch_size * patch_size.
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"Image dimensions ({H}x{W}) must be divisible by patch_size ({patch_size})"
    )

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # (B, C, H//p, W//p, p, p)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    # (B, C, num_patches, p, p)
    patches = patches.permute(0, 2, 1, 3, 4)
    # (B, num_patches, C, p, p)
    patches = patches.contiguous().view(B, -1, C * patch_size * patch_size)
    # (B, num_patches, patch_dim)
    return patches
