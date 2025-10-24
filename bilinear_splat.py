"""
Differentiable bilinear splatting for image reconstruction.

This module provides a differentiable way to accumulate small images (ROIs)
onto a larger canvas using bilinear interpolation, enabling end-to-end training.
"""

import torch
import torch.nn.functional as F


def bilinear_splat(I, offsets, canvas_size):
    """
    Bilinear splatting of small image(s) I into a larger canvas.

    This is fully differentiable and allows gradients to flow back through
    the offset positions and image intensities.

    Args:
        I: Tensor (N, H, W) or (N, 1, H, W) — local grayscale images
        offsets: Tensor (N, 2) — global (x, y) float offsets for each image
        canvas_size: tuple (Hc, Wc) — size of the output canvas

    Returns:
        G: Tensor (1, 1, Hc, Wc) — reconstructed global canvas

    Notes:
        - Offsets are in (x, y) format where x is horizontal, y is vertical
        - Uses bilinear weights to distribute each pixel to 4 neighbors
        - Automatically normalizes overlapping regions
        - Fully differentiable for end-to-end training
    """
    # Handle both (N, H, W) and (N, 1, H, W) inputs
    if I.dim() == 3:
        I = I.unsqueeze(1)  # Add channel dimension

    N, C, H, W = I.shape
    assert C == 1, "Only grayscale images supported (C=1)"

    Hc, Wc = canvas_size
    device = I.device

    # Initialize canvas and weight accumulator
    G = torch.zeros((1, 1, Hc, Wc), device=device, dtype=I.dtype)
    Wsum = torch.zeros_like(G)

    # Create meshgrid once (more efficient)
    y_local, x_local = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    for i in range(N):
        ox, oy = offsets[i]

        # Global coordinates for each pixel in the local image
        x_global = x_local + ox
        y_global = y_local + oy

        # Bilinear interpolation: find 4 nearest neighbors
        x0 = torch.floor(x_global).long()
        y0 = torch.floor(y_global).long()
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute interpolation weights
        dx = x_global - x0.float()
        dy = y_global - y0.float()

        # Valid region mask (all 4 corners must be in bounds)
        valid = (x0 >= 0) & (x1 < Wc) & (y0 >= 0) & (y1 < Hc)

        # Get the image values (broadcast to match spatial dimensions)
        I_val = I[i, 0]  # Shape: (H, W)

        # Splat to all 4 neighbors with bilinear weights
        corners = [
            (y0, x0, (1 - dx) * (1 - dy)),  # Top-left
            (y0, x1, dx * (1 - dy)),  # Top-right
            (y1, x0, (1 - dx) * dy),  # Bottom-left
            (y1, x1, dx * dy),  # Bottom-right
        ]

        for y_t, x_t, weight in corners:
            # Apply validity mask to weights
            w_val = weight * valid

            # Weighted image contribution
            contribution = I_val * w_val

            # Accumulate using index_put with accumulate=True
            # This handles overlapping writes correctly
            G[0, 0].index_put_((y_t, x_t), contribution, accumulate=True)
            Wsum[0, 0].index_put_((y_t, x_t), w_val, accumulate=True)

    # Normalize by accumulated weights to handle overlaps
    # Add epsilon to avoid division by zero
    G = G / (Wsum + 1e-8)

    return G


def bilinear_splat_batch(I, offsets, canvas_size):
    """
    Batch version of bilinear splatting (more efficient for large N).

    Args:
        I: Tensor (N, H, W) or (N, 1, H, W) — local grayscale images
        offsets: Tensor (N, 2) — global (x, y) float offsets
        canvas_size: tuple (Hc, Wc) — size of output canvas

    Returns:
        G: Tensor (1, 1, Hc, Wc) — reconstructed global canvas

    Note: This vectorized version is faster but may use more memory.
    """
    if I.dim() == 3:
        I = I.unsqueeze(1)

    N, C, H, W = I.shape
    assert C == 1, "Only grayscale images supported"

    Hc, Wc = canvas_size
    device = I.device

    G = torch.zeros((1, 1, Hc, Wc), device=device, dtype=I.dtype)
    Wsum = torch.zeros_like(G)

    # Create meshgrid: (H, W)
    y_local, x_local = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Broadcast to (N, H, W)
    x_global = x_local[None, :, :] + offsets[:, 0, None, None]  # (N, H, W)
    y_global = y_local[None, :, :] + offsets[:, 1, None, None]  # (N, H, W)

    # Bilinear neighbors
    x0 = torch.floor(x_global).long()
    y0 = torch.floor(y_global).long()
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x_global - x0.float()
    dy = y_global - y0.float()

    # Valid mask
    valid = (x0 >= 0) & (x1 < Wc) & (y0 >= 0) & (y1 < Hc)

    I_flat = I[:, 0, :, :]  # (N, H, W)

    # Splat all corners
    corners = [
        (y0, x0, (1 - dx) * (1 - dy)),
        (y0, x1, dx * (1 - dy)),
        (y1, x0, (1 - dx) * dy),
        (y1, x1, dx * dy),
    ]

    for y_t, x_t, weight in corners:
        w_val = weight * valid  # (N, H, W)
        contribution = I_flat * w_val  # (N, H, W)

        # Flatten for scatter
        y_flat = y_t.flatten()
        x_flat = x_t.flatten()
        contrib_flat = contribution.flatten()
        w_flat = w_val.flatten()

        # Only accumulate valid positions
        mask = (y_flat >= 0) & (y_flat < Hc) & (x_flat >= 0) & (x_flat < Wc)

        if mask.any():
            G[0, 0].index_put_(
                (y_flat[mask], x_flat[mask]), contrib_flat[mask], accumulate=True
            )
            Wsum[0, 0].index_put_(
                (y_flat[mask], x_flat[mask]), w_flat[mask], accumulate=True
            )

    # G = G / (Wsum + 1e-8)
    return G


def bilinear_splat_training_batch(I, offsets, canvas_size):
    """
    Fully batched bilinear splatting for training batches.

    Processes multiple samples in parallel, where each sample contains
    T timesteps of frames to splat onto its own canvas.

    Args:
        I: Tensor (B, T, H, W) — batch of temporal sequences of local images
        offsets: Tensor (B, T, 2) — global (x, y) float offsets for each frame
        canvas_size: tuple (Hc, Wc) — size of output canvas

    Returns:
        G: Tensor (B, 1, Hc, Wc) — reconstructed global canvas for each batch item

    Notes:
        - This is the most efficient version for training with batches
        - Each batch item gets its own independent canvas
        - Uses vectorized operations across timesteps (T dimension)
        - Fully differentiable for end-to-end training
    """
    B, T, H, W = I.shape
    Hc, Wc = canvas_size
    device = I.device
    dtype = I.dtype

    # Initialize output for all batch items: (B, 1, Hc, Wc)
    G_batch = torch.zeros((B, 1, Hc, Wc), device=device, dtype=dtype)
    Wsum_batch = torch.zeros_like(G_batch)

    # Create meshgrid once for efficiency: (H, W)
    y_local, x_local = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Broadcast to (B, T, H, W)
    # offsets shape: (B, T, 2) -> extract x and y
    x_global = x_local[None, None, :, :] + offsets[:, :, 0, None, None]  # (B, T, H, W)
    y_global = y_local[None, None, :, :] + offsets[:, :, 1, None, None]  # (B, T, H, W)

    # Bilinear interpolation: find 4 nearest neighbors
    x0 = torch.floor(x_global).long()
    y0 = torch.floor(y_global).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Compute interpolation weights
    dx = x_global - x0.float()
    dy = y_global - y0.float()

    # Valid region mask (all 4 corners must be in bounds)
    valid = (x0 >= 0) & (x1 < Wc) & (y0 >= 0) & (y1 < Hc)

    # Define the 4 corners with their weights
    corners = [
        (y0, x0, (1 - dx) * (1 - dy)),  # Top-left
        (y0, x1, dx * (1 - dy)),  # Top-right
        (y1, x0, (1 - dx) * dy),  # Bottom-left
        (y1, x1, dx * dy),  # Bottom-right
    ]

    # Create flattened canvas for all batch items
    # Instead of processing each batch separately, encode batch index in linear indices
    G_flat = torch.zeros(B * Hc * Wc, device=device, dtype=dtype)
    Wsum_flat = torch.zeros(B * Hc * Wc, device=device, dtype=dtype)

    # Create batch offset tensor: (B, 1, 1, 1) broadcasts to (B, T, H, W)
    batch_offset = torch.arange(B, device=device).view(B, 1, 1, 1) * (Hc * Wc)

    # Process all corners with vectorized scatter
    for y_t, x_t, weight in corners:
        # Compute weighted values for all batches: (B, T, H, W)
        w_val = weight * valid  # (B, T, H, W)
        contribution = I * w_val  # (B, T, H, W)

        # Compute linear indices that encode batch position
        # linear_idx = b * (Hc * Wc) + y * Wc + x
        linear_idx = batch_offset + y_t * Wc + x_t  # (B, T, H, W)

        # Flatten all dimensions: (B*T*H*W,)
        idx_flat = linear_idx.flatten()
        contrib_flat = contribution.flatten()
        w_flat = w_val.flatten()

        # Filter valid indices (in bounds and passes valid mask)
        valid_mask = (idx_flat >= 0) & (idx_flat < B * Hc * Wc)

        if valid_mask.any():
            # Single scatter operation for all batch items
            G_flat.scatter_add_(0, idx_flat[valid_mask], contrib_flat[valid_mask])
            Wsum_flat.scatter_add_(0, idx_flat[valid_mask], w_flat[valid_mask])

    # Reshape back to batch format: (B, Hc, Wc) -> (B, 1, Hc, Wc)
    G_batch = G_flat.view(B, Hc, Wc).unsqueeze(1)
    Wsum_batch = Wsum_flat.view(B, Hc, Wc).unsqueeze(1)

    # Normalize by accumulated weights
    # G_batch = G_batch / (Wsum_batch + 1e-8)

    return G_batch
