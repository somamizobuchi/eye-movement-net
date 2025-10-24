"""
Reconstruction utilities that integrate FullModel output with bilinear splatting.

This module bridges the gap between FullModel's velocity predictions and
the bilinear splatting reconstruction.
"""

import torch
from bilinear_splat import bilinear_splat_batch, bilinear_splat_training_batch


def integrate_velocities(velocities, initial_position, dt=1.0):
    """
    Integrate velocities to get absolute positions using cumulative sum.

    Args:
        velocities: Tensor (T, 2) — velocities (dx/dt, dy/dt) at each timestep
        initial_position: Tensor (2,) — starting position (x0, y0)
        dt: float — time step (default: 1.0 for frame-by-frame)

    Returns:
        positions: Tensor (T, 2) — absolute positions (x, y) at each timestep
    """
    # Multiply velocities by dt to get displacements
    displacements = velocities * dt  # (T, 2)

    # Cumulative sum to get relative positions
    relative_positions = torch.cumsum(displacements, dim=0)  # (T, 2)

    # Add initial position to get absolute positions
    positions = initial_position.unsqueeze(0) + relative_positions  # (T, 2)

    return positions


def integrate_velocities_batch(velocities, initial_positions, dt=1.0):
    """
    Integrate velocities for entire batch (vectorized version).

    Args:
        velocities: Tensor (B, T, 2) — velocities for batch
        initial_positions: Tensor (B, 2) — starting positions for batch
        dt: float — time step (default: 1.0 for frame-by-frame)

    Returns:
        positions: Tensor (B, T, 2) — absolute positions at each timestep for batch
    """
    # Multiply velocities by dt to get displacements
    displacements = velocities * dt  # (B, T, 2)

    # Cumulative sum along time dimension to get relative positions
    relative_positions = torch.cumsum(displacements, dim=1)  # (B, T, 2)

    # Add initial position to get absolute positions
    positions = initial_positions.unsqueeze(1) + relative_positions  # (B, T, 2)

    return positions


def reconstruct_from_fullmodel_output(
    eye_velocities,
    reconstructed_frames,
    initial_position,
    canvas_size,
    dt=1.0,
    batch_idx=0,
):
    """
    Reconstruct global image from FullModel outputs using bilinear splatting.

    Args:
        eye_velocities: Tensor (batch_size, t_out, 2) — predicted velocities
        reconstructed_frames: Tensor (batch_size, t_out, N, N) — predicted ROI frames
        initial_position: Tensor (2,) or Tensor (batch_size, 2) — starting position(s)
        canvas_size: tuple (Hc, Wc) — size of reconstructed canvas
        dt: float — time step for velocity integration
        batch_idx: int — which batch element to reconstruct (default: 0)

    Returns:
        G: Tensor (1, 1, Hc, Wc) — reconstructed global canvas
    """
    # Extract single batch element
    velocities = eye_velocities[batch_idx]  # (t_out, 2)
    frames = reconstructed_frames[batch_idx]  # (t_out, N, N)

    # Handle initial position
    if initial_position.dim() == 2:
        init_pos = initial_position[batch_idx]  # (2,)
    else:
        init_pos = initial_position  # (2,)

    # Integrate velocities to get absolute positions
    positions = integrate_velocities(velocities, init_pos, dt)  # (t_out, 2)

    # Apply bilinear splatting (use batch version for better performance)
    G = bilinear_splat_batch(frames, positions, canvas_size)

    return G


def reconstruct_batch(
    eye_velocities,
    reconstructed_frames,
    initial_positions,
    canvas_size,
    dt=1.0,
):
    """
    Reconstruct global images for an entire batch.

    Args:
        eye_velocities: Tensor (batch_size, t_out, 2) — predicted velocities
        reconstructed_frames: Tensor (batch_size, t_out, N, N) — predicted ROI frames
        initial_positions: Tensor (batch_size, 2) — starting positions
        canvas_size: tuple (Hc, Wc) — size of reconstructed canvas
        dt: float — time step for velocity integration

    Returns:
        reconstructions: list of Tensors, each (1, 1, Hc, Wc)
    """
    batch_size = eye_velocities.shape[0]
    reconstructions = []

    for i in range(batch_size):
        G = reconstruct_from_fullmodel_output(
            eye_velocities,
            reconstructed_frames,
            initial_positions,
            canvas_size,
            dt,
            batch_idx=i,
        )
        reconstructions.append(G)

    return reconstructions


def reconstruct_batch_optimized(
    eye_velocities,
    reconstructed_frames,
    initial_positions,
    canvas_size,
    dt=1.0,
):
    """
    Reconstruct global images for an entire batch (OPTIMIZED VERSION).

    This is the fastest version for training, using fully vectorized operations.

    Args:
        eye_velocities: Tensor (batch_size, t_out, 2) — predicted velocities
        reconstructed_frames: Tensor (batch_size, t_out, N, N) — predicted ROI frames
        initial_positions: Tensor (batch_size, 2) — starting positions
        canvas_size: tuple (Hc, Wc) — size of reconstructed canvas
        dt: float — time step for velocity integration

    Returns:
        G: Tensor (batch_size, 1, Hc, Wc) — reconstructed canvases for all batch items
    """
    # Integrate velocities for entire batch at once
    positions = integrate_velocities_batch(eye_velocities, initial_positions, dt)

    # Apply batched bilinear splatting
    G = bilinear_splat_training_batch(reconstructed_frames, positions, canvas_size)

    return G


def compute_reconstruction_loss(
    eye_velocities,
    reconstructed_frames,
    initial_positions,
    target_images,
    mask=None,
    canvas_size=None,
    dt=1.0,
):
    """
    Compute reconstruction loss for training.

    Args:
        eye_velocities: Tensor (batch_size, t_out, 2)
        reconstructed_frames: Tensor (batch_size, t_out, N, N)
        initial_positions: Tensor (batch_size, 2)
        target_images: Tensor (batch_size, Hc, Wc) — ground truth images
        mask: Optional Tensor (batch_size, Hc, Wc) — valid region mask
        canvas_size: tuple (Hc, Wc) — if None, inferred from target_images
        dt: float — time step

    Returns:
        loss: Tensor (scalar) — mean squared error between reconstruction and target
    """
    if canvas_size is None:
        canvas_size = target_images.shape[-2:]

    batch_size = eye_velocities.shape[0]
    total_loss = 0.0

    for i in range(batch_size):
        # Reconstruct
        G = reconstruct_from_fullmodel_output(
            eye_velocities,
            reconstructed_frames,
            initial_positions,
            canvas_size,
            dt,
            batch_idx=i,
        )

        # Compute MSE
        target = target_images[i].unsqueeze(0).unsqueeze(0)  # (1, 1, Hc, Wc)

        if mask is not None:
            # Only compute loss on valid regions
            m = mask[i].unsqueeze(0).unsqueeze(0)  # (1, 1, Hc, Wc)
            loss = ((G - target) ** 2 * m).sum() / (m.sum() + 1e-8)
        else:
            loss = ((G - target) ** 2).mean()

        total_loss += loss

    return total_loss / batch_size


def compute_reconstruction_loss_optimized(
    eye_velocities,
    reconstructed_frames,
    initial_positions,
    target_images,
    mask=None,
    canvas_size=None,
    dt=1.0,
):
    """
    Compute reconstruction loss for training (OPTIMIZED VERSION).

    This version uses fully batched operations for better performance.

    Args:
        eye_velocities: Tensor (batch_size, t_out, 2)
        reconstructed_frames: Tensor (batch_size, t_out, N, N)
        initial_positions: Tensor (batch_size, 2)
        target_images: Tensor (batch_size, Hc, Wc) — ground truth images
        mask: Optional Tensor (batch_size, Hc, Wc) — valid region mask
        canvas_size: tuple (Hc, Wc) — if None, inferred from target_images
        dt: float — time step

    Returns:
        loss: Tensor (scalar) — mean squared error between reconstruction and target
    """
    if canvas_size is None:
        canvas_size = target_images.shape[-2:]

    # Reconstruct entire batch at once
    G_batch = reconstruct_batch_optimized(
        eye_velocities,
        reconstructed_frames,
        initial_positions,
        canvas_size,
        dt,
    )  # (batch_size, 1, Hc, Wc)

    # Prepare target with channel dimension
    target = target_images.unsqueeze(1)  # (batch_size, 1, Hc, Wc)

    if mask is not None:
        # Apply mask to loss computation
        m = mask.unsqueeze(1)  # (batch_size, 1, Hc, Wc)
        squared_error = (G_batch - target) ** 2 * m
        # Average over spatial dimensions, then over batch
        loss = squared_error.sum() / (m.sum() + 1e-8)
    else:
        # Standard MSE
        loss = ((G_batch - target) ** 2).mean()

    return loss
