#!/usr/bin/env python
"""
Script to plot spatial and temporal kernels of the encoder model side by side.

Usage:
    python plot_encoder_kernels.py --logdir=./logs/experiment_name
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import fire
from pathlib import Path
from encoder import Encoder


def load_model_from_logdir(logdir, device="cpu"):
    """
    Load encoder model from a log directory containing model checkpoint.

    Args:
        logdir: Path to the log directory containing model files
        device: Device to load the model on

    Returns:
        model: Loaded encoder model
    """
    logdir_path = Path(logdir)

    # Look for common checkpoint filenames
    checkpoint_names = ["model.pt", "final.pt", "best_model.pt", "checkpoint.pt"]
    checkpoint_path = None

    for name in checkpoint_names:
        potential_path = logdir_path / name
        if potential_path.exists():
            checkpoint_path = potential_path
            break

    if checkpoint_path is None:
        # Look for any .pt file
        pt_files = list(logdir_path.glob("*.pt"))
        if pt_files:
            checkpoint_path = pt_files[0]
        else:
            raise FileNotFoundError(f"No checkpoint file found in {logdir}")

    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Infer model parameters from state dict
    spatial_kernels_shape = state_dict["spatial_kernels"].shape
    temporal_kernels_shape = state_dict["temporal_kernels"].shape

    n_channels = spatial_kernels_shape[0]
    kernel_size = spatial_kernels_shape[1]
    kernel_length = temporal_kernels_shape[1]

    # Get decoder sizes from linear layer weights
    decoder1_weight_shape = state_dict["decoder1.weight"].shape
    decoder2_weight_shape = state_dict["decoder2.weight"].shape

    decoder_size = decoder1_weight_shape[0]

    # Estimate kernel_delay (default to 0 if not obvious)
    kernel_delay = 0

    # Create model
    model = Encoder(
        kernel_size=kernel_size,
        kernel_length=kernel_length,
        kernel_delay=kernel_delay,
        n_channels=n_channels,
        decoder_size=decoder_size,
    ).to(device)

    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    model.kernel_delay = 1

    return model


def plot_spatial_temporal_kernels(model, save_path=None, max_kernels=None):
    """
    Plot spatial and temporal kernels side by side for each J kernel.
    Each row shows one kernel: spatial on left, temporal on right.

    Args:
        model: The encoder model
        save_path: Path to save the plot (optional)
        max_kernels: Maximum number of kernels to plot
    """
    with torch.no_grad():
        spatial_kernels = model.spatial_kernels.detach().cpu().numpy()
        temporal_kernels = model.get_temporal_kernels().detach().cpu().numpy()

    n_kernels = spatial_kernels.shape[0]

    if max_kernels is not None:
        n_kernels = min(n_kernels, max_kernels)

    if n_kernels == 0:
        print("No kernels to plot.")
        return

    # Create figure with 2 columns (spatial, temporal) and n_kernels rows
    fig, axes = plt.subplots(n_kernels, 2, figsize=(8, 2 * n_kernels))

    # Handle case with single kernel
    if n_kernels == 1:
        axes = axes.reshape(1, 2)

    # Get global min/max for consistent scaling
    spatial_min, spatial_max = spatial_kernels.min(), spatial_kernels.max()
    temporal_min, temporal_max = temporal_kernels.min(), temporal_kernels.max()

    for i in range(n_kernels):
        # Plot spatial kernel (left column)
        spatial_ax = axes[i, 0]
        spatial_kernel = spatial_kernels[i]
        
        spatial_ax.imshow(spatial_kernel, cmap="RdBu_r", vmin=spatial_min, vmax=spatial_max)
        spatial_ax.set_title(f"Spatial Kernel {i+1}", fontsize=10)
        spatial_ax.axis("off")

        # Plot temporal kernel (right column)
        temporal_ax = axes[i, 1]
        temporal_kernel = temporal_kernels[i]

        temporal_ax.plot(temporal_kernel, "b-", linewidth=2)
        temporal_ax.set_title(f"Temporal Kernel {i+1}", fontsize=10)
        temporal_ax.set_ylim(temporal_min, temporal_max)
        temporal_ax.grid(True, alpha=0.3)
        temporal_ax.set_xlabel("Time")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def main(logdir, max_kernels=None, save_path=None, device="cpu"):
    """
    Main function to load model and plot kernels.

    Args:
        logdir: Path to log directory containing model checkpoint
        max_kernels: Maximum number of kernels to plot (optional)
        save_path: Path to save the plot (optional)
        device: Device to use for loading model
    """
    try:
        # Load model
        model = load_model_from_logdir(logdir, device)

        # Print model info
        print(f"Model loaded successfully!")
        print(f"Number of channels (J): {model.J}")
        print(f"Kernel size: {model.N}x{model.N}")
        print(f"Temporal kernel length: {model.T}")
        print(f"Decoder size (K): {model.K}")

        # Generate save path if not provided
        if save_path is None:
            logdir_name = Path(logdir).name
            save_path = f"encoder_kernels_{logdir_name}.png"

        # Plot kernels
        plot_spatial_temporal_kernels(model, save_path, max_kernels)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    fire.Fire(main)
