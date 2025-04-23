#!/usr/bin/env python
"""
Script to analyze grid search results and visualize model kernels.

This script loads trained models from a grid search run and creates
combined plots showing spatial and temporal kernels together.

Usage:
    python analyze_grid_search_combined.py --grid_dir=./grid_search/20250224-1234 --max_kernels=10
"""

import os
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import Encoder
from utils import rescale
import diplib as dip
from tqdm import tqdm


def radial_profile(data, center=None):
    """
    Calculates the radial profile of a 2D array.

    Args:
        data: 2D numpy array.
        center: Tuple (x, y) indicating the center. If None, defaults to the array center.

    Returns:
        A 1D numpy array representing the radial profile.
    """
    y, x = np.indices((data.shape))
    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    else:
        center = np.array(center)

    r = np.hypot(x - center[0], y - center[1])
    r = r.astype(np.int64)

    radial_sum = np.bincount(r.ravel(), data.ravel())
    radial_count = np.bincount(r.ravel())
    radial_mean = radial_sum / (radial_count + 1e-8)
    return radial_mean


def plot_combined_kernels(model, save_path=None, title=None, max_kernels=None):
    """
    Create a combined plot with spatial, temporal kernels and radial power spectrum side by side in rows.

    Args:
        model: The encoder model
        save_path: Path to save the plot (optional)
        title: Title for the plot (optional)
        max_kernels: Maximum number of kernels to plot
    """
    # Get kernels
    with torch.no_grad():
        temporal_kernels = model.pad_temporal().detach().cpu().numpy()
        spatial_kernels = model.spatial_kernels.detach().cpu().numpy()

    # Determine kernel size
    kernel_size = int(np.sqrt(spatial_kernels.shape[1]))

    # Number of kernels to plot (limit to max_kernels)
    if max_kernels is not None:
        n_kernels = min(spatial_kernels.shape[0], max_kernels)
    else:
        n_kernels = spatial_kernels.shape[0]

    # Create figure with tighter spacing
    fig, axes = plt.subplots(n_kernels, 3, figsize=(18, 1.5 * n_kernels))

    # Ensure axes is 2D even with only one kernel
    if n_kernels == 1:
        axes = axes.reshape(1, 3)

    # Plot each kernel pair (spatial and temporal)
    for i in range(n_kernels):
        # Reshape the spatial kernel into 2D
        spatial_kernel = spatial_kernels[i].reshape(kernel_size, kernel_size)

        # Normalize for visualization
        spatial_kernel = (spatial_kernel - spatial_kernel.min()) / (
            spatial_kernel.max() - spatial_kernel.min() + 1e-8
        )

        # Plot spatial kernel on the left
        im = axes[i, 0].imshow(spatial_kernel, cmap="viridis")
        axes[i, 0].set_title(f"Spatial Kernel {i + 1}")
        axes[i, 0].axis("off")

        # Plot temporal kernel in the middle
        axes[i, 1].plot(temporal_kernels[i])
        axes[i, 1].set_title(f"Temporal Kernel {i + 1}")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlabel("Time Steps")
        axes[i, 1].set_ylabel("Kernel Value")

        # Calculate and plot radial power spectrum on the right
        power_spectrum = (
            np.abs(
                np.fft.fftshift(
                    np.fft.fft2(spatial_kernel - spatial_kernel.mean())
                    / spatial_kernel.size
                )
            )
            ** 2
        )
        # radial_ps = radial_profile(power_spectrum)
        radial_ps = dip.RadialMean(power_spectrum, binSize=1)
        axes[i, 2].plot(10.0 * np.log10(radial_ps))
        # axes[i, 2].imshow(10.0 * np.log10(power_spectrum), cmap="viridis")
        axes[i, 2].set_title(f"Radial Power Spectrum {i + 1}")
        axes[i, 2].set_xscale("log")
        axes[i, 2].axis("tight")
        axes[i, 2].set_ylim(-60, axes[i, 2].get_ylim()[1])
        axes[i, 2].set_xlim(axes[i, 2].get_xlim()[0], 10)
        # axes[i, 2].set_xlabel("Radial Frequency")
        # axes[i, 2].set_ylabel("Power")
        # axes[i, 2].grid(True, alpha=0.3)

    # Tighter layout with reduced spacing
    plt.tight_layout(pad=0.3, h_pad=0.5, w_pad=0)
    plt.subplots_adjust(right=0.9, hspace=0.1, wspace=0.3)

    # Add a colorbar for the spatial kernels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Set main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    else:
        fig.suptitle("Spatial and Temporal Kernels", fontsize=16, y=0.98)

    # Add just enough spacing at the top for the title
    plt.subplots_adjust(top=0.94)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """
    Load model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        model: Loaded model
        params: Parameters used for training
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config file to get model parameters
    grid_dir = Path(checkpoint_path).parent.parent
    config_path = grid_dir / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize model with the same parameters
    model = Encoder(
        config["kernel_size"],
        config["kernel_length"],
        config["n_kernels"],
        config["fs"],
        config["temporal_pad"],
    ).to(device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get the parameters used for this run
    params = checkpoint.get("parameters", {})

    return model, params


def analyze_grid_search(grid_dir, max_kernels=10, device="cpu"):
    """
    Analyze grid search results and create combined plots for each run.

    Args:
        grid_dir: Path to the grid search directory
        max_kernels: Maximum number of kernels to plot per run
        device: Device to load models on
    """
    grid_dir = Path(grid_dir)

    # Make sure the grid directory exists
    if not grid_dir.exists():
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")

    # Create a directory for the plots
    plots_dir = grid_dir / "kernel_plots"
    plots_dir.mkdir(exist_ok=True)

    # Get all run directories
    run_dirs = [d for d in grid_dir.glob("run_*") if d.is_dir()]

    print(f"Found {len(run_dirs)} model runs in {grid_dir}")

    # Create a summary file for easy reference
    summary_path = plots_dir / "parameter_summary.txt"
    with open(summary_path, "w") as summary_file:
        summary_file.write("Grid Search Parameter Summary\n")
        summary_file.write("==========================\n\n")

    # Process each run
    pbar = tqdm(sorted(run_dirs), desc="Processing runs", unit="run")
    for run_dir in pbar:
        run_id = run_dir.name
        # print(f"Processing {run_id}...")

        # Find the checkpoint file
        checkpoint_path = run_dir / "final.pt"

        if not checkpoint_path.exists():
            print(f"  Warning: Checkpoint not found in {run_dir}")
            continue

        try:
            # Load the model
            model, params = load_model_from_checkpoint(checkpoint_path, device)

            # Create a parameter string for the title
            params_str = ", ".join([f"{k}={v:.3g}" for k, v in params.items()])
            title = f"Run {run_id}: {params_str}"

            # Plot and save combined kernels
            combined_save_path = plots_dir / f"{run_id}_combined.png"
            plot_combined_kernels(
                model,
                save_path=combined_save_path,
                title=title,
                max_kernels=max_kernels,
            )

            # Add to summary file
            with open(summary_path, "a") as summary_file:
                summary_file.write(f"Run: {run_id}\n")
                summary_file.write(f"Parameters: {params_str}\n")
                summary_file.write(f"Plot: {run_id}_combined.png\n\n")

            # print(f"  Saved combined plot to {combined_save_path}")
            pbar.set_postfix_str(f"Saved to {combined_save_path}")

        except Exception as e:
            print(f"  Error processing {run_id}: {str(e)}")

    print(f"All plots saved to {plots_dir}")
    print(f"Parameter summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grid search results with combined kernel plots"
    )
    parser.add_argument(
        "--grid_dir", type=str, required=True, help="Path to the grid search directory"
    )
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=10,
        help="Maximum number of kernels to plot per run",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (e.g., 'cpu', 'cuda')"
    )

    args = parser.parse_args()

    # Analyze grid search and create combined plots
    analyze_grid_search(args.grid_dir, args.max_kernels, args.device)


if __name__ == "__main__":
    main()
