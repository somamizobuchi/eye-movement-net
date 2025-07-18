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
import tkinter.filedialog
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys # <--- IMPORT SYS HERE
from model import Encoder # Assuming your model.py defines this
from utils import rescale # Assuming your utils.py defines this
import diplib as dip
from tqdm import tqdm

def plot_spatial_temporal_kernels_side_by_side(model, save_path=None, title=None, max_kernels=None):
    """
    Plots spatial (encoder) kernels and corresponding temporal kernels side by side.

    Args:
        model: The encoder model.
        save_path: Path to save the plot (optional).
        title: Title for the plot (optional).
        max_kernels: Maximum number of kernels to plot.
    """
    with torch.no_grad():
        temporal_kernels = model.pad_temporal().detach().cpu().numpy()
        spatial_kernels = model.spatial_kernels.detach().cpu().numpy()

    if spatial_kernels.shape[1] == 0:
        print("Spatial kernels have zero size in plot_spatial_temporal_kernels_side_by_side.")
        return
    kernel_size_spatial = int(np.sqrt(spatial_kernels.shape[1]))

    if max_kernels is not None:
        n_kernels = min(spatial_kernels.shape[0], max_kernels)
    else:
        n_kernels = spatial_kernels.shape[0]

    if n_kernels == 0:
        print("No kernels to plot in plot_spatial_temporal_kernels_side_by_side.")
        return


    if n_kernels == 1:
        axes = axes.reshape(1, 2)

    n_rows = int(np.sqrt(n_kernels))
    n_cols = int(np.ceil(n_kernels / n_rows))

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(10, 15))

    cmax = np.max(spatial_kernels)
    cmin = np.min(spatial_kernels)

    ymax = np.max(temporal_kernels)
    ymin = np.min(temporal_kernels)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < n_kernels:
                spatial_kernel = spatial_kernels[idx].reshape(kernel_size_spatial, kernel_size_spatial)
                spatial_kernel = (spatial_kernel - spatial_kernel.min()) / (spatial_kernel.max() - spatial_kernel.min())
                temporal_kernel = temporal_kernels[idx]
                

                # Plot spatial
                axes[i * 2, j].imshow(spatial_kernel, cmap="viridis")

                # Plot temporal
                axes[i * 2 + 1, j].plot(np.flip(temporal_kernel))

            axes[i * 2, j].axis("off")
            axes[i * 2 + 1, j].axis("off")
            axes[i * 2 + 1, j].set_ylim([ymin, ymax])



    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5) # Adjusted h_pad
    plt.subplots_adjust(top=0.95 if title else 0.98, hspace=0.05, wspace=0.05) 

    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    elif n_kernels > 0:
        fig.suptitle("Spatial and Temporal Kernels (Side-by-Side)", fontsize=16, y=0.98)

    # if n_kernels > 0 and im_ref is not None:
        # fig.subplots_adjust(right=0.88) 
        # cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
        # fig.colorbar(im_ref, cax=cbar_ax, label="Normalized Intensity")


    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
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
        config: Full configuration from config.json
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    run_dir = Path(checkpoint_path).parent
    grid_dir = run_dir.parent
    config_path = grid_dir / "config.json"

    if not config_path.exists():
        config_path_alt = run_dir / "config.json" 
        if config_path_alt.exists():
            config_path = config_path_alt
        else: # Try one level up from grid_dir if it's a common project structure
            config_path_grandparent = grid_dir.parent / "config.json"
            if config_path_grandparent.exists():
                config_path = config_path_grandparent
            else:
                raise FileNotFoundError(f"Config file not found at {grid_dir / 'config.json'}, {config_path_alt}, or {config_path_grandparent}")


    with open(config_path, "r") as f:
        config = json.load(f)

    model_fs = config.get("fs", config.get("Fs", 1)) # Allow "Fs" as well, default 1
    model_temporal_pad = config.get("temporal_pad", "valid")

    # Check for necessary keys before initializing Encoder
    required_keys = ["kernel_size", "kernel_length", "n_kernels"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required keys in config.json for model initialization: {', '.join(missing_keys)}")


    model = Encoder(
        config["kernel_size"],
        config["kernel_length"],
        config["n_kernels"],
        model_fs,
        model_temporal_pad,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    params = checkpoint.get("parameters", {})

    return model, params, config


def analyze_grid_search(grid_dir, max_kernels=10, device="cpu"):
    grid_dir_path = Path(grid_dir)

    if not grid_dir_path.exists():
        raise FileNotFoundError(f"Grid directory not found: {grid_dir_path}")

    plots_dir = grid_dir_path / "kernel_plots"
    plots_dir.mkdir(exist_ok=True)

    run_dirs = [d for d in grid_dir_path.glob("run_*") if d.is_dir()]
    if not run_dirs:
        print(f"No 'run_*' directories found in {grid_dir_path}. Trying to find checkpoints directly in subdirectories.")
        # Attempt to find 'final.pt' in any immediate subdirectory if 'run_*' fails
        run_dirs = [d.parent for d in grid_dir_path.glob("*/final.pt") if d.parent.is_dir()]
        if not run_dirs:
            print(f"No model runs or checkpoint files found in {grid_dir_path} or its immediate subdirectories.")
            return
        run_dirs = sorted(list(set(run_dirs))) # Remove duplicates and sort

    print(f"Found {len(run_dirs)} model run(s) in or under {grid_dir_path}")

    summary_path = plots_dir / "parameter_summary.txt"
    with open(summary_path, "w") as summary_file:
        summary_file.write("Grid Search Parameter Summary\n")
        summary_file.write(f"Analyzed: {grid_dir_path.resolve()}\n")
        summary_file.write("==========================\n\n")

    pbar = tqdm(sorted(run_dirs), desc="Processing runs", unit="run")
    for run_dir in pbar:
        run_id = run_dir.name # This will be 'run_XXX' or the parent folder name
        checkpoint_path = run_dir / "final.pt"

        if not checkpoint_path.exists():
            pbar.write(f"  Warning: Checkpoint 'final.pt' not found in {run_dir}, skipping.")
            with open(summary_path, "a") as summary_file:
                summary_file.write(f"Run Directory: {run_dir.name}\n")
                summary_file.write(f"  Status: Checkpoint 'final.pt' not found. Skipped.\n\n")
            continue

        pbar.set_description(f"Processing {run_id}")
        try:
            model, params, config = load_model_from_checkpoint(checkpoint_path, device)
            # fs from config, ensuring it's correctly fetched for temporal spectra
            fs = config.get("fs", config.get("Fs", 1)) 

            params_str_list = []
            if params: 
                for k, v in params.items():
                    if isinstance(v, float):
                        params_str_list.append(f"{k}={v:.3g}")
                    else:
                        params_str_list.append(f"{k}={v}")
            params_str = ", ".join(params_str_list) if params_str_list else "N/A"
            
            # Use a sanitized run_id for filenames if it contains problematic characters
            safe_run_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in run_id)
            title_base = f"{run_id} ({params_str})" if params_str != "N/A" else run_id


            plot_spatial_temporal_kernels_side_by_side(model, save_path=plots_dir / f"{safe_run_id}.png", title=title_base)


            with open(summary_path, "a") as summary_file:
                summary_file.write(f"Run Directory: {run_dir.name}\n")
                summary_file.write(f"  Checkpoint: {checkpoint_path.name}\n")
                summary_file.write(f"  Parameters: {params_str}\n")
                summary_file.write(f"  Config fs: {fs}\n")

            pbar.set_postfix_str(f"Plots saved for {run_id}")

        except Exception as e:
            pbar.write(f"  Error processing {run_dir.name} ({checkpoint_path}): {type(e).__name__}: {str(e)}")
            import traceback
            pbar.write(traceback.format_exc())
            with open(summary_path, "a") as summary_file:
                summary_file.write(f"Run Directory: {run_dir.name}\n")
                summary_file.write(f"  Checkpoint: {checkpoint_path.name}\n")
                summary_file.write(f"  Status: Error during processing - {type(e).__name__}: {str(e)}\n\n")


    print(f"\nAll analysis plots saved to {plots_dir.resolve()}")
    print(f"Parameter summary saved to {summary_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grid search results with combined kernel plots and new visualizations."
    )
    parser.add_argument(
        "--grid_dir", type=str, required=True, help="Path to the grid search directory or a single run directory containing final.pt"
    )
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=10,
        help="Maximum number of kernels to plot per run (default: 10)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (e.g., 'cpu', 'cuda')"
    )

    args = parser.parse_args()

    analyze_grid_search(args.grid_dir, args.max_kernels, args.device)


if __name__ == "__main__":
    main()