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
import sys # <--- IMPORT SYS HERE
from model import Encoder # Assuming your model.py defines this
from utils import rescale # Assuming your utils.py defines this
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
    radial_mean = radial_sum / (radial_count + 1e-8) # Add epsilon to avoid division by zero
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

    if n_kernels == 0:
        print("No kernels to plot in plot_combined_kernels.")
        return

    # Create figure with tighter spacing
    fig, axes = plt.subplots(n_kernels, 3, figsize=(18, 2.5 * n_kernels)) # Increased height per kernel

    # Ensure axes is 2D even with only one kernel
    if n_kernels == 1:
        axes = axes.reshape(1, 3)

    im_ref = None # For colorbar reference

    # Plot each kernel pair (spatial and temporal)
    for i in range(n_kernels):
        # Reshape the spatial kernel into 2D
        spatial_kernel = spatial_kernels[i].reshape(kernel_size, kernel_size)

        # Normalize for visualization
        spatial_kernel_min = spatial_kernel.min()
        spatial_kernel_max = spatial_kernel.max()
        spatial_kernel_norm = (spatial_kernel - spatial_kernel_min) / (
            spatial_kernel_max - spatial_kernel_min + 1e-8
        )

        # Plot spatial kernel on the left
        im = axes[i, 0].imshow(spatial_kernel_norm, cmap="viridis")
        if i == 0: im_ref = im # Store reference for colorbar
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
        
        radial_ps_data = []
        try:
            # dip.RadialMean can return an empty image if input is too small or all zero
            radial_ps_img = dip.RadialMean(power_spectrum, binSize=1) # Returns a diplib image
            if radial_ps_img.IsForged() and np.prod(radial_ps_img.Sizes()) > 0:
                 radial_ps_data = np.array(radial_ps_img).squeeze() # Convert to numpy array
            
            if len(radial_ps_data) > 1: 
                 axes[i, 2].plot(10.0 * np.log10(radial_ps_data + 1e-20)) # Add epsilon for log
            else:
                axes[i,2].text(0.5, 0.5, "Not enough data\nfor radial mean", ha='center', va='center', fontsize=9)
        except Exception as e:
            print(f"Error in dip.RadialMean for kernel {i+1}: {e}")
            axes[i,2].text(0.5, 0.5, "RadialMean Error", ha='center', va='center', fontsize=9)


        axes[i, 2].set_title(f"Radial Power Spectrum {i + 1} (dB)")
        axes[i, 2].set_xscale("log")
        axes[i, 2].axis("tight")
        if len(radial_ps_data) > 1:
            axes[i, 2].set_ylim(-60, max(0, axes[i, 2].get_ylim()[1] if axes[i, 2].get_ylim()[1] > -60 else 0))
            current_xlim = axes[i, 2].get_xlim()
            axes[i, 2].set_xlim(max(1, current_xlim[0] if current_xlim[0] > 0 else 1), 
                                min(kernel_size // 2 if kernel_size//2 > 0 else 10, 
                                    current_xlim[1] if current_xlim[1] > 0 else kernel_size//2))

    # Tighter layout with reduced spacing
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5) # Adjusted padding
    plt.subplots_adjust(right=0.9, top=0.95 if title else 0.98) # Adjusted top for suptitle

    # Add a colorbar for the spatial kernels
    if n_kernels > 0 and im_ref is not None: # Only add colorbar if there are plots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(im_ref, cax=cbar_ax, label="Normalized Intensity")

    # Set main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    elif n_kernels > 0 : # Only add suptitle if there is a figure to add to
        fig.suptitle("Spatial, Temporal Kernels & Radial Spectra", fontsize=16, y=0.98)


    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig) # Close the figure after saving
    else:
        plt.show()


def plot_radial_spectra_overlay(model, save_path=None, title=None, max_kernels=None):
    """
    Create a plot with all radial power spectra overlaid for comparison.

    Args:
        model: The encoder model
        save_path: Path to save the plot (optional)
        title: Title for the plot (optional)
        max_kernels: Maximum number of kernels to plot
    """
    with torch.no_grad():
        spatial_kernels = model.spatial_kernels.detach().cpu().numpy()

    if spatial_kernels.shape[1] == 0: # No spatial dimension
        print("Spatial kernels have zero size in plot_radial_spectra_overlay.")
        return
    kernel_size = int(np.sqrt(spatial_kernels.shape[1]))
    if max_kernels is not None:
        n_kernels = min(spatial_kernels.shape[0], max_kernels)
    else:
        n_kernels = spatial_kernels.shape[0]

    if n_kernels == 0:
        print("No kernels to plot in plot_radial_spectra_overlay.")
        return

    plt.figure(figsize=(12, 7)) # Adjusted figure size
    colors = plt.cm.viridis(np.linspace(0, 1, n_kernels))

    max_freq_len = 0 
    plotted_anything = False

    for i in range(n_kernels):
        spatial_kernel = spatial_kernels[i].reshape(kernel_size, kernel_size)
        spatial_kernel_min = spatial_kernel.min()
        spatial_kernel_max = spatial_kernel.max()
        spatial_kernel_norm = (spatial_kernel - spatial_kernel_min) / (
            spatial_kernel_max - spatial_kernel_min + 1e-8
        )
        power_spectrum = (
            np.abs(
                np.fft.fftshift(
                    np.fft.fft2(spatial_kernel_norm - spatial_kernel_norm.mean())
                    / spatial_kernel_norm.size
                )
            )
            ** 2
        )
        try:
            radial_ps_img = dip.RadialMean(power_spectrum, binSize=1)
            radial_ps_data = []
            if radial_ps_img.IsForged() and np.prod(radial_ps_img.Sizes()) > 0:
                radial_ps_data = np.array(radial_ps_img).squeeze()

            if len(radial_ps_data) > 1:
                max_freq_len = max(max_freq_len, len(radial_ps_data))
                plt.plot(10.0 * np.log10(radial_ps_data + 1e-20), color=colors[i], label=f"Kernel {i+1}", alpha=0.8)
                plotted_anything = True
            else:
                print(f"Skipping radial_ps for overlay kernel {i+1} due to insufficient data after RadialMean.")
        except Exception as e:
            print(f"Error in dip.RadialMean for overlay kernel {i+1}: {e}")

    if not plotted_anything:
        print("No radial spectra were plotted in overlay.")
        plt.close()
        return

    plt.title(title if title else "Overlay of Radial Power Spectra")
    plt.xlabel("Radial Frequency (pixels)") 
    plt.ylabel("Power (dB)")
    plt.xscale("log")

    plt.xlim(1, kernel_size // 2 if kernel_size // 2 > 1 else max(2,max_freq_len))
    plt.ylim(-60, 0) 
    plt.grid(True, alpha=0.4, which="both", linestyle='--') 
    if n_kernels > 0 and n_kernels <= 20 : 
        plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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

    fig, axes = plt.subplots(n_kernels, 2, figsize=(12, 2.5 * n_kernels))
    im_ref = None # For colorbar

    if n_kernels == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_kernels):
        # Spatial Kernel
        spatial_kernel = spatial_kernels[i].reshape(kernel_size_spatial, kernel_size_spatial)
        spatial_kernel_min = spatial_kernel.min()
        spatial_kernel_max = spatial_kernel.max()
        spatial_kernel_norm = (spatial_kernel - spatial_kernel_min) / (
            spatial_kernel_max - spatial_kernel_min + 1e-8
        )
        im = axes[i, 0].imshow(spatial_kernel_norm, cmap="viridis")
        if i == 0 : im_ref = im
        axes[i, 0].set_title(f"Spatial Kernel {i + 1}")
        axes[i, 0].axis("off")

        # Temporal Kernel
        axes[i, 1].plot(temporal_kernels[i])
        axes[i, 1].set_title(f"Temporal Kernel {i + 1}")
        axes[i, 1].set_xlabel("Time Steps")
        axes[i, 1].set_ylabel("Kernel Value")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5) # Adjusted h_pad
    plt.subplots_adjust(top=0.95 if title else 0.98) 

    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    elif n_kernels > 0:
        fig.suptitle("Spatial and Temporal Kernels (Side-by-Side)", fontsize=16, y=0.98)

    if n_kernels > 0 and im_ref is not None:
        fig.subplots_adjust(right=0.88) 
        cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
        fig.colorbar(im_ref, cax=cbar_ax, label="Normalized Intensity")


    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_temporal_spectra_overlay(model, fs, save_path=None, title=None, max_kernels=None):
    """
    Plots 1-sided power spectrum of the temporal kernels, overlaid.

    Args:
        model: The encoder model.
        fs: Sampling frequency of the temporal data.
        save_path: Path to save the plot (optional).
        title: Title for the plot (optional).
        max_kernels: Maximum number of kernels to plot.
    """
    with torch.no_grad():
        temporal_kernels = model.pad_temporal().detach().cpu().numpy() 

    if max_kernels is not None:
        n_kernels = min(temporal_kernels.shape[0], max_kernels)
    else:
        n_kernels = temporal_kernels.shape[0]

    if n_kernels == 0:
        print("No kernels to plot in plot_temporal_spectra_overlay.")
        return

    plt.figure(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0, 1, n_kernels))
    plotted_anything = False
    max_power_val = -np.inf


    for i in range(n_kernels):
        kernel = temporal_kernels[i]
        kernel_length = len(kernel)

        if kernel_length == 0:
            print(f"Temporal kernel {i+1} has zero length, skipping.")
            continue

        fft_values = np.fft.rfft(kernel - np.mean(kernel))
        power_spectrum = np.abs(fft_values)**2 / kernel_length 
        frequencies = np.fft.rfftfreq(kernel_length)
        
        db_power = 10 * np.log10(power_spectrum + 1e-20)
        max_power_val = max(max_power_val, np.max(db_power))


        plt.plot(frequencies, db_power, color=colors[i], label=f"Kernel {i+1}", alpha=0.8)
        plotted_anything = True

    if not plotted_anything:
        print("No temporal spectra were plotted in overlay.")
        plt.close()
        return

    plt.title(title if title else "Overlay of Temporal Kernel Power Spectra")
    plt.xlabel(f"Frequency (normalized)")
    plt.ylabel("Power (dB)")
    plt.grid(True, alpha=0.4, which="both", linestyle='--')
    if n_kernels > 0 and n_kernels <= 20:
        plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    # plt.xlim(0, fs / 2) 
    plt.ylim(-10, max(0, max_power_val + 1) if max_power_val > -np.inf else 10 ) # Dynamic upper y-limit, ensure non-negative range
    plt.xscale("log")


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

    plots_dir = grid_dir_path / "kernel_plots_analysis"
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


            # # 1. Combined kernels plot (original)
            # combined_save_path = plots_dir / f"{safe_run_id}_kernels_combined.png"
            # plot_combined_kernels(
            #     model,
            #     save_path=combined_save_path,
            #     title=f"Combined Kernels & Spectra - {title_base}",
            #     max_kernels=max_kernels,
            # )

            # # 2. Radial spectra overlay plot (original)
            # radial_overlay_save_path = plots_dir / f"{safe_run_id}_radial_spectra_overlay.png"
            # plot_radial_spectra_overlay(
            #     model,
            #     save_path=radial_overlay_save_path,
            #     title=f"Radial Spectra Overlay - {title_base}",
            #     max_kernels=max_kernels,
            # )

            # # 3. NEW: Spatial and Temporal Kernels Side-by-Side
            # st_side_by_side_save_path = plots_dir / f"{safe_run_id}_spatial_temporal_sidebyside.png"
            # plot_spatial_temporal_kernels_side_by_side(
            #     model,
            #     save_path=st_side_by_side_save_path,
            #     title=f"Spatial & Temporal Kernels - {title_base}",
            #     max_kernels=max_kernels
            # )

            # 4. NEW: Temporal Spectra Overlay
            temporal_spectra_overlay_save_path = plots_dir / f"{safe_run_id}_temporal_spectra_overlay.png"
            plot_temporal_spectra_overlay(
                model,
                fs=fs, 
                save_path=temporal_spectra_overlay_save_path,
                title=f"Temporal Spectra Overlay (Fs={fs}Hz) - {title_base}",
                max_kernels=max_kernels
            )

            with open(summary_path, "a") as summary_file:
                summary_file.write(f"Run Directory: {run_dir.name}\n")
                summary_file.write(f"  Checkpoint: {checkpoint_path.name}\n")
                summary_file.write(f"  Parameters: {params_str}\n")
                summary_file.write(f"  Config fs: {fs}\n")
                summary_file.write(f"  Combined Kernels Plot: {combined_save_path.name}\n")
                summary_file.write(f"  Radial Overlay Plot: {radial_overlay_save_path.name}\n")
                summary_file.write(f"  Spatial-Temporal Side-by-Side Plot: {st_side_by_side_save_path.name}\n")
                summary_file.write(f"  Temporal Spectra Overlay Plot: {temporal_spectra_overlay_save_path.name}\n\n")

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