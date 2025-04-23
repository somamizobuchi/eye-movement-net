from analyze_grid_search import plot_combined_kernels
import os
import json
from pathlib import Path
import torch
from model import Encoder


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
    grid_dir = Path(checkpoint_path).parent
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


if __name__ == "__main__":
    # checkpoint_path = Path("./checkpoints/20250415-1740/499999.pt")
    checkpoint_path = Path("./checkpoints/20250415-1622/551863.pt")
    model, params = load_model_from_checkpoint(checkpoint_path, device="cpu")

    image_path = checkpoint_path.parent / "kernels.png"

    plot_combined_kernels(model, save_path=image_path)
