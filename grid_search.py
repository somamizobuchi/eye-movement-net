import torch
import itertools
import json
from torch.utils.data import DataLoader
from datasets.recon_dataset import ReconDataset
from eye_trace_encoder import EyeTraceEncoder
from eye_trace_trainer import EyeTraceTrainer
from datetime import datetime
import os


def run_experiment(params, experiment_name, base_dir="experiments"):
    """Run a single training experiment with given hyperparameters."""

    # Device configuration
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nStarting experiment: {experiment_name}")
    print(f"Using device: {device}")
    print(f"Parameters: {params}")

    # Dataset parameters
    img_size = 256
    roi_size = 16
    total_samples = 128

    # Model parameters
    kernel_size = roi_size
    kernel_length = 16
    kernel_delay = 1
    n_channels = 64
    decoder_size = 32
    pad_start = kernel_length - 1

    # Training parameters (fixed)
    batch_size = 8
    learning_rate = 1e-3
    n_iterations = 75000
    velocity_loss_weight = 1.0

    # Create experiment directory
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment configuration to JSON
    config = {
        "hyperparameters": params,
        "dataset": {
            "img_size": img_size,
            "roi_size": roi_size,
            "total_samples": total_samples,
            "saccade": False,
        },
        "model": {
            "kernel_size": roi_size,
            "kernel_length": kernel_length,
            "kernel_delay": kernel_delay,
            "n_channels": n_channels,
            "decoder_size": decoder_size,
            "noise_std": 0.001,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "velocity_loss_weight": velocity_loss_weight,
        },
    }

    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Create dataset
    print("Creating dataset...")
    dataset = ReconDataset(
        img_size=img_size,
        roi_size=roi_size,
        total_samples=total_samples,
        pad_start=pad_start,
        saccade=False,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
    )

    # Create encoder model
    print("Creating model...")
    encoder = EyeTraceEncoder(
        kernel_size=kernel_size,
        kernel_length=kernel_length,
        kernel_delay=kernel_delay,
        n_channels=n_channels,
        decoder_size=decoder_size,
        noise_std=config["model"]["noise_std"],
    )

    print(
        f"Model created with {sum(p.numel() for p in encoder.parameters())} parameters"
    )

    # Create trainer
    trainer = EyeTraceTrainer(
        encoder=encoder,
        dataloader=dataloader,
        learning_rate=learning_rate,
        device=device,
        log_dir=experiment_dir,  # Everything goes in experiment_dir
        log_every=5000,
        save_every=100000,
        checkpoint_dir=experiment_dir,  # Checkpoints in same dir
        l2_spatial_weight=params["l2_spatial_weight"],
        l2_temporal_weight=params["l2_temporal_weight"],
        temporal_smoothness_weight=params["temporal_smoothness_weight"],
        kernel_variance_weight=params["kernel_variance_weight"],
        velocity_loss_weight=velocity_loss_weight,
    )

    # Start training
    print(f"Starting training for {n_iterations} iterations...")
    trainer.train(n_iterations)

    # Close trainer
    trainer.close()
    print(f"Experiment {experiment_name} completed!")

    return experiment_dir


def main():
    """Run grid search over hyperparameters."""

    # Define hyperparameter grid
    param_grid = {
        "l2_spatial_weight": [0.001, 0.01, 0.1],
        "l2_temporal_weight": [0.001, 0.01, 0.1],
        "temporal_smoothness_weight": [0.001, 0.01, 0.1],
        "kernel_variance_weight": [0.001],  # Fixed for now
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"=" * 80)
    print(f"GRID SEARCH: Running {len(combinations)} experiments")
    print(f"=" * 80)

    # Create base experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/grid_search_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    # Save grid search configuration
    config_path = os.path.join(base_dir, "grid_config.txt")
    with open(config_path, "w") as f:
        f.write("Grid Search Configuration\n")
        f.write("=" * 50 + "\n\n")
        for key, values in param_grid.items():
            f.write(f"{key}: {values}\n")
        f.write(f"\nTotal combinations: {len(combinations)}\n")

    # Run all experiments
    completed_experiments = []
    for i, params in enumerate(combinations):
        # Create simple experiment name
        experiment_name = f"exp_{i:03d}"

        try:
            exp_dir = run_experiment(params, experiment_name, base_dir)
            completed_experiments.append((experiment_name, params, exp_dir))
            print(f"\n✓ Completed {i+1}/{len(combinations)}")
        except Exception as e:
            print(f"\n✗ Experiment {experiment_name} failed with error: {e}")
            continue

    # Save summary
    summary_path = os.path.join(base_dir, "experiments_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Grid Search Results Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"Completed: {len(completed_experiments)}/{len(combinations)} experiments\n\n"
        )

        for exp_name, params, exp_dir in completed_experiments:
            f.write(f"\n{exp_name}:\n")
            for key, val in params.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"  Directory: {exp_dir}\n")

    print("\n" + "=" * 80)
    print(f"GRID SEARCH COMPLETED!")
    print(f"Results saved in: {base_dir}")
    print(f"View in TensorBoard: tensorboard --logdir {base_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
