import torch
from torch.utils.data import DataLoader
from datasets.recon_dataset import ReconDataset
from FullModel import FullModel
from full_model_trainer import FullModelTrainer
from datetime import datetime


def main():
    # Device configuration
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset parameters
    img_size = 48
    roi_size = 20  # ROI size matching kernel size
    drift_samples = 16

    # Model parameters
    kernel_size = roi_size  # 32x32 spatial kernels
    kernel_length = 20  # Temporal kernel length
    kernel_delay = 1  # Temporal kernel delay
    n_channels = 64  # Number of spatiotemporal channels
    decoder_size = 400  # Intermediate decoder size
    velocity_hidden_channels = 64  # Hidden channels for velocity decoder
    pad_start = kernel_length * 2 - 2
    total_samples = drift_samples + pad_start

    # Training parameters
    batch_size = 8
    learning_rate = 1e-3
    n_iterations = 100_000
    l2_spatial_weight = 1e-2
    l2_temporal_weight = 1e-2
    temporal_smoothness_weight = 1e-3
    kernel_variance_weight = 1e-4
    reconstruction_loss_weight = 1.0
    position_loss_weight = 1.0

    # Create dataset
    print("Creating dataset...")
    dataset = ReconDataset(
        img_size=img_size,
        roi_size=roi_size,
        total_samples=total_samples,
        pad_start=pad_start,
        saccade=False,  # No saccades for smooth trajectories
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    # Create model
    print("Creating model...")
    model = FullModel(
        kernel_size=kernel_size,
        kernel_length=kernel_length,
        kernel_delay=kernel_delay,
        n_channels=n_channels,
        decoder_size=decoder_size,
        velocity_hidden_channels=velocity_hidden_channels,
        noise_std=0.00,
        max_velocity=5.0,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    start_time = datetime.now().strftime("%y%m%d%H%M")

    # Create trainer
    trainer = FullModelTrainer(
        model=model,
        dataloader=dataloader,
        learning_rate=learning_rate,
        device=device,
        log_dir=f"runs/{start_time}/logs",
        log_every=250,
        save_every=100_000,
        checkpoint_dir=f"runs/{start_time}/checkpoints",
        l2_spatial_weight=l2_spatial_weight,
        l2_temporal_weight=l2_temporal_weight,
        temporal_smoothness_weight=temporal_smoothness_weight,
        kernel_variance_weight=kernel_variance_weight,
        reconstruction_loss_weight=reconstruction_loss_weight,
        position_loss_weight=position_loss_weight,
        balanced_losses=[
            "reconstruction",
            "position",
            # "spatial_l2",
            # "temporal_l2",
            # "temporal_smoothness",
            # "kernel_variance",
        ],
    )

    # Start training
    print(f"Starting training for {n_iterations} iterations...")
    trainer.train(n_iterations)

    # Close trainer
    trainer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
