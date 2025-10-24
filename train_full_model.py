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
    img_size = 64
    roi_size = 24  # ROI size matching kernel size
    drift_samples = 32

    # Model parameters
    kernel_size = roi_size  # 32x32 spatial kernels
    kernel_length = 24  # Temporal kernel length
    kernel_delay = 1  # Temporal kernel delay
    n_channels = 64  # Number of spatiotemporal channels
    decoder_size = 64  # Intermediate decoder size
    pad_start = kernel_length - 1
    total_samples = drift_samples + pad_start

    # Training parameters
    batch_size = 8
    learning_rate = 1e-3
    n_iterations = 10000
    l2_spatial_weight = 1e-3
    l2_temporal_weight = 1e-3
    temporal_smoothness_weight = 1e-3
    kernel_variance_weight = 1e-3
    reconstruction_loss_weight = 1e3

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
        num_workers=4,
    )

    # Create model
    print("Creating model...")
    model = FullModel(
        kernel_size=kernel_size,
        kernel_length=kernel_length,
        kernel_delay=kernel_delay,
        n_channels=n_channels,
        decoder_size=decoder_size,
        noise_std=0.05,
        max_velocity=50.0,
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
        log_every=100,
        save_every=10000,
        checkpoint_dir=f"runs/{start_time}/checkpoints",
        l2_spatial_weight=l2_spatial_weight,
        l2_temporal_weight=l2_temporal_weight,
        temporal_smoothness_weight=temporal_smoothness_weight,
        kernel_variance_weight=kernel_variance_weight,
        reconstruction_loss_weight=reconstruction_loss_weight,
    )

    # Start training
    print(f"Starting training for {n_iterations} iterations...")
    trainer.train(n_iterations)

    # Close trainer
    trainer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
