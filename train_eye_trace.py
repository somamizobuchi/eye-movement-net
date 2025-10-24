import torch
from torch.utils.data import DataLoader
from datasets.recon_dataset import ReconDataset
from eye_trace_encoder import EyeTraceEncoder
from eye_trace_trainer import EyeTraceTrainer
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
    img_size = 256
    roi_size = 16
    total_samples = 128

    # Model parameters
    kernel_size = roi_size  # 32x32 spatial kernels
    kernel_length = 16  # Temporal kernel length
    kernel_delay = 1  # Temporal kernel delay
    n_channels = 64  # Number of spatiotemporal channels
    decoder_size = 32  # Intermediate decoder size
    pad_start = kernel_length - 1

    # Training parameters
    batch_size = 8
    learning_rate = 1e-3
    n_iterations = 100000
    l2_spatial_weight = 0.001
    l2_temporal_weight = 0.01
    temporal_smoothness_weight = 0.001
    kernel_variance_weight = 0.0001
    velocity_loss_weight = 1.0

    # Create dataset
    print("Creating dataset...")
    dataset = ReconDataset(
        img_size=img_size,
        roi_size=roi_size,
        total_samples=total_samples,
        pad_start=pad_start,
        saccade=False,  # Include saccades in eye movements
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
        noise_std=0.001,
    )

    print(
        f"Model created with {sum(p.numel() for p in encoder.parameters())} parameters"
    )

    start_time = datetime.now().strftime("%y%m%d%H%M")

    # Create trainer
    trainer = EyeTraceTrainer(
        encoder=encoder,
        dataloader=dataloader,
        learning_rate=learning_rate,
        device=device,
        log_dir=f"runs/{start_time}/logs",
        log_every=1000,
        save_every=100000,
        checkpoint_dir=f"runs/{start_time}/checkpoints",
        l2_spatial_weight=l2_spatial_weight,
        l2_temporal_weight=l2_temporal_weight,
        temporal_smoothness_weight=temporal_smoothness_weight,
        kernel_variance_weight=kernel_variance_weight,
        velocity_loss_weight=velocity_loss_weight,
    )

    # Start training
    print(f"Starting training for {n_iterations} iterations...")
    trainer.train(n_iterations)

    # Close trainer
    trainer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
