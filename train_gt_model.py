import torch
from torch.utils.data import DataLoader
from datasets.recon_dataset import ReconDataset
from GtModel import GtModel
from gt_model_trainer import GtModelTrainer
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
    roi_size = 24  # Spatial kernel size
    drift_samples = 64

    # Model parameters - GtEncoder specific
    kernel_size = roi_size  # 32x32 spatial kernels
    kernel_length = 24  # Temporal kernel length
    spacing = 3.0  # Hexagonal grid spacing in pixels
    ppd = 60.0  # Pixels per degree of visual angle
    fs = 240  # Sampling frequency in Hz
    eccentricity = 0.3  # RGC eccentricity in degrees
    cell_type = "P"  # Parvocellular cells
    decoder_size = 64  # Intermediate decoder size
    pad_start = kernel_length - 1
    total_samples = drift_samples + pad_start

    # Training parameters
    batch_size = 8
    learning_rate = 1e-3
    n_iterations = 100_000
    reconstruction_loss_weight = 1e2
    position_loss_weight = 0.1

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
    print("Creating GtModel...")
    model = GtModel(
        kernel_size=kernel_size,
        kernel_length=kernel_length,
        spacing=spacing,
        ppd=ppd,
        fs=fs,
        decoder_size=decoder_size,
        eccentricity=eccentricity,
        cell_type=cell_type,
        noise_std=0.00,
        max_velocity=5.0,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Encoder: J = {model.J} channels (2 * {model.J // 2} grid points)")

    start_time = datetime.now().strftime("%y%m%d%H%M")

    # Create trainer
    trainer = GtModelTrainer(
        model=model,
        dataloader=dataloader,
        learning_rate=learning_rate,
        device=device,
        log_dir=f"runs/{start_time}/gt_model/logs",
        log_every=250,
        save_every=100_000,
        checkpoint_dir=f"runs/{start_time}/gt_model/checkpoints",
        reconstruction_loss_weight=reconstruction_loss_weight,
        position_loss_weight=position_loss_weight,
    )

    # Start training
    print(f"Starting training for {n_iterations} iterations...")
    trainer.train(n_iterations)

    # Close trainer
    trainer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
