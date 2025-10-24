from encoder import Encoder
from trainer import Trainer
from datasets import VideoDataset, ReconDataset
from torch.utils.data import DataLoader
from datetime import datetime

if __name__ == "__main__":
    fs = 1000
    kernel_size = 16
    kernel_length = 16
    kernel_delay = 1
    n_channels = 64
    decoder_size = 256
    noise_std = 0.1

    batch_size = 8

    log_every = 1000
    save_every = 100_000

    reconstruct = True

    # Initialize your model
    encoder = Encoder(
        kernel_size=kernel_size,  # n
        kernel_length=kernel_length,  # T
        kernel_delay=kernel_delay,
        n_channels=n_channels,  # J
        decoder_size=decoder_size,  # K
        noise_std=noise_std,
    )

    dataset = (
        ReconDataset(
            img_size=32,
            roi_size=kernel_size,
            total_samples=kernel_length * 3,
            sampling_frequency=fs,
            diffusion_coefficient=20 / 3600,
            pixels_per_degree=240,
            pad_start=kernel_length - 1,
            average=False,
        )
        if reconstruct
        else VideoDataset(
            "data/bm_fixation_videos.npy",
            kernel_size,
            kernel_length * 2 - 1,
        )
    )

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    start_time = datetime.now().strftime("%y%m%d%H%M")

    # Initialize trainer
    trainer = Trainer(
        encoder,
        dataloader,
        learning_rate=1e-3,
        device="cpu",
        log_dir=f"runs/{start_time}/logs",
        log_every=log_every,
        save_every=save_every,
        checkpoint_dir=f"runs/{start_time}/checkpoints",
        l2_reg_weight=5e-1,
        spatial_var_weight=1e-4,
        reconstruct=reconstruct,
    )

    # Train the model
    trainer.train(n_iterations=500_000)
