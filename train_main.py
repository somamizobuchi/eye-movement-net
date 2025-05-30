import datetime
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import fire
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from data import FixationDataset, WhitenedDataset, PinkNoise3Dataset
from model import Encoder
from utils import (
    accumulate_frames,
    rescale,
)


@dataclass
class TrainingConfig:
    # Model parameters
    kernel_size: int = 20
    kernel_length: int = 20
    n_kernels: int = 100
    fs: int = 1000  # Hz
    ppd: int = 180.0  # pixels per degree
    diffusion_constant: float = 15.0 / 3600.0  # deg^2/sec
    drift_samples: int = 64
    temporal_pad = (0, 5)

    # Training parameters
    batch_size: int = 16
    total_iterations: int = 4_000_000
    log_iterations: int = 1000
    checkpoint_iterations: int = 50_000

    # Loss weights
    alpha: float = 5e-1  # Temporal jerk energy (smoothness)
    delta: float = 5e-1  # Spatial jerk energy (smoothness)
    beta: float = 1e-1  # Firing rate (encoder output)
    gamma: float = 1e1  # Regularization

    # Checkpoint loading
    load_checkpoint: bool = False
    checkpoint_run: Optional[str] = None
    checkpoint_iteration: Optional[int] = None

    # Device configuration
    device: str = "cpu"


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_paths()
        self.setup_model()
        self.setup_logging()
        self.setup_signal_handlers()

        self.running_metrics = {
            "loss": 0.0,
            "mse": 0.0,
            "jerk_temporal": 0.0,
            "jerk_spatial": 0.0,
            "reg": 0.0,
            "fr": 0.0,
        }

    def setup_paths(self):
        """Initialize paths for checkpoints and logs."""
        if self.config.load_checkpoint:
            if not self.config.checkpoint_run:
                raise ValueError(
                    "checkpoint_run must be specified when load_checkpoint is True"
                )

            self.run_id = self.config.checkpoint_run
            self.checkpoint_dir = Path(f"./checkpoints/{self.run_id}")

            if not self.checkpoint_dir.exists():
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {self.checkpoint_dir}"
                )

            print(f"\nAttempting to load checkpoint from run: {self.run_id}")

            # If checkpoint_iteration is not specified, find the latest
            if self.config.checkpoint_iteration is None:
                _, self.iteration = self.find_latest_checkpoint()
            else:
                checkpoint_path = (
                    self.checkpoint_dir / f"{self.config.checkpoint_iteration}.pt"
                )
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint file not found: {checkpoint_path}"
                    )
                self.iteration = self.config.checkpoint_iteration
        else:
            self.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.checkpoint_dir = Path(f"./checkpoints/{self.run_id}")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.iteration = 0

        # Save configuration
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def setup_model(self):
        """Initialize model, dataset, and optimizer."""
        self.model = Encoder(
            self.config.kernel_size,
            self.config.kernel_length,
            self.config.n_kernels,
            self.config.fs,
            self.config.temporal_pad,
        ).to(self.config.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        if self.config.load_checkpoint:
            if self.config.checkpoint_iteration is None:
                print(f"Checkpoint: {self.config.checkpoint_run}")
                checkpoint_path, _ = self.find_latest_checkpoint()
            else:
                checkpoint_path = (
                    self.checkpoint_dir / f"{self.config.checkpoint_iteration}.pt"
                )

            state = torch.load(
                checkpoint_path, map_location=self.config.device, weights_only=True
            )
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")

        # self.dataset = FixationDataset(
        #     self.config.kernel_size,
        #     self.config.fs,
        #     self.config.ppd,
        #     self.config.drift_samples,
        #     self.config.diffusion_constant,
        # )

        # self.dataset = WhitenedDataset(
        #     # "data/cd02A_patches.npy",
        #     "data/pink_noise.npy",
        #     self.config.fs,
        #     self.config.ppd,
        #     self.config.kernel_size,
        #     self.config.drift_samples,
        #     self.config.diffusion_constant,
        #     self.config.device,
        # )

        self.dataset = PinkNoise3Dataset(
            "data/pink_noise_videos.npy",
            self.config.kernel_size,
            self.config.drift_samples,
        )

        self.data_loader = DataLoader(
            self.dataset, shuffle=True, batch_size=self.config.batch_size
        )

    def setup_logging(self):
        """Initialize TensorBoard logging."""
        self.writer = SummaryWriter(log_dir=f"logs/{self.run_id}")

    def setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals by saving checkpoint and exiting."""
        print("\nReceived shutdown signal. Saving checkpoint...")
        self.save_checkpoint()
        sys.exit(0)

    def find_latest_checkpoint(self) -> tuple[Path, int]:
        """Find the latest checkpoint file in the checkpoint directory."""
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))

        # Parse all checkpoint files to get iteration numbers
        checkpoints = []
        for f in checkpoint_files:
            try:
                iteration = int(f.stem)
                checkpoints.append((f, iteration))
            except ValueError:
                continue

        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        # Sort by iteration number and get the latest
        latest_file, latest_iteration = max(checkpoints, key=lambda x: x[1])
        print(f"Found latest checkpoint: {latest_file} (iteration {latest_iteration})")
        return latest_file, latest_iteration

    def save_checkpoint(self):
        """Save model checkpoint and metrics."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "metrics": self.running_metrics,
        }

        save_path = self.checkpoint_dir / f"{self.iteration}.pt"
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint at iteration {self.iteration}")

    def log_metrics(self):
        """Log metrics to TensorBoard."""
        metrics_count = self.config.log_iterations

        # Log kernels and visualizations
        with torch.no_grad():
            # Temporal kernels
            temporal_kernels = self.model.pad_temporal().detach().cpu()
            self.writer.add_figure(
                "Kernel/temporal",
                self.plot_temporal_kernels(temporal_kernels),
                self.iteration,
            )
            # self.writer.add_figure(
            #     "Kernel/temporal_decoder",
            #     self.plot_temporal_kernels(self.model.temporal_decoder),
            #     self.iteration,
            # )

            # Spatial kernels
            spatial_kernels = rescale(
                self.model.spatial_kernels.transpose(0, -1).reshape(
                    self.config.kernel_size, self.config.kernel_size, 1, -1
                )
            )
            self.writer.add_images(
                "Kernel/spatial", spatial_kernels, self.iteration, dataformats="WHCN"
            )

            # Projective field
            self.writer.add_images(
                "Decoder/spatial",
                rescale(
                    self.model.spatial_decoder.transpose(0, -1).view(
                        self.config.kernel_size,
                        self.config.kernel_size,
                        1,
                        self.config.n_kernels,
                    )
                ),
                self.iteration,
                dataformats="WHCN",
            )

            # Reconstruction
            # self.writer.add_images(
            #     "Reconstruction/spatial",
            #     rescale(
            #         torch.cat((self.current_target, self.current_reconstruction), dim=1)
            #     )[:, None, :, :],
            #     self.iteration,
            #     dataformats="NCHW",
            # )
            self.writer.add_video(
                "Reconstruction",
                rescale(
                    torch.cat(
                        [self.current_target[0], self.current_reconstruction[0]], dim=2
                    )
                )[None, :, None, :, :].expand(-1, -1, 3, -1, -1),
                self.iteration,
                fps=10,
            )

        # Log loss metrics
        for name, value in self.running_metrics.items():
            self.writer.add_scalar(
                f"Loss/{name}", value / metrics_count, self.iteration
            )

        # Reset running metrics
        self.running_metrics = {k: 0.0 for k in self.running_metrics}

    @staticmethod
    def plot_temporal_kernels(kernels):
        """Create temporal kernels plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(kernels.numpy().T)
        return fig

    def train_step(self):
        """Execute single training step."""
        retinal_input = next(iter(self.data_loader))

        self.optimizer.zero_grad()
        out, fr = self.model(retinal_input.clone().to(self.config.device))

        # Current
        self.current_target = retinal_input[:, self.model.kernel_length - 1 :].to(
            self.config.device
        )
        self.current_reconstruction = out

        # Compute losses
        loss_mse = torch.nn.functional.l1_loss(
            self.current_reconstruction, self.current_target
        )
        loss_jerk_temporal = self.config.alpha * self.model.kernel_temporal_jerk()
        loss_jerk_spatial = self.config.delta * self.model.kernel_spatial_jerk()
        loss_fr = self.config.beta * out.abs().mean()
        # loss_fr = self.config.beta * fr.diff(dim=2).abs().mean(dim=2).mean()
        loss_reg = self.config.gamma * (
            self.model.spatial_kernels.square().mean()
            + self.model.temporal_kernels.square().mean()
            + self.model.spatial_decoder.square().mean()
        )
        # Add smoothness constraint for spatial kernels as well
        # loss = loss_mse + loss_jerk + loss_fr + loss_reg
        loss = loss_mse + loss_jerk_spatial

        if torch.isnan(loss):
            print(f"NaN detected in loss at iteration {self.iteration}, skipping.")
            return None

        loss.backward()
        self.optimizer.step()

        # Update metrics
        with torch.no_grad():
            self.running_metrics["loss"] += loss.item()
            self.running_metrics["mse"] += loss_mse.item()
            self.running_metrics["jerk_temporal"] += loss_jerk_temporal.item()
            self.running_metrics["jerk_spatial"] += loss_jerk_spatial.item()
            self.running_metrics["fr"] += loss_fr.item()
            self.running_metrics["reg"] += loss_reg.item()

        return loss.item()

    def train(self):
        """Main training loop."""
        iterations = self.config.total_iterations // self.config.batch_size

        progress_bar = trange(
            self.iteration,
            self.iteration + iterations,
            desc="Training progress",
            ncols=100,
        )

        for i in progress_bar:
            self.iteration = i
            loss = self.train_step()

            if loss is None:
                continue

            # Logging
            if i % self.config.log_iterations == (self.config.log_iterations - 1):
                progress_bar.set_postfix_str(
                    f"Loss={self.running_metrics['loss'] / self.config.log_iterations:.5f}"
                )
                self.log_metrics()

            # Checkpointing
            if i % self.config.checkpoint_iterations == (
                self.config.checkpoint_iterations - 1
            ):
                self.save_checkpoint()

        # Logging
        if i % self.config.log_iterations != (self.config.log_iterations - 1):
            progress_bar.set_postfix_str(
                f"Loss={self.running_metrics['loss'] / self.config.log_iterations}"
            )
            self.log_metrics()

        # Checkpointing
        if i % self.config.checkpoint_iterations != (
            self.config.checkpoint_iterations - 1
        ):
            self.save_checkpoint()


def main(
    # Allow overriding any config parameter via CLI
    **kwargs,
):
    """
    Train the model with the specified configuration.

    Args:
        **kwargs: Override any parameter from TrainingConfig
    """
    # Create config with CLI overrides
    config = TrainingConfig(
        **{k: v for k, v in kwargs.items() if k in TrainingConfig.__dataclass_fields__}
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
