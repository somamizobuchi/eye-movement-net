import datetime
import signal
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import itertools

import fire
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from datasets import VideoDataset
from model import Encoder
from utils import (
    rescale,
)


@dataclass
class TrainingConfig:
    # Model parameters
    kernel_size: int = 24
    kernel_length: int = 24
    n_kernels: int = 108
    fs: int = 1000  # Hz
    ppd: int = 180.0  # pixels per degree
    drift_samples: int = 64
    temporal_pad: List[int] = field(default_factory=lambda: [0, 5])

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

    # Grid search parameters
    grid_search: bool = False
    # Dictionary mapping parameter names to lists of values to try
    grid_params: Dict[str, List[float]] = field(default_factory=dict)
    # Number of iterations to run for each grid search configuration
    grid_search_iterations: int = 100_000


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
            if self.config.grid_search:
                self.run_id = (
                    f"grid_search_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
                )
                self.grid_search_dir = Path(f"./grid_search/{self.run_id}")
                self.grid_search_dir.mkdir(parents=True, exist_ok=True)
                self.checkpoint_dir = self.grid_search_dir
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

        self.dataset = VideoDataset(
            "data/pink_noise_videos.npy",
            self.config.kernel_size,
            self.config.drift_samples,
        )

        self.data_loader = DataLoader(
            self.dataset, shuffle=True, batch_size=self.config.batch_size
        )

    def setup_logging(self):
        """Initialize TensorBoard logging."""
        if self.config.grid_search:
            # Create a summary writer for grid search results
            self.writer = SummaryWriter(log_dir=f"logs/grid_search/{self.run_id}")
        else:
            self.writer = SummaryWriter(log_dir=f"logs/{self.run_id}")

    def setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals by saving checkpoint and exiting."""
        print("\nReceived shutdown signal. Saving checkpoint...")
        if not self.config.grid_search:
            self.save_checkpoint()
        else:
            # For grid search, save the current grid search results
            self.save_grid_search_results()
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

    def train_step(self, alpha=None, delta=None, beta=None, gamma=None):
        """
        Execute single training step.

        Args:
            alpha, delta, beta, gamma: Optional override values for the loss weights.
                If not provided, uses the values from config.
        """
        retinal_input = next(iter(self.data_loader))

        self.optimizer.zero_grad()
        out, fr = self.model(retinal_input.clone().to(self.config.device))

        # Current
        self.current_target = retinal_input[:, self.model.kernel_length - 1 :].to(
            self.config.device
        )
        self.current_reconstruction = out

        # Use provided loss weights or fall back to config values
        alpha = alpha if alpha is not None else self.config.alpha
        delta = delta if delta is not None else self.config.delta
        beta = beta if beta is not None else self.config.beta
        gamma = gamma if gamma is not None else self.config.gamma

        # Compute losses
        loss_mse = torch.nn.functional.l1_loss(
            self.current_reconstruction, self.current_target
        )
        loss_jerk_temporal = alpha * self.model.kernel_temporal_jerk()
        loss_jerk_spatial = delta * self.model.kernel_spatial_jerk()
        loss_fr = beta * out.abs().mean()
        loss_reg = gamma * (
            self.model.spatial_kernels.square().mean()
            + self.model.temporal_kernels.square().mean()
            + self.model.spatial_decoder.square().mean()
        )

        # Total loss - now using all components
        loss = loss_mse + loss_jerk_temporal + loss_jerk_spatial + loss_fr + loss_reg

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

    def run_grid_search(self):
        """
        Perform grid search over the loss weight parameters.
        """
        print("Starting grid search...")

        # Extract grid search parameters
        grid_params = self.config.grid_params

        # Default grid search parameters if none provided
        if not grid_params:
            grid_params = {
                "alpha": [1e-2, 5e-2, 1e-1, 5e-1, 1e0],
                "delta": [1e-2, 5e-2, 1e-1, 5e-1, 1e0],
                "beta": [1e-3, 1e-2, 5e-2, 1e-1, 5e-1],
                "gamma": [1e-1, 1e0, 1e1, 1e2],
            }

        # Generate all combinations of parameters
        param_names = list(grid_params.keys())
        param_values = list(grid_params.values())
        param_combinations = list(itertools.product(*param_values))

        # Create a results file
        results_file = self.grid_search_dir / "grid_search_results.csv"
        with open(results_file, "w") as f:
            # Write header
            header = ",".join(param_names + ["iteration", "train_loss"])
            f.write(header + "\n")

        # Loop through all parameter combinations
        for i, params in enumerate(param_combinations):
            param_dict = {name: value for name, value in zip(param_names, params)}

            print(f"\nGrid search combination {i + 1}/{len(param_combinations)}:")
            for name, value in param_dict.items():
                print(f"  {name}: {value}")

            # Reset model for each parameter combination
            self.setup_model()
            self.running_metrics = {k: 0.0 for k in self.running_metrics}

            # Create a subdirectory for this combination
            run_dir = self.grid_search_dir / f"run_{i}"
            run_dir.mkdir(exist_ok=True)

            # Train for specified number of iterations
            iterations = self.config.grid_search_iterations
            progress_bar = trange(
                0,
                iterations,
                desc=f"Training combination {i + 1}/{len(param_combinations)}",
                ncols=100,
            )

            for j in progress_bar:
                # Use the current parameter combination
                loss = self.train_step(**param_dict)

                if loss is None:
                    continue

                # Log periodically
                if j % self.config.log_iterations == (self.config.log_iterations - 1):
                    # Log metrics to TensorBoard
                    for name, value in self.running_metrics.items():
                        self.writer.add_scalar(
                            f"Grid/{name}/combo_{i}",
                            value / self.config.log_iterations,
                            j,
                        )

                    # Log to results file
                    with open(results_file, "a") as f:
                        param_values_str = ",".join(
                            str(param_dict[name]) for name in param_names
                        )
                        train_loss = (
                            self.running_metrics["loss"] / self.config.log_iterations
                        )
                        line = f"{param_values_str},{j},{train_loss}"
                        f.write(line + "\n")

                    # Reset running metrics
                    self.running_metrics = {k: 0.0 for k in self.running_metrics}

                    # Update progress bar
                    progress_bar.set_postfix_str(f"Loss: {train_loss:.5f}")

            # Save the final model for this combination
            final_checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iteration": iterations,
                "metrics": self.running_metrics,
                "parameters": param_dict,
            }
            torch.save(final_checkpoint, run_dir / "final.pt")

        print(f"\nGrid search complete. Results saved to {self.grid_search_dir}")

    def save_grid_search_results(self):
        """Save the current grid search results."""
        # This is a placeholder for any additional cleanup needed when
        # grid search is interrupted
        pass

    def train(self):
        """Main training loop."""
        if self.config.grid_search:
            # Run grid search instead of regular training
            self.run_grid_search()
            return

        # Regular training
        self._run_training_loop()

    def _run_training_loop(self):
        """Internal method for the main training loop."""
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

        # Final logging if needed
        if i % self.config.log_iterations != (self.config.log_iterations - 1):
            progress_bar.set_postfix_str(
                f"Loss={self.running_metrics['loss'] / self.config.log_iterations:.5f}"
            )
            self.log_metrics()

        # Final checkpoint if needed
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

    Grid Search Usage:
        python train_main.py --grid_search=True --grid_search_iterations=10000
                            --alpha_values=0.1,0.5,1.0 --delta_values=0.1,0.5,1.0
                            --beta_values=0.01,0.1,0.5 --gamma_values=0.1,1.0,10.0
    """
    # Handle grid search parameters if provided
    grid_params = {}
    param_keys = {
        "alpha_values": "alpha",
        "beta_values": "beta",
        "delta_values": "delta",
        "gamma_values": "gamma",
    }

    # Parse grid search parameters
    for key, param_name in param_keys.items():
        if key in kwargs:
            # Get the value and ensure it's a string
            values_str = kwargs.pop(key)

            # Handle if value is already a list or tuple
            if isinstance(values_str, (list, tuple)):
                values = [float(v) for v in values_str]
            else:
                # Convert comma-separated string to list of floats
                values = [float(v.strip()) for v in str(values_str).split(",")]

            grid_params[param_name] = values

    # Add the parsed grid_params if any
    if grid_params and kwargs.get("grid_search", False):
        kwargs["grid_params"] = grid_params

    # Create config with CLI overrides
    config = TrainingConfig(
        **{k: v for k, v in kwargs.items() if k in TrainingConfig.__dataclass_fields__}
    )

    if config.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            config.device = "cpu"
    elif config.device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS is not available. Using CPU instead.")
            config.device = "cpu"

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
