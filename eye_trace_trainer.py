import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from eye_trace_encoder import EyeTraceEncoder
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


class EyeTraceTrainer:
    """
    Trainer class for the EyeTraceEncoder model.

    Args:
        encoder: The EyeTraceEncoder model
        dataloader: DataLoader for training data
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
        log_dir: Directory for tensorboard logs
        log_every: Log to tensorboard every n iterations
        save_every: Save model checkpoint every n iterations
        checkpoint_dir: Directory to save model checkpoints
        l2_spatial_weight: Weight for L2 regularization on spatial kernels
        l2_temporal_weight: Weight for L2 regularization on temporal kernels
        temporal_smoothness_weight: Weight for temporal smoothness regularization
        velocity_loss_weight: Weight for velocity prediction loss
        kernel_variance_weight: Weight for kernel variance regularization
    """

    def __init__(
        self,
        encoder,
        dataloader,
        learning_rate=1e-3,
        device="cuda",
        log_dir="runs/eye_trace_experiment",
        log_every=100,
        save_every=1000,
        checkpoint_dir="eye_trace_checkpoints",
        l2_spatial_weight=5e-1,
        l2_temporal_weight=5e-1,
        temporal_smoothness_weight=0.01,
        velocity_loss_weight=1.0,
        kernel_variance_weight=0.1,
    ):
        self.encoder = encoder
        self.dataloader = dataloader
        self.device = device
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.l2_spatial_weight = l2_spatial_weight
        self.l2_temporal_weight = l2_temporal_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.velocity_loss_weight = velocity_loss_weight
        self.kernel_variance_weight = kernel_variance_weight
        self.learning_rate = learning_rate

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Move model to device
        self.encoder.to(device)

        # Setup optimizer
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)

        # Loss function for velocity regression
        self.criterion = nn.L1Loss()

        # Setup tensorboard writer
        self.writer = SummaryWriter(log_dir)

        # Track metrics for final summary
        self.all_losses = []

        # Create infinite iterator from dataloader
        self.data_iter = iter(dataloader)

    def save_checkpoint(self, iteration):
        """Save model checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"eye_trace_checkpoint_iter_{iteration}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at iteration {iteration}: {checkpoint_path}")

    def get_next_batch(self):
        """Get next batch from dataloader, restart if exhausted."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # Restart iterator if we run out of data
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch

    def train_step(self, batch):
        """Perform a single training step."""
        # Extract data from batch
        # batch[0]: video frames (batch_size, t, n, n)
        # batch[2]: eye_trace - list/tuple of numpy arrays, each of shape (2, t)
        video_frames = batch[0].to(self.device)
        eye_trace_list = batch[2]  # List of numpy arrays

        eye_velocities_target = torch.diff(eye_trace_list.float(), 1, 2).to(self.device)
        eye_velocities_target = eye_velocities_target[:, :, self.encoder.T - 1 :]

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        predicted_velocities = self.encoder(video_frames)
        predicted_velocities = predicted_velocities[:, :-1, :].permute(0, 2, 1)

        # Compute velocity prediction loss
        velocity_loss = self.velocity_loss_weight * self.criterion(
            predicted_velocities, eye_velocities_target
        )

        # Compute L2 regularization loss on spatial
        spatial_l2 = torch.mean(self.encoder.spatial_kernels**2)
        spatial_l2_loss = self.l2_spatial_weight * spatial_l2

        # Compute L2 on the area under temporal kernels (force zero-mean)
        temporal_l2 = self.encoder.temporal_kernels.sum(dim=1).square().mean()
        temporal_l2_loss = self.l2_temporal_weight * temporal_l2

        # Compute temporal smoothness loss (L2 on differences)
        temporal_smoothness = torch.mean(
            torch.diff(self.encoder.temporal_kernels, dim=1) ** 2
        )
        temporal_smoothness_loss = self.temporal_smoothness_weight * temporal_smoothness

        # Compute kernel variance loss to encourage localized kernels
        kernel_variance_loss = (
            self.kernel_variance_weight * self.encoder.kernel_variance()
        )

        # Total loss
        loss = (
            velocity_loss
            + spatial_l2_loss
            + temporal_l2_loss
            + temporal_smoothness_loss
            + kernel_variance_loss
        )

        # Store losses for logging
        if self.iteration % self.log_every == 0:
            self.current_predicted_velocities = predicted_velocities
            self.current_target_velocities = eye_velocities_target
            self.current_velocity_loss = velocity_loss.item()
            self.current_spatial_l2_loss = spatial_l2_loss.item()
            self.current_temporal_l2_loss = temporal_l2_loss.item()
            self.current_temporal_smoothness_loss = temporal_smoothness_loss.item()
            self.current_kernel_variance_loss = kernel_variance_loss.item()

        # Track all losses for summary
        self.all_losses.append(loss.item())

        # Backward pass
        loss.backward()

        # Update parameters
        self.optimizer.step()

        return loss.item()

    def log_to_tensorboard(self, loss, iteration):
        """Log metrics and visualizations to tensorboard."""
        # Log scalar losses
        self.writer.add_scalar("Loss/Total", loss, iteration + 1)
        self.writer.add_scalar(
            "Loss/Velocity", self.current_velocity_loss, iteration + 1
        )
        self.writer.add_scalar(
            "Loss/Spatial_L2", self.current_spatial_l2_loss, iteration + 1
        )
        self.writer.add_scalar(
            "Loss/Temporal_L2", self.current_temporal_l2_loss, iteration + 1
        )
        self.writer.add_scalar(
            "Loss/Temporal_Smoothness",
            self.current_temporal_smoothness_loss,
            iteration + 1,
        )
        self.writer.add_scalar(
            "Loss/Kernel_Variance", self.current_kernel_variance_loss, iteration + 1
        )

        # Log velocity predictions vs targets
        # Plot velocities for first sample in batch
        target_vels = self.current_target_velocities[0].cpu().numpy()  # (t, 2)
        predicted_vels = (
            self.current_predicted_velocities[0].detach().cpu().numpy()
        )  # (t, 2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot X velocities over time
        ax1.plot(target_vels[0, :], label="Target Vx", linewidth=2, alpha=0.7)
        ax1.plot(predicted_vels[0, :], label="Predicted Vx", linewidth=2, alpha=0.7)
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("X Velocity")
        ax1.set_title("X Velocity Prediction")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Y velocities over time
        ax2.plot(target_vels[1, :], label="Target Vy", linewidth=2, alpha=0.7)
        ax2.plot(predicted_vels[1, :], label="Predicted Vy", linewidth=2, alpha=0.7)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Y Velocity")
        ax2.set_title("Y Velocity Prediction")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.writer.add_figure("Predictions/Velocities", fig, global_step=iteration + 1)
        plt.close(fig)

        # Plot 2D velocity vector field (as scatter plot)
        # fig, ax = plt.subplots(figsize=(10, 8))
        # ax.scatter(
        #     target_vels[:, 0],
        #     target_vels[:, 1],
        #     label="Target Velocities",
        #     alpha=0.7,
        #     s=20,
        # )
        # ax.scatter(
        #     predicted_vels[:, 0],
        #     predicted_vels[:, 1],
        #     label="Predicted Velocities",
        #     alpha=0.7,
        #     s=20,
        # )
        # ax.set_xlabel("X Velocity")
        # ax.set_ylabel("Y Velocity")
        # ax.set_title("2D Eye Movement Velocities")
        # ax.legend()
        # ax.grid(True, alpha=0.3)
        # ax.set_aspect("equal")

        # self.writer.add_figure(
        #     "Predictions/2D_Velocities", fig, global_step=iteration + 1
        # )
        # plt.close(fig)

        # Log spatial kernels
        spatial_kernels = self.encoder.spatial_kernels.detach().cpu()
        spatial_kernels_norm = self.rescale(spatial_kernels)
        self.writer.add_images(
            f"Kernels/Spatial",
            spatial_kernels_norm.unsqueeze(1),  # Add channel dimension
            global_step=iteration + 1,
            dataformats="NCHW",
        )

        # Log temporal kernels
        temporal_kernels = self.encoder.get_temporal_kernels().detach().cpu()
        temporal_kernels = torch.fliplr(temporal_kernels)  # conv1d calculates xcorr
        fig, ax = plt.subplots(figsize=(10, 6))
        for j in range(temporal_kernels.shape[0]):
            ax.plot(
                temporal_kernels[j, :].numpy(),
                label=f"Kernel {j}",
                linewidth=2,
                marker="o",
                markersize=4,
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Kernel Value")
        ax.set_title("Temporal Kernels")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        self.writer.add_figure("Kernels/Temporal", fig, global_step=iteration + 1)
        plt.close(fig)

    def train(self, n_iterations):
        """
        Run training loop for n_iterations.

        Args:
            n_iterations: Number of training iterations
        """
        self.encoder.train()

        # Training loop with tqdm progress bar
        pbar = tqdm(range(n_iterations), desc="Training Eye Trace Encoder", unit="iter")

        for iteration in pbar:
            self.iteration = iteration + 1

            # Get next batch
            batch = self.get_next_batch()

            # Perform training step
            loss = self.train_step(batch)

            # Log to tensorboard every log_every iterations
            if self.iteration % self.log_every == 0:
                self.log_to_tensorboard(loss, iteration)
                pbar.set_postfix({"Loss": f"{loss:.6f}"})

            # Save checkpoint every save_every iterations
            if self.iteration % self.save_every == 0:
                self.save_checkpoint(iteration + 1)

        # Save final checkpoint
        self.save_checkpoint(n_iterations)

        # Log final hyperparameters with summary metrics to TensorBoard
        final_loss = self.all_losses[-1] if self.all_losses else 0.0
        avg_loss_last_1000 = (
            sum(self.all_losses[-1000:]) / len(self.all_losses[-1000:])
            if len(self.all_losses) >= 1000
            else final_loss
        )
        min_loss = min(self.all_losses) if self.all_losses else 0.0

        self.writer.add_hparams(
            {
                "lr": self.learning_rate,
                "l2_spatial": self.l2_spatial_weight,
                "l2_temporal": self.l2_temporal_weight,
                "temporal_smooth": self.temporal_smoothness_weight,
                "kernel_var": self.kernel_variance_weight,
                "velocity_weight": self.velocity_loss_weight,
            },
            {
                "hparam/final_loss": final_loss,
                "hparam/avg_loss_last_1k": avg_loss_last_1000,
                "hparam/min_loss": min_loss,
            },
        )

        # Close tensorboard writer
        self.writer.close()
        print("Eye trace training completed!")

    def close(self):
        """Close tensorboard writer."""
        self.writer.close()

    def rescale(self, tensor):
        """Rescale tensor to [0, 1] range."""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_max > tensor_min:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return tensor
