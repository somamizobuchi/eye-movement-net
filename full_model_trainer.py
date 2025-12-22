import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from FullModel import FullModel
from reconstruct_from_model import reconstruct_batch_optimized
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


def bending_energy_loss(y):
    """
    Compute bending energy loss for smoothness regularization.

    Args:
        y: Tensor of shape (batch, channels, length) representing velocities or positions

    Returns:
        Scalar loss value
    """
    # y: (batch, channels, length)
    batch_size, n_channels, length = y.shape
    kernel = torch.tensor([1., -2., 1.], device=y.device).view(1, 1, 3)
    # Use depthwise convolution (groups=n_channels) to apply kernel to each channel
    y2 = F.conv1d(y, kernel.expand(n_channels, 1, 3), padding=1, groups=n_channels)
    # Return mean squared second derivative
    return (y2 ** 2).mean()


class FullModelTrainer:
    """
    Trainer class for the FullModel with bifurcated reconstruction.

    This trainer focuses on reconstruction quality without velocity supervision.

    Args:
        model: The FullModel instance
        dataloader: DataLoader for training data
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
        log_dir: Directory for tensorboard logs
        log_every: Log to tensorboard every n iterations
        save_every: Save model checkpoint every n iterations
        checkpoint_dir: Directory to save model checkpoints
        l2_spatial_weight: Weight for L2 regularization on spatial kernels
        l2_temporal_weight: Weight for L2 regularization on temporal kernels
        temporal_smoothness_weight: Weight for temporal smoothness of velocity decoder
        reconstruction_loss_weight: Weight for reconstruction MSE loss
        kernel_variance_weight: Weight for kernel variance regularization
        position_loss_weight: Weight for position supervision loss
        v1_l2_weight: Weight for V1Decoder regularization
        balanced_losses: List of loss names to use GradNorm balancing on. Valid names:
            'reconstruction', 'position', 'spatial_l2', 'temporal_l2',
            'temporal_smoothness', 'kernel_variance', 'v1_l2'. Default: ['reconstruction', 'position']
    """

    def __init__(
        self,
        model,
        dataloader,
        learning_rate=1e-3,
        device="cuda",
        log_dir="runs/full_model_experiment",
        log_every=100,
        save_every=1000,
        checkpoint_dir="full_model_checkpoints",
        l2_spatial_weight=1e-3,
        l2_temporal_weight=1e-2,
        temporal_smoothness_weight=1e-3,
        reconstruction_loss_weight=1.0,
        kernel_variance_weight=1e-4,
        position_loss_weight=0.1,
        v1_l2_weight=1e-3,
        balanced_losses=None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.spatial_l2_weight = l2_spatial_weight
        self.temporal_l2_weight = l2_temporal_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.kernel_variance_weight = kernel_variance_weight
        self.position_loss_weight = position_loss_weight
        self.v1_l2_weight = v1_l2_weight
        self.learning_rate = learning_rate

        # Configure which losses use GradNorm balancing vs fixed weights
        if balanced_losses is None:
            balanced_losses = ["reconstruction", "position"]

        # Validate loss names
        valid_losses = {
            "reconstruction",
            "position",
            "spatial_l2",
            "temporal_l2",
            "temporal_smoothness",
            "kernel_variance",
            "v1_l2",
        }
        for loss_name in balanced_losses:
            if loss_name not in valid_losses:
                raise ValueError(
                    f"Invalid loss name '{loss_name}'. Valid options: {valid_losses}"
                )

        self.balanced_losses = set(balanced_losses)
        self.fixed_losses = valid_losses - self.balanced_losses

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Move model to device
        self.model.to(device)

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Setup tensorboard writer
        self.writer = SummaryWriter(log_dir)

        # Track metrics for final summary
        self.all_losses = []

        # Create infinite iterator from dataloader
        self.data_iter = iter(dataloader)

    def _compute_gradient_norm(self, loss):
        """
        Compute the gradient norm of a loss with respect to model parameters.

        Args:
            loss: Scalar loss tensor

        Returns:
            Gradient norm (scalar)
        """
        # Compute gradients for this loss
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # Compute norm of all gradients
        grad_norm = 0.0
        for grad in grads:
            if grad is not None:
                grad_norm += grad.norm(2).item() ** 2

        grad_norm = grad_norm**0.5
        return grad_norm

    def save_checkpoint(self, iteration):
        """Save model checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"full_model_checkpoint_iter_{iteration}.pt"
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
        # batch[0]: retinal_input (batch_size, t, n, n)
        # batch[1]: target image (batch_size, H, W)
        # batch[2]: eye_trace (batch_size, 2, t)
        # batch[3]: mask (batch_size, H, W)
        retinal_input = batch[0].to(self.device)
        target = batch[1].to(self.device)
        eye_trace = batch[2].to(self.device)
        mask = batch[3].to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        eye_velocities, reconstructed_frames = self.model(retinal_input)

        # Extract initial positions from eye trace for entire batch
        # eye_trace shape: (batch_size, 2, t)
        # With pad_start = kernel_length * 2 - 2, we need to account for the extra padding
        # The model receives input with padding, then V1Decoder reduces by (kernel_length - 1)
        # Initial position should be at pad_start + model.T - 1 - (kernel_length - 1)
        # = pad_start + model.T - kernel_length
        pad_start = self.model.T * 2 - 2
        initial_idx = pad_start + self.model.T - self.model.T
        initial_positions = (
            eye_trace[:, :, initial_idx].transpose(0, 1).transpose(0, 1)
        )  # (batch_size, 2)

        # Get ground truth positions for position supervision
        # Shape: (batch_size, t_out, 2)
        t_out = eye_velocities.shape[1]
        gt_positions = eye_trace[:, :, initial_idx : initial_idx + t_out].transpose(1, 2)  # (batch_size, t_out, 2)

        # Integrate predicted velocities to get predicted positions
        from reconstruct_from_model import integrate_velocities_batch
        predicted_positions = integrate_velocities_batch(
            eye_velocities, initial_positions, dt=1.0
        )  # (batch_size, t_out, 2)

        # Compute losses (unweighted for GradNorm calculation)
        position_loss_unweighted = ((predicted_positions - gt_positions) ** 2).mean()

        # Reconstruct entire batch at once using optimized function
        canvas_size = target.shape[-2:]
        reconstructed_canvases = reconstruct_batch_optimized(
            eye_velocities,
            reconstructed_frames,
            initial_positions,
            canvas_size,
            dt=1.0,
        )  # (batch_size, 1, Hc, Wc)

        # Compute MSE on masked region for entire batch
        target_imgs = target.unsqueeze(1)  # (batch_size, 1, Hc, Wc)
        mask_imgs = mask.unsqueeze(1).float()  # (batch_size, 1, Hc, Wc)

        # Compute loss with mask
        squared_error = torch.abs(reconstructed_canvases - target_imgs)
        # squared_error = (
        #     squared_error * mask_imgs
        #     + 5.0 * squared_error * (~mask.unsqueeze(1)).float()
        # )
        reconstruction_loss_unweighted = squared_error.sum() / (mask_imgs.sum() + 1e-8)

        # Compute L2 regularization loss on spatial kernels
        spatial_l2 = self.model.encoder.spatial_kernels.square().mean()

        # Compute L2 on the area under temporal kernels (force zero-mean)
        temporal_l2 = self.model.encoder.temporal_kernels.square().mean()

        # Compute bending energy loss on temporal kernels for smoothness
        # temporal_kernels: (n_channels, kernel_length)
        # Reshape to (1, n_channels, kernel_length) for conv1d
        temporal_kernels_for_smoothness = self.model.encoder.temporal_kernels.unsqueeze(0)
        temporal_smoothness = bending_energy_loss(temporal_kernels_for_smoothness)

        # Compute regularization on V1Decoder weights
        # L2 regularization on V1 temporal kernels
        v1_temporal_l2 = self.model.v1_decoder.temporal_kernels.square().mean()

        # L2 regularization on V1 spatial kernels (linear layer weights)
        v1_spatial_l2 = self.model.v1_decoder.spatial_kernels.weight.square().mean()

        # Combine V1 regularization losses
        v1_l2_loss = v1_temporal_l2 + v1_spatial_l2

        # Compute kernel variance loss to encourage localized kernels
        kernel_variance_loss_unweighted = self.model.kernel_variance()

        # Dictionary of all unweighted losses
        unweighted_losses = {
            "reconstruction": reconstruction_loss_unweighted,
            "position": position_loss_unweighted,
            "spatial_l2": spatial_l2,
            "temporal_l2": temporal_l2,
            "temporal_smoothness": temporal_smoothness,
            "kernel_variance": kernel_variance_loss_unweighted,
            "v1_l2": v1_l2_loss,
        }

        # GradNorm: Compute gradient norms only for balanced losses
        if self.balanced_losses:
            grad_norms = {}
            for loss_name in self.balanced_losses:
                grad_norms[loss_name] = self._compute_gradient_norm(
                    unweighted_losses[loss_name]
                )

            # Balance weights so gradients contribute equally
            avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)

            # Update weights inversely proportional to gradient magnitude
            for loss_name in self.balanced_losses:
                weight_attr = f"{loss_name}_weight"
                setattr(
                    self,
                    weight_attr,
                    avg_grad_norm / (grad_norms[loss_name] + 1e-8),
                )

        # Compute weighted losses
        weighted_losses = {}
        for loss_name, unweighted_loss in unweighted_losses.items():
            weight_attr = f"{loss_name}_weight"
            weight = getattr(self, weight_attr)
            weighted_losses[loss_name] = weight * unweighted_loss

        # Extract individual weighted losses for convenience
        reconstruction_loss = weighted_losses["reconstruction"]
        position_loss = weighted_losses["position"]
        spatial_l2_loss = weighted_losses["spatial_l2"]
        temporal_l2_loss = weighted_losses["temporal_l2"]
        temporal_smoothness_loss = weighted_losses["temporal_smoothness"]
        kernel_variance_loss = weighted_losses["kernel_variance"]

        # Total loss
        loss = (
            reconstruction_loss
            + spatial_l2_loss
            + temporal_l2_loss
            + temporal_smoothness_loss
            + kernel_variance_loss
            + position_loss
        )

        # Store losses for logging
        if self.iteration % self.log_every == 0:
            # Store raw (unweighted) losses
            self.current_reconstruction_loss = reconstruction_loss_unweighted.item()
            self.current_position_loss = position_loss_unweighted.item()
            self.current_spatial_l2_loss = spatial_l2.item()
            self.current_temporal_l2_loss = temporal_l2.item()
            self.current_temporal_smoothness_loss = temporal_smoothness.item()
            self.current_kernel_variance_loss = kernel_variance_loss_unweighted.item()

            # Store sample reconstructions for visualization
            self.current_target = target[0].detach().cpu()
            self.current_mask = mask[0].detach().cpu()
            # Use the already computed reconstruction from the batch
            self.current_reconstruction = reconstructed_canvases[0, 0].detach().cpu()

            # Store positions for visualization
            # Integrate predicted velocities to get positions
            pred_velocities = eye_velocities[0]  # (t_out, 2)
            t_out = pred_velocities.shape[0]

            # Get initial position
            initial_pos = initial_positions[0]  # (2,)

            # Integrate velocities to positions
            from reconstruct_from_model import integrate_velocities

            predicted_positions = integrate_velocities(
                pred_velocities, initial_pos, dt=1.0
            )  # (t_out, 2)
            self.current_predicted_positions = predicted_positions.detach().cpu()

            # Extract ground truth positions from eye trace
            # eye_trace shape: (batch_size, 2, t)
            # We need positions starting at initial_idx (same as initial_position) through the output timesteps
            # This ensures gt_positions[0] matches the initial position
            pad_start = self.model.T * 2 - 2
            initial_idx = pad_start + self.model.T - self.model.T
            gt_positions = eye_trace[
                0, :, initial_idx : initial_idx + t_out
            ].T  # (t_out, 2)

            # Verify first ground truth position matches initial position
            # (They should be the same since both are at time T-1)
            assert torch.allclose(
                gt_positions[0], initial_pos, atol=1e-5
            ), f"First GT position {gt_positions[0]} doesn't match initial position {initial_pos}"

            # Ensure gt_positions matches the length of predicted positions
            if gt_positions.shape[0] < t_out:
                # Pad with last position if needed (edge case)
                last_pos = (
                    gt_positions[-1:]
                    if gt_positions.shape[0] > 0
                    else torch.zeros((1, 2), device=self.device)
                )
                padding = last_pos.repeat(t_out - gt_positions.shape[0], 1)
                gt_positions = torch.cat([gt_positions, padding], dim=0)

            self.current_gt_positions = gt_positions.detach().cpu()

        # Track all losses for summary
        self.all_losses.append(loss.item())

        # Backward pass
        loss.backward()

        # Update parameters
        self.optimizer.step()

        return loss.item()

    def log_to_tensorboard(self, loss, iteration):
        """Log metrics and visualizations to tensorboard."""
        # Log scalar losses (raw, unweighted values)
        self.writer.add_scalar("Loss/Total", loss, iteration + 1)
        self.writer.add_scalar(
            "Loss/Reconstruction", self.current_reconstruction_loss, iteration + 1
        )
        self.writer.add_scalar(
            "Loss/Position", self.current_position_loss, iteration + 1
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

        # Log GradNorm loss weights (dynamically adjusted for balanced losses)
        for loss_name in self.balanced_losses:
            weight_attr = f"{loss_name}_weight"
            weight = getattr(self, weight_attr)
            self.writer.add_scalar(
                f"Weights/{loss_name.replace('_', ' ').title()}", weight, iteration + 1
            )

        # Log fixed loss weights (not adjusted by GradNorm)
        for loss_name in self.fixed_losses:
            weight_attr = f"{loss_name}_weight"
            weight = getattr(self, weight_attr)
            self.writer.add_scalar(
                f"Weights/{loss_name.replace('_', ' ').title()}", weight, iteration + 1
            )

        # Visualize reconstruction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Apply mask to ground truth for visualization
        masked_target = self.current_target * self.current_mask
        axes[0].imshow(masked_target.numpy(), cmap="gray")
        axes[0].set_title("Ground Truth (Masked)")
        axes[0].axis("off")

        axes[1].imshow(self.current_reconstruction.numpy(), cmap="gray")
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        error = (
            torch.abs(self.current_reconstruction - self.current_target)
            * self.current_mask
        )
        axes[2].imshow(error.numpy(), cmap="hot")
        axes[2].set_title(f"Absolute Error")
        axes[2].axis("off")

        plt.tight_layout()
        self.writer.add_figure(
            "Reconstruction/Comparison", fig, global_step=iteration + 1
        )
        plt.close(fig)

        # Log spatial kernels
        spatial_kernels = self.model.encoder.spatial_kernels.detach().cpu()
        spatial_kernels_norm = self.rescale(spatial_kernels)
        self.writer.add_images(
            f"Kernels/Spatial",
            spatial_kernels_norm.unsqueeze(1),  # Add channel dimension
            global_step=iteration + 1,
            dataformats="NCHW",
        )

        # Log encoder temporal kernels
        temporal_kernels = self.model.get_temporal_kernels().detach().cpu()
        temporal_kernels = torch.fliplr(temporal_kernels)  # conv1d calculates xcorr

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot subset of kernels to avoid clutter
        n_plot = min(10, temporal_kernels.shape[0])
        for j in range(n_plot):
            ax.plot(
                temporal_kernels[j, :].numpy(),
                label=f"Kernel {j}",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Kernel Value")
        ax.set_title(f"Encoder Temporal Kernels (showing {n_plot}/{temporal_kernels.shape[0]})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()

        self.writer.add_figure("Kernels/Temporal_Encoder", fig, global_step=iteration + 1)
        plt.close(fig)

        # Log V1 decoder temporal kernels (with delay padding)
        v1_temporal_kernels = self.model.v1_decoder.get_temporal_kernels().detach().cpu()
        v1_temporal_kernels = torch.fliplr(v1_temporal_kernels)  # conv1d calculates xcorr

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot subset of kernels to avoid clutter
        n_plot_v1 = min(10, v1_temporal_kernels.shape[0])
        for j in range(n_plot_v1):
            ax.plot(
                v1_temporal_kernels[j, :].numpy(),
                label=f"Kernel {j}",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Kernel Value")
        ax.set_title(f"V1 Decoder Temporal Kernels (showing {n_plot_v1}/{v1_temporal_kernels.shape[0]})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()

        self.writer.add_figure("Kernels/Temporal_V1Decoder", fig, global_step=iteration + 1)
        plt.close(fig)

        # Log position comparison
        pred_pos = self.current_predicted_positions.numpy()  # (t_out, 2)
        gt_pos = self.current_gt_positions.numpy()  # (t_out, 2)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot X positions
        time_steps = np.arange(pred_pos.shape[0])
        axes[0].plot(
            time_steps,
            gt_pos[:, 0],
            label="Ground Truth",
            linewidth=2,
            color="blue",
            alpha=0.7,
        )
        axes[0].plot(
            time_steps,
            pred_pos[:, 0],
            label="Predicted",
            linewidth=2,
            color="red",
            alpha=0.7,
            linestyle="--",
        )
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("X Position")
        axes[0].set_title("X Position Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot Y positions
        axes[1].plot(
            time_steps,
            gt_pos[:, 1],
            label="Ground Truth",
            linewidth=2,
            color="blue",
            alpha=0.7,
        )
        axes[1].plot(
            time_steps,
            pred_pos[:, 1],
            label="Predicted",
            linewidth=2,
            color="red",
            alpha=0.7,
            linestyle="--",
        )
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Y Position")
        axes[1].set_title("Y Position Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.writer.add_figure("Positions/Comparison", fig, global_step=iteration + 1)
        plt.close(fig)

    def train(self, n_iterations):
        """
        Run training loop for n_iterations.

        Args:
            n_iterations: Number of training iterations
        """
        self.model.train()

        # Training loop with tqdm progress bar
        pbar = tqdm(range(n_iterations), desc="Training Full Model", unit="iter")

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
            },
            {
                "hparam/final_loss": final_loss,
                "hparam/avg_loss_last_1k": avg_loss_last_1000,
                "hparam/min_loss": min_loss,
            },
        )

        # Close tensorboard writer
        self.writer.close()
        print("Full model training completed!")

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
