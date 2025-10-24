import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from encoder import Encoder
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


class Trainer:
    """
    Simple trainer class for the Encoder model.

    Args:
        encoder: The Encoder model
        dataloader: DataLoader for training data
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
        log_dir: Directory for tensorboard logs
        log_every: Log to tensorboard every n iterations
        save_every: Save model checkpoint every n iterations
        checkpoint_dir: Directory to save model checkpoints
    """

    def __init__(
        self,
        encoder,
        dataloader,
        learning_rate=1e-3,
        device="cuda",
        log_dir="runs/experiment",
        log_every=100,
        save_every=1000,
        checkpoint_dir="checkpoints",
        l2_reg_weight=5e-1,
        spatial_var_weight=0.001,
    ):
        self.encoder = encoder
        self.dataloader = dataloader
        self.device = device
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.l2_reg_weight = l2_reg_weight
        self.spatial_var_weight = spatial_var_weight

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Move model to device
        self.encoder.to(device)

        # Setup optimizer
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)

        # Loss function - assuming reconstruction task
        self.criterion = nn.L1Loss()

        # Setup tensorboard writer
        self.writer = SummaryWriter(log_dir)

        # Create infinite iterator from dataloader
        self.data_iter = iter(dataloader)

    def compute_spatial_variance(self):
        """
        Calculate total variance of pixel values around center of mass for batched 2D images.
        Uses squared pixel values (energy) to determine center of mass.

        Uses images stored in self.encoder.kernel and device from self.device.

        Returns:
            variances: Tensor of shape (n,) - variance for each image in the batch
        """
        images = self.encoder.spatial_kernels  # Shape: (n, x, x)
        n, x, _ = images.shape
        device = self.device

        # Create coordinate grids (broadcasted for efficiency)
        i_coords = torch.arange(x, device=device, dtype=torch.float32).view(1, x, 1)
        j_coords = torch.arange(x, device=device, dtype=torch.float32).view(1, 1, x)

        # Use energy (squared pixel values) as mass weights
        energy = images**2

        # Calculate total energy for each image
        total_energy = energy.sum(dim=(1, 2))  # Shape: (n,)

        # Avoid division by zero
        total_energy = torch.clamp(total_energy, min=1e-8)

        # Calculate center of mass coordinates using energy weights
        com_i = (energy * i_coords).sum(dim=(1, 2)) / total_energy  # Shape: (n,)
        com_j = (energy * j_coords).sum(dim=(1, 2)) / total_energy  # Shape: (n,)

        # Calculate squared distances from center of mass
        # Reshape com coordinates for broadcasting
        com_i = com_i.view(n, 1, 1)
        com_j = com_j.view(n, 1, 1)

        # Squared distance from center of mass for each pixel
        squared_distances = (i_coords - com_i) ** 2 + (j_coords - com_j) ** 2

        # Calculate variance using original pixel values weighted by energy
        variances = (energy * squared_distances).sum(dim=(1, 2)) / total_energy.view(n)

        return variances

    def save_checkpoint(self, iteration):
        """Save model checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_iter_{iteration}.pt"
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
        # Assuming batch is a tensor or tuple/list containing input tensor
        if isinstance(batch, (tuple, list)):
            x = batch[0]  # First element is usually the input
        else:
            x = batch

        # Move data to device
        x = x.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.encoder(x)

        # Compute loss (example: reconstruction loss)
        # You might need to modify this based on your specific task
        # For autoencoder-style training, you might compare output to some target
        target = x[
            :, : -self.encoder.T + 1, :, :
        ]  # Adjust target shape to match output: (batch_size, t-T+1, n, n)
        reconstruction_loss = self.criterion(output, target)

        # Compute L2 regularization loss on spatial and temporal kernels
        spatial_l2 = torch.mean(self.encoder.spatial_kernels**2)
        temporal_l2 = torch.mean(self.encoder.temporal_kernels**2)
        decoder1_l2 = torch.mean(self.encoder.decoder1.weight**2)
        decoder2_l2 = torch.mean(self.encoder.decoder2.weight**2)
        l2_reg_loss = self.l2_reg_weight * torch.mean(
            spatial_l2 + temporal_l2 + decoder1_l2 + decoder2_l2
        )

        # Compute spatial variance loss
        spatial_var_loss = (
            self.spatial_var_weight * self.compute_spatial_variance().mean()
        )

        # Total loss
        loss = reconstruction_loss + l2_reg_loss + spatial_var_loss

        if self.iteration % self.log_every == 0:
            self.current_output = output
            self.current_target = target
            self.current_reconstruction_loss = reconstruction_loss.item()
            self.current_l2_loss = l2_reg_loss.item()
            self.current_spatial_var_loss = spatial_var_loss.item()

        # Backward pass
        loss.backward()

        # Update parameters
        self.optimizer.step()

        return loss.item()

    def log_to_tensorboard(self, loss, iteration):
        # Log scalar losses
        self.writer.add_scalar("Loss/Total", loss, iteration + 1)
        self.writer.add_scalar(
            "Loss/Reconstruction", self.current_reconstruction_loss, iteration + 1
        )
        self.writer.add_scalar(
            "Loss/L2_Regularization", self.current_l2_loss, iteration + 1
        )
        self.writer.add_scalar(
            "Loss/Spatial_Variance", self.current_spatial_var_loss, iteration + 1
        )
        # Log example output video (first sample in batch)
        video = torch.cat(
            [self.current_target[0].unsqueeze(0), self.current_output[0].unsqueeze(0)],
            dim=0,
        )
        video = self.rescale(video).cpu()  # Rescale to [0, 1] for visualization
        self.writer.add_video(
            "Training/Output",
            video.unsqueeze(2).expand(
                -1, -1, 3, -1, -1
            ),  # Add batch dimension for video logging
            global_step=iteration + 1,
        )

        # Log spatial kernels
        spatial_kernels = self.encoder.spatial_kernels.detach().cpu()
        spatial_kernels = self.rescale(spatial_kernels)
        self.writer.add_images(
            f"Kernels/Spatial",
            spatial_kernels.unsqueeze(1),  # Add channel dimension
            global_step=iteration + 1,
            dataformats="NCHW",
        )

        # Log temporal kernels
        temporal_kernels = self.encoder.get_temporal_kernels().detach().cpu()
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
        pbar = tqdm(range(n_iterations), desc="Training", unit="iter")

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
                # Update progress bar with current loss

        # Save final checkpoint
        self.save_checkpoint(n_iterations)

        # Close tensorboard writer
        self.writer.close()
        print("Training completed!")

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
