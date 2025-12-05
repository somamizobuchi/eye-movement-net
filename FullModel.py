import torch
import torch.nn as nn
from encoder import Encoder
from V1Decoder import V1Decoder
from eye_trace_decoder import TemporalVelocityDecoder
from recon_decoder import ReconstructionDecoder


class FullModel(nn.Module):
    """
    Video encoder with bifurcated decoding for both velocity prediction and reconstruction.

    Architecture:
    - Encoder: (2+1)D convolution to extract spatiotemporal features
    - Velocity decoder: Predicts eye movement velocities
    - Reconstruction decoder: Reconstructs original frames

    Key improvements:
    - Modular encoder architecture
    - Batch normalization
    - ReLU activations
    - Dual output heads for velocity and reconstruction
    """

    def __init__(
        self,
        kernel_size: int,
        kernel_length: int,
        kernel_delay: int,
        n_channels: int,
        decoder_size: int,
        velocity_hidden_channels: int | None = None,
        noise_std: float = 0.05,
        max_velocity: float = 10.0,
    ):
        super(FullModel, self).__init__()

        self.N = kernel_size
        self.T = kernel_length
        self.J = n_channels
        self.K = decoder_size
        self.max_velocity = max_velocity

        # Default velocity_hidden_channels to decoder_size if not specified
        if velocity_hidden_channels is None:
            velocity_hidden_channels = decoder_size

        # Modular encoder for (2+1)D convolution (includes BN and activation)
        self.encoder = Encoder(
            kernel_size=kernel_size,
            kernel_length=kernel_length,
            kernel_delay=kernel_delay,
            n_channels=n_channels,
            noise_std=noise_std,
        )

        # V1 decoder - temporal convolution followed by spatial linear transformation
        self.v1_decoder = V1Decoder(
            input_dims=n_channels,
            output_dims=decoder_size,
            kernel_length=kernel_length,
        )

        # Velocity decoder - predicts eye movements
        self.velocity_decoder = TemporalVelocityDecoder(
            in_channels=decoder_size,
            hidden_channels=velocity_hidden_channels,
            out_channels=2,
            max_velocity=max_velocity,
        )

        # Reconstruction decoder - reconstructs frames
        self.recon_decoder = ReconstructionDecoder(
            n_channels=decoder_size,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        """
        Forward pass through the model with bifurcated output.

        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, t, n, n)
            return_reconstruction (bool): If True, return both velocities and reconstruction

        Returns:
            If return_reconstruction=False:
                torch.Tensor: Eye velocities of shape (batch_size, t-T+1, 2)
            If return_reconstruction=True:
                tuple: (eye_velocities, reconstructed_frames)
                    - eye_velocities: (batch_size, t-T+1, 2)
                    - reconstructed_frames: (batch_size, t-T+1, N, N)
        """
        # Apply (2+1)D convolution encoding (includes BN and activation)
        # features: (batch_size, J, t-T+1)
        features = self.encoder(x)

        # V1 decoder: temporal convolution then linear transformation per timepoint
        # v1_features: (batch_size, K, t-T+1)
        v1_features = self.v1_decoder(features)

        # Velocity decoding
        eye_velocities = self.velocity_decoder(v1_features)  # (batch_size, t_out, 2)

        # Reconstruction decoding
        reconstructed_frames = self.recon_decoder(v1_features)  # (batch_size, t_out, N, N)

        return eye_velocities, reconstructed_frames

    def get_temporal_kernels(self) -> torch.Tensor:
        """Get temporal kernels from the encoder."""
        return self.encoder.get_temporal_kernels()

    def kernel_variance(self):
        """
        Calculate the spatial variance of kernel weights to measure how spread out
        they are. This encourages kernels to be more localized.
        """
        # Get spatial kernels from encoder
        kernels = self.encoder.spatial_kernels

        # Normalize kernels
        kernels_norm = kernels / torch.norm(
            kernels.reshape(self.J, -1), dim=1, keepdim=True
        ).view(self.J, 1, 1)

        # Square the weights to get the energy distribution
        kernel_energy = kernels_norm.pow(2)

        # Compute weight projections along each axis
        Wx = kernel_energy.sum(dim=2)  # Sum along y-axis (projection onto x-axis)
        Wy = kernel_energy.sum(dim=1)  # Sum along x-axis (projection onto y-axis)

        # Create coordinate grids
        coords_x = torch.arange(self.N, dtype=torch.float32, device=kernels.device)
        coords_y = torch.arange(self.N, dtype=torch.float32, device=kernels.device)

        # Calculate center of mass for each kernel along each axis
        mean_x = torch.sum(coords_x.view(1, -1) * Wx, dim=1)  # Shape: [J]
        mean_y = torch.sum(coords_y.view(1, -1) * Wy, dim=1)  # Shape: [J]

        # Calculate variance around center of mass
        var_x = torch.sum(
            (coords_x.view(1, -1) - mean_x.view(-1, 1)).pow(2) * Wx, dim=1
        )
        var_y = torch.sum(
            (coords_y.view(1, -1) - mean_y.view(-1, 1)).pow(2) * Wy, dim=1
        )

        # Total variance is the sum of variances along both axes, averaged across all kernels
        total_variance = (var_x + var_y).mean()

        return total_variance
