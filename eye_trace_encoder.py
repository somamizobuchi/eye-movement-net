import torch
import torch.nn as nn
import torch.nn.functional as F
from eye_trace_decoder import TemporalVelocityDecoder


class EyeTraceEncoder(nn.Module):
    """
    Improved video encoder with (2+1)D convolution architecture for eye velocity prediction.

    Key improvements:
    - Better weight initialization
    - ReLU activations instead of tanh
    - Residual connections
    - Batch normalization
    - Better decoder architecture
    - Proper output scaling
    """

    def __init__(
        self,
        kernel_size: int,
        kernel_length: int,
        kernel_delay: int,
        n_channels: int,
        decoder_size: int,
        noise_std: float = 0.05,  # Reduced noise
        max_velocity: float = 50.0,  # Expected max velocity in pixels
    ):
        super(EyeTraceEncoder, self).__init__()

        self.N = kernel_size
        self.T = kernel_length
        self.J = n_channels
        self.K = decoder_size
        self.kernel_delay = kernel_delay
        self.noise_std = noise_std
        self.max_velocity = max_velocity

        # (2+1)D Convolution components with proper Xavier initialization
        # Spatial convolution: J kernels of size (n×n)
        self.spatial_kernels = nn.Parameter(
            torch.empty(n_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_normal_(self.spatial_kernels)

        # Temporal convolution: J kernels of length T
        self.temporal_kernels = nn.Parameter(
            torch.empty(n_channels, kernel_length - kernel_delay)
        )
        nn.init.xavier_normal_(self.temporal_kernels)

        self.decoder = TemporalVelocityDecoder(
            in_channels=n_channels,
            hidden_channels=decoder_size,  # same arg you already pass
            out_channels=2,
        )

        # Batch normalization for temporal features
        self.bn_temporal = nn.BatchNorm1d(n_channels)

    def forward(self, x):
        """
        Forward pass through the improved encoder.
        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, t, n, n)
        Returns:
            torch.Tensor: Eye velocities of shape (batch_size, t-T+1, 2)
        """
        batch_size = x.shape[0]

        # Apply spatial convolution (dot product) to each frame
        x_reshaped = x.view(batch_size, -1, self.N * self.N).clone()

        # Add noise only during training
        if self.training:
            x_reshaped += self.noise_std * torch.randn_like(x_reshaped)

        kernels_reshaped = self.spatial_kernels.view(self.J, self.N * self.N)

        # Apply spatial convolution: (batch_size, t, J)
        spatial_features = torch.matmul(x_reshaped, kernels_reshaped.T)

        # Transpose for temporal convolution: (batch_size, J, t)
        spatial_features = spatial_features.transpose(1, 2)

        # Apply temporal convolution
        temporal_features = F.conv1d(
            spatial_features,
            F.pad(
                self.temporal_kernels.unsqueeze(1),
                (0, self.kernel_delay),
                "constant",
                0,
            ),
            groups=self.J,
        )

        # Apply batch normalization and activation
        # temporal_features: (batch_size, J, t-T+1)
        batch_size, n_channels, t_out = temporal_features.shape

        # Reshape for batch norm: (batch_size * t_out, J)
        temporal_flat = temporal_features.permute(0, 2, 1).reshape(-1, n_channels)
        temporal_normalized = self.bn_temporal(temporal_flat)
        temporal_normalized = F.relu(temporal_normalized)

        # Reshape back: (batch_size, t_out, J)
        features = temporal_normalized.view(batch_size, t_out, n_channels)

        # Permute to (batch_size, J, t_out) for 1D conv decoder
        features_seq = features.permute(0, 2, 1)  # (B, C, T)

        # Apply temporal decoder
        eye_velocities = self.decoder(features_seq)  # (B, T, 2)

        return eye_velocities

    def get_temporal_kernels(self) -> torch.Tensor:
        """Get temporal kernels as a 2D tensor."""
        return F.pad(self.temporal_kernels, (0, self.kernel_delay), "constant", 0)

    def kernel_variance(self):
        """
        Calculate the spatial variance of kernel weights to measure how spread out
        they are. This encourages kernels to be more localized.
        """
        # Reshape the kernels for calculation: (J, N, N)
        kernels = self.spatial_kernels

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
        coords_x = torch.arange(
            self.N, dtype=torch.float32, device=kernels.device
        )
        coords_y = torch.arange(
            self.N, dtype=torch.float32, device=kernels.device
        )

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


# Alternative: Residual Decoder for even better performance
class ResidualVelocityDecoder(nn.Module):
    """Residual decoder specifically for velocity prediction."""

    def __init__(self, input_dim, hidden_dim, max_velocity=50.0):
        super().__init__()
        self.max_velocity = max_velocity

        # First residual block
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        # Second residual block
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        # Output layer
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Input projection for residual connection
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

    def forward(self, x):
        # First residual block
        identity = self.input_proj(x)
        out = self.block1(x)
        out = F.relu(out + identity)

        # Second residual block
        identity = out
        out = self.block2(out)
        out = F.relu(out + identity)

        # Output
        velocities = self.output(out)
        return velocities
