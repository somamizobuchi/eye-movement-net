import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import hex_grid_torch, generate_rgc_spatial_rf, generate_rgc_impulse_response
import numpy as np


class GtEncoder(nn.Module):
    """
    Modular (2+1)D convolution encoder with primate RGC-inspired kernels.

    Architecture:
    1. Input: video of shape (batch_size, t, n, n)
    2. Spatial convolution: dot product with 2*J kernels of size (nxn)
       (J center-surround pairs + J inverted versions from hexagonal pattern)
    3. Temporal convolution: 1D convolution with 2*J kernels of length T
    4. Batch normalization
    5. Softplus activation
    6. Output: spatiotemporal features (batch_size, 2*J, t-T+1)

    Args:
        kernel_size (int): Spatial dimension of input video (nxn frames)
        kernel_length (int): Temporal kernel length (T)
        spacing (float): Hexagonal grid spacing in pixels
        ppd (float): Pixels per degree of visual angle
        fs (float): Sampling frequency in Hz
        eccentricity (float): Eccentricity in degrees for RGC RF parameters
        cell_type (str): 'P' for Parvocellular or 'M' for Magnocellular
        noise_std (float): Standard deviation of noise added during training
    """

    def __init__(
        self,
        kernel_size: int,
        kernel_length: int,
        spacing: float,
        ppd: float,
        fs: float,
        eccentricity: float = 0.3,
        cell_type: str = "M",
        noise_std: float = 0.05,
    ):
        super(GtEncoder, self).__init__()

        self.noise_std = noise_std
        self.N = kernel_size
        self.T = kernel_length

        # Generate hexagonal grid points
        pts = hex_grid_torch(kernel_size, kernel_size, spacing, "cpu")
        pts = pts.numpy()
        K = len(pts)  # Number of grid points

        # Generate spatial kernels from hexagonal grid
        spatial_kernels_list = []
        for pt in pts:
            cx, cy = float(pt[0]), float(pt[1])
            _, _, kernel = generate_rgc_spatial_rf(
                cell_type=cell_type,
                eccentricity=eccentricity,
                ppd=ppd,
                size_pixels=kernel_size,
                center=(cx, cy),
            )
            spatial_kernels_list.append(kernel)

        # ON Center
        spatial_kernels_original = torch.from_numpy(
            np.array(spatial_kernels_list)
        ).float()
        # OFF Center
        spatial_kernels_inverted = -spatial_kernels_original

        # Generate temporal kernels from RGC impulse response
        _, temporal_response = generate_rgc_impulse_response(
            cell_type=cell_type, num_samples=kernel_length, fs=int(fs)
        )
        temporal_kernel = torch.from_numpy(
            temporal_response[:kernel_length][::-1].copy()
        ).float()

        # J = 2*K (K original + K inverted)
        self.J = 2 * K

        # Spatial kernels: (J, N, N) = (2*K, N, N)
        spatial_kernels = torch.cat(
            [spatial_kernels_original, spatial_kernels_inverted], dim=0
        )
        self.register_buffer("spatial_kernels", spatial_kernels)

        # Temporal kernels: (J, T) = (2*K, T) - same kernel for all channels
        temporal_kernels = temporal_kernel.unsqueeze(0).repeat(self.J, 1)
        self.register_buffer("temporal_kernels", temporal_kernels)

        # Batch normalization for temporal features (J channels)
        self.bn_temporal = nn.BatchNorm1d(self.J)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, t, n, n)

        Returns:
            torch.Tensor: Spatiotemporal features of shape (batch_size, J, t-T+1)
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
        # Input: (batch_size, J, t)
        # Output: (batch_size, J, t-T+1)
        temporal_features = F.conv1d(
            spatial_features,
            self.get_temporal_kernels().unsqueeze(1),
            groups=self.J,
        )

        # Apply batch normalization then activation (standard order)
        # BN expects (batch_size, channels, ...), which matches (batch_size, J, t-T+1)
        normalized_features = self.bn_temporal(temporal_features)
        features = F.softplus(normalized_features)

        return features

    def get_temporal_kernels(self) -> torch.Tensor:
        """
        Get temporal kernels as a 2D tensor.
        Returns:
            torch.Tensor: Temporal kernels of shape (2*J, T)
        """
        return self.temporal_kernels
