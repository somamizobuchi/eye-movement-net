import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Modular (2+1)D convolution encoder.

    Architecture:
    1. Input: video of shape (batch_size, t, n, n)
    2. Spatial convolution: dot product with J kernels of size (nxn)
    3. Temporal convolution: 1D convolution with J kernels of length T
    4. Batch normalization
    5. Softplus activation
    6. Output: spatiotemporal features (batch_size, J, t-T+1)

    Args:
        kernel_size (int): Spatial dimension of input video (nxn frames)
        kernel_length (int): Temporal kernel size (T)
        kernel_delay (int): Delay in temporal kernels
        n_channels (int): Number of spatiotemporal channels (J)
        noise_std (float): Standard deviation of noise added during training
    """

    def __init__(
        self,
        kernel_size: int,
        kernel_length: int,
        kernel_delay: int,
        n_channels: int,
        noise_std: float = 0.05,
    ):
        super(Encoder, self).__init__()

        self.N = kernel_size
        self.T = kernel_length
        self.J = n_channels
        self.kernel_delay = kernel_delay
        self.noise_std = noise_std

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

        # Batch normalization for temporal features
        self.bn_temporal = nn.BatchNorm1d(n_channels)

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
            torch.Tensor: Temporal kernels of shape (J, T)
        """
        return F.pad(self.temporal_kernels, (0, self.kernel_delay), "constant", 0)
