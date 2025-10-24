import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionDecoder(nn.Module):
    """
    Reconstruction decoder that projects spatiotemporal features back to image space.

    Architecture:
    - Input: spatiotemporal features (batch_size, J, t)
    - Linear projection: J → N²
    - Output: reconstructed frames (batch_size, t, N, N)

    Args:
        n_channels (int): Number of input spatiotemporal channels (J)
        kernel_size (int): Spatial dimension of output frames (N×N)
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
    ):
        super(ReconstructionDecoder, self).__init__()

        self.J = n_channels
        self.N = kernel_size

        # Linear projection from spatiotemporal features to image space
        # Shape: (J, N²)
        self.spatial_decoder = nn.Parameter(
            torch.empty(n_channels, kernel_size * kernel_size)
        )
        nn.init.xavier_normal_(self.spatial_decoder)

    def forward(self, x):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Spatiotemporal features of shape (batch_size, J, t)

        Returns:
            torch.Tensor: Reconstructed frames of shape (batch_size, t, N, N)
        """
        # x: (batch_size, J, t)
        batch_size, n_channels, t = x.shape

        # Transpose for matrix multiplication: (batch_size, t, J)
        x = x.transpose(1, 2)

        # Linear projection: (batch_size, t, J) @ (J, N²) = (batch_size, t, N²)
        x = torch.matmul(x, self.spatial_decoder)

        # Reshape to frame dimensions: (batch_size, t, N, N)
        x = x.view(batch_size, t, self.N, self.N)

        return x
