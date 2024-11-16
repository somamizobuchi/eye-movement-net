import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any
from data import EMStats

from utils import repeat_first_frame


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_kernel_size: int = 32,
        temporal_kernel_length: int = 32,
        n_kernels: int = 32,
        f_samp_hz: float = 240,
        temporal_delay: int = 4,
    ):
        super().__init__()
        self.kernel_size = spatial_kernel_size
        self.kernel_length = temporal_kernel_length
        self.n_kernels = n_kernels
        self.fs = f_samp_hz
        self.temporal_delay = temporal_delay
        # Soft-plus activation second dim [beta, threshold]
        # self.softplus_params = torch.nn.Parameter(torch.zeros([n_kernels, 2]))

        self.spatial_kernels = torch.nn.Parameter(
            torch.full([self.kernel_size**2, self.n_kernels], 0.0)
        )
        self.temporal_kernels = torch.nn.Parameter(
            torch.full([n_kernels, self.kernel_length - self.temporal_delay], 0.0)
        )
        self.decoder = torch.nn.Linear(n_kernels, self.kernel_size**2)

        self.non_linear = ChannelLearnableReLU(n_kernels)

        # initialize parameters
        torch.nn.init.kaiming_normal_(self.spatial_kernels)
        torch.nn.init.kaiming_normal_(self.temporal_kernels)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.st_conv(x)

    # Spatio-temporal (separable) convolution
    def st_conv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial convolution (i.e. dot product with kernels)
        # (B, T, X*Y) @ (X*Y, K) = (T, K)
        x = x.to(self.spatial_kernels.device)
        x = x.view(*x.shape[0:2], -1) @ self.spatial_kernels.unsqueeze(0)

        # Temporal convolution (kernel-wise)
        # out = (K, T)
        x = F.conv1d(
            x.transpose(1, -1),
            self.temporal_kernels_full().unsqueeze(1),
            groups=self.n_kernels,
        )

        x = torch.nan_to_num(x, 1e-6)

        # x = F.softplus(x)
        x = self.non_linear(x)

        noise = torch.poisson(x)
        x = x + noise

        encoder_output = x.clone()

        # Apply weights to each kernel at every time point to up-sample each frame to match
        # the original frame dimensions
        x = self.decoder(x.transpose(-1, 1))

        x = x.view(x.shape[0], -1, self.kernel_size, self.kernel_size)

        return (x, encoder_output)

    def jerk_energy_loss(self):
        return self.temporal_kernels_full().diff(dim=1).square().mean()

    def temporal_kernels_full(self) -> torch.Tensor:
        return torch.concat(
            (
                self.temporal_kernels,
                torch.zeros(
                    [self.n_kernels, self.temporal_delay],
                    dtype=torch.float,
                    device=self.temporal_kernels.device,
                ),
            ),
            dim=1,
        )


class ChannelLearnableReLU(nn.Module):
    def __init__(self, n_channels, init_slope=1.0):
        super().__init__()
        # One parameter per channel
        self.slopes = nn.Parameter(torch.full((n_channels,), init_slope))

    def forward(self, x):
        # x shape: [batch_size, n_channels, m_inputs]
        # Reshape slopes to [1, n_channels, 1] for broadcasting
        slopes = self.slopes.view(1, -1, 1)
        return torch.maximum(torch.zeros_like(x), slopes * x)
