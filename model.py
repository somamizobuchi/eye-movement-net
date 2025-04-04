import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_kernel_size: int = 32,
        temporal_kernel_length: int = 32,
        n_kernels: int = 32,
        f_samp_hz: float = 240,
        temporal_pad: Tuple[int, int] = (7, 2),
    ):
        super().__init__()
        self.kernel_size = spatial_kernel_size
        self.kernel_length = temporal_kernel_length
        self.n_kernels = n_kernels
        self.fs = f_samp_hz
        self.temporal_pad = temporal_pad
        # Soft-plus activation second dim [beta, threshold]
        # self.softplus_params = torch.nn.Parameter(torch.zeros([n_kernels, 2]))

        self.spatial_kernels = torch.nn.Parameter(
            torch.full([self.n_kernels, self.kernel_size**2], 0.0)
        )
        self.temporal_kernels = torch.nn.Parameter(
            torch.full([n_kernels, self.kernel_length - sum(self.temporal_pad)], 0.0)
        )
        # self.spatial_decoder = torch.nn.Linear(
        #     n_kernels, self.kernel_size**2, bias=False
        # )
        self.spatial_decoder = torch.nn.Parameter(
            torch.full([self.n_kernels, self.kernel_size**2], 0.0)
        )
        # self.temporal_decoder = torch.nn.Parameter(
        #     torch.full([self.n_kernels, 45], 0.0)
        # )

        # initialize parameters
        torch.nn.init.xavier_normal_(self.spatial_kernels)
        torch.nn.init.xavier_normal_(self.temporal_kernels)
        torch.nn.init.xavier_normal_(self.spatial_decoder)
        # torch.nn.init.xavier_normal_(self.temporal_decoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.st_conv(x)

    # Spatio-temporal (separable) convolution
    def st_conv(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial convolution (i.e. dot product with kernels)
        # (B, T, X*Y) @ (X*Y, K) = (T, K)
        x = input.reshape(*input.shape[0:2], -1) @ self.spatial_kernels.T.unsqueeze(0)

        # Temporal convolution (kernel-wise)
        x = F.conv1d(
            x.transpose(1, -1),
            self.pad_temporal().unsqueeze(1),
            groups=self.n_kernels,
        )

        x = torch.nan_to_num(x, 1e-6)

        # Apply non-linearity
        x = F.relu(x)
        x = x + 0.1 * torch.randn_like(x)
        # x = x + torch.poisson(x)

        encoder_output = x.clone()

        # Decoder
        # x = x * self.temporal_decoder.unsqueeze(0)
        x = x.transpose_(1, -1) @ self.spatial_decoder

        # Reshape to original frame dimensions (B, T, X, Y)
        x = x.view(x.shape[0], -1, self.kernel_size, self.kernel_size)

        return (x, encoder_output)

    def pad_temporal(self) -> torch.Tensor:
        return torch.nn.functional.pad(
            self.temporal_kernels, self.temporal_pad, "constant", 0
        )

    def temporal_kernels_full(self, input: torch.Tensor) -> torch.Tensor:
        return torch.concat(
            (
                input,
                torch.zeros(
                    [self.n_kernels, self.temporal_delay],
                    dtype=torch.float,
                    device=input.device,
                ),
            ),
            dim=1,
        )

    def kernel_spatial_jerk(self):
        padded = F.pad(
            self.spatial_kernels.view(-1, self.kernel_size, self.kernel_size),
            (1, 0, 1, 0),
            mode="constant",
            value=0,
        )
        padded_decoder = F.pad(
            self.spatial_decoder.view(-1, self.kernel_size, self.kernel_size),
            (1, 0, 1, 0),
            mode="constant",
            value=0,
        )
        out = torch.mean(
            torch.sqrt(
                padded.diff(dim=1).square()[:, :, 1:]
                + padded.diff(dim=2).square()[:, 1:]
            )
        ) + torch.mean(
            torch.sqrt(
                padded_decoder.diff(dim=1).square()[:, :, 1:]
                + padded_decoder.diff(dim=2).square()[:, 1:]
            )
        )

        if torch.isnan(out):
            return torch.zeros(1)

        return out

    def kernel_temporal_jerk(self):
        padded = F.pad(self.temporal_kernels, (1, 0), mode="constant", value=0)
        return (
            padded.diff(
                dim=1,
            )
            .abs()
            .mean(dim=1)
            .mean()
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
