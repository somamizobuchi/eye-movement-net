import torch
import torch.functional as F
import numpy as np

from utils import repeat_first_frame


class Encoder(torch.nn.Module):
    def __init__(self,
                 spatial_kernel_size: int = 32,
                 temporal_kernel_length: int = 32,
                 n_kernels: int = 32,
                 f_samp_hz: float = 240,
                 ):
        super().__init__()
        self.kernel_size = spatial_kernel_size
        self.kernel_length = temporal_kernel_length
        self.n_kernels = n_kernels
        self.fs = f_samp_hz

        self.spatial_kernels = torch.nn.Parameter(torch.zeros([self.kernel_size**2, self.n_kernels]).float())
        self.temporal_kernels = torch.nn.Parameter(torch.zeros([n_kernels, self.kernel_length]).float())
        self.neuron_weights = torch.nn.Parameter(torch.zeros([self.n_kernels, self.kernel_size**2]));

        self.activation = torch.nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.st_conv(x)
        return x

    
    # Spatio-temporal (separable) convolution
    def st_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Repeat first frame for valid convolution
        x = repeat_first_frame(x, self.kernel_length-1)

        # Spatial convolution (i.e. dot product with kernels)
        # (T, X*Y) @ (X*Y, K) = (T, K)
        y = x.view(*x.shape[0:1], -1) @ self.spatial_kernels

        # Temporal convolution (kernel-wise)
        out = torch.nn.functional.conv1d(y.T, self.temporal_kernels.unsqueeze(1), groups=self.n_kernels)

        # Apply weights to each kernel at every time point to up-sample each frame to match
        # the original frame dimensions
        out = out.T @ self.neuron_weights
        
        return out.view(-1, self.kernel_size, self.kernel_size)