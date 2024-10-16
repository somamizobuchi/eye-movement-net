import torch
import torch.nn.functional as F
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
        # Soft-plus activation second dim [beta, threshold]
        self.softplus_params = torch.nn.Parameter(torch.zeros([n_kernels, 2]))

        self.spatial_kernels = torch.nn.Parameter(torch.zeros([self.kernel_size**2, self.n_kernels]).float())
        self.temporal_kernels = torch.nn.Parameter(torch.zeros([n_kernels, self.kernel_length]).float())
        self.neuron_weights = torch.nn.Parameter(torch.zeros([self.n_kernels, self.kernel_size**2]));


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.st_conv(x)
        return x

    def initialize(self):
        torch.nn.init.kaiming_normal_(self.spatial_kernels)
        torch.nn.init.kaiming_normal_(self.temporal_kernels)
        torch.nn.init.kaiming_normal_(self.neuron_weights)

        # Initialize to the default params (see torch.nn.functional.softplus)
        self.softplus_params = torch.nn.Parameter(
            torch.tensor([1., 20.]).repeat([self.n_kernels, 1]))

    
    # Spatio-temporal (separable) convolution
    def st_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Repeat first frame for valid convolution
        # x = repeat_first_frame(x, self.kernel_length-1)

        # Spatial convolution (i.e. dot product with kernels)
        # (B, T, X*Y) @ (X*Y, K) = (T, K)
        x = x.view(*x.shape[0:2], -1) @ self.spatial_kernels.unsqueeze(0)

        # Temporal convolution (kernel-wise)
        # out = (K, T)
        x = F.conv1d(x.transpose(1, -1), self.temporal_kernels.unsqueeze(1), groups=self.n_kernels)

        # y = x.clone()
        # # Separate non-linearities for each channel
        # for batch in range(x.shape[0]):
        #     for i in range(self.n_kernels):
        #         y[batch,i,:] = torch.nn.functional.softplus(
        #             y[batch,i,:], 
        #             beta=self.softplus_params[i, 0].item(), 
        #             threshold=self.softplus_params[i, 1].item())
        # x = y

        x = F.softplus(x)

        # Add poisson noise
        noise = torch.poisson(x)
        x = x + noise

        # Apply weights to each kernel at every time point to up-sample each frame to match
        # the original frame dimensions
        x = x.transpose(-1, 1) @ self.neuron_weights
        
        return x.view(x.shape[0], -1, self.kernel_size, self.kernel_size)