import torch
import torch.nn as nn


class V1Decoder(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, kernel_length: int, kernel_delay: int = 1):
        super().__init__()
        self.kernel_length = kernel_length
        self.kernel_delay = kernel_delay
        # Create temporal kernels with delay, padding will be added during forward pass
        self.temporal_kernels = nn.Parameter(torch.empty(input_dims, kernel_length - kernel_delay))
        self.spatial_kernels = nn.Linear(input_dims, output_dims)
        self.relu = nn.ReLU()

        nn.init.xavier_normal_(self.temporal_kernels)
        nn.init.kaiming_normal_(self.spatial_kernels.weight)

    def get_temporal_kernels(self) -> torch.Tensor:
        """
        Get temporal kernels with delay padding applied.
        Returns:
            torch.Tensor: Temporal kernels of shape (input_dims, kernel_length)
        """
        return torch.nn.functional.pad(
            self.temporal_kernels, (0, self.kernel_delay), "constant", 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying temporal convolution followed by spatial linear transformation.

        Args:
            x: Input tensor of shape (batch_size, input_dims, time)

        Returns:
            Output tensor of shape (batch_size, output_dims, time)
        """
        batch_size, input_dims, time = x.shape

        # Apply temporal convolution: convolve each channel with temporal kernels
        # temporal_kernels: (input_dims, kernel_length - kernel_delay)
        # x: (batch_size, input_dims, time)
        # Use depthwise convolution (groups=input_dims) to apply one kernel per channel
        # Get padded kernels with delay enforced at the beginning
        padded_kernels = self.get_temporal_kernels()  # (input_dims, kernel_length)

        # No padding - output size will be (time - kernel_length + 1)
        temporal_conv = torch.nn.functional.conv1d(
            x,  # (batch_size, input_dims, time)
            padded_kernels.unsqueeze(1),  # (input_dims, 1, kernel_length)
            groups=input_dims,  # Apply one filter per input channel
        )  # (batch_size, input_dims, time - kernel_length + 1)

        # Get the output time dimension after convolution
        time_out = temporal_conv.shape[2]

        # Apply linear transformation at each time point
        # Reshape to (batch_size * time_out, input_dims)
        temporal_conv_reshaped = temporal_conv.permute(0, 2, 1).reshape(-1, input_dims)

        # Apply linear layer
        output = self.spatial_kernels(temporal_conv_reshaped)  # (batch_size * time_out, output_dims)

        # Reshape back to (batch_size, time_out, output_dims) then permute to (batch_size, output_dims, time_out)
        output = output.reshape(batch_size, time_out, -1).permute(0, 2, 1)

        return output
