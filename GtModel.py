import torch
import torch.nn as nn
from gt_encoder import GtEncoder
from eye_trace_decoder import TemporalVelocityDecoder
from recon_decoder import ReconstructionDecoder


class GtModel(nn.Module):
    """
    Video encoder with bifurcated decoding for both velocity prediction and reconstruction.
    Uses GtEncoder with biologically-inspired RGC kernels.

    Architecture:
    - GtEncoder: Hexagonal grid with RGC spatial RFs and temporal impulse response kernels
    - Velocity decoder: Predicts eye movement velocities
    - Reconstruction decoder: Reconstructs original frames

    Key features:
    - Biologically-inspired spatial and temporal kernels
    - Hexagonal sampling pattern
    - Center-surround (original + inverted) kernel pairs
    - Dual output heads for velocity and reconstruction
    """

    def __init__(
        self,
        kernel_size: int,
        kernel_length: int,
        spacing: float,
        ppd: float,
        fs: float,
        decoder_size: int,
        eccentricity: float = 0.3,
        cell_type: str = "M",
        noise_std: float = 0.05,
        max_velocity: float = 10.0,
    ):
        super(GtModel, self).__init__()

        self.N = kernel_size
        self.T = kernel_length
        self.K = decoder_size
        self.max_velocity = max_velocity

        # GtEncoder with biologically-inspired kernels
        self.encoder = GtEncoder(
            kernel_size=kernel_size,
            kernel_length=kernel_length,
            spacing=spacing,
            ppd=ppd,
            fs=fs,
            eccentricity=eccentricity,
            cell_type=cell_type,
            noise_std=noise_std,
        )

        # J is automatically set by GtEncoder (J = 2*K where K is number of grid points)
        self.J = self.encoder.J

        # Velocity decoder - predicts eye movements
        self.velocity_decoder = TemporalVelocityDecoder(
            in_channels=self.J,
            hidden_channels=decoder_size,
            out_channels=2,
            max_velocity=max_velocity,
        )

        # Reconstruction decoder - reconstructs frames
        self.recon_decoder = ReconstructionDecoder(
            n_channels=self.J,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        """
        Forward pass through the model with bifurcated output.

        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, t, n, n)

        Returns:
            tuple: (eye_velocities, reconstructed_frames)
                - eye_velocities: (batch_size, t-T+1, 2)
                - reconstructed_frames: (batch_size, t-T+1, N, N)
        """
        # Apply GtEncoder
        # features: (batch_size, J, t-T+1)
        features = self.encoder(x)

        # Velocity decoding
        eye_velocities = self.velocity_decoder(features)  # (batch_size, t_out, 2)

        # Reconstruction decoding
        reconstructed_frames = self.recon_decoder(features)  # (batch_size, t_out, N, N)

        return eye_velocities, reconstructed_frames

    def get_temporal_kernels(self) -> torch.Tensor:
        """Get temporal kernels from the encoder."""
        return self.encoder.get_temporal_kernels()
