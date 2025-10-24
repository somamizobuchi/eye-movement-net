import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalVelocityDecoder(nn.Module):
    """
    Temporal decoder for per-frame 2D velocity prediction.
    Uses 1D convolutions over the temporal dimension to incorporate context.
    """

    def __init__(self, in_channels, hidden_channels=128, out_channels=2):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: (B, J, T_out)
        Returns:
            velocities: (B, T_out, 2)
        """
        x = self.temporal_net(features)  # (B, hidden, T_out)
        v = self.head(x)  # (B, 2, T_out)
        v = v.permute(0, 2, 1)  # → (B, T_out, 2)
        return v
