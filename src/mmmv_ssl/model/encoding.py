# pylint: disable=invalid-name

"""
Positional Encoder
"""

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """
    Positional encoder class
    """

    def __init__(self, d: int, T: int = 1000, repeat: int | None = None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode DOYs
        """
        self.denom = self.denom.to(batch_positions.device)
        if not self.updated_location:
            self.updated_location = True
        sinusoid_table = (
                batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(
            sinusoid_table[:, :, 0::2]
        )  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(
            sinusoid_table[:, :, 1::2]
        )  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table
