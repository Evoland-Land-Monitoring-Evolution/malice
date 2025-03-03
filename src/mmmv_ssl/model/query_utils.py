# pylint: disable=invalid-name

"""
Temporal Meta Query class for learnable queries in
Malice decoder
"""

import torch
from einops import repeat
from mmmv_ssl.model.encoding import PositionalEncoder
from torch import nn


class TempMetaQuery(nn.Module):
    """
    Temporal Meta Query class for learnable queries in
    Malice decoder
    """
    def __init__(self, input_channels: int, pe_config: PositionalEncoder | None = None):
        super().__init__()
        if isinstance(pe_config, PositionalEncoder):
            self.pe_encoding: PositionalEncoder = pe_config
        else:
            self.pe_encoding = PositionalEncoder(
                d=input_channels
            )

    def forward(self, q: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        pe = self.pe_encoding(doy)  # b t,c
        query = repeat(q, " c -> b t c", b=pe.shape[0], t=pe.shape[1])
        return torch.cat([pe, query], dim=-1)
