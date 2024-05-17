import torch
import torch.nn as nn
from einops import repeat
from hydra.utils import instantiate
from mt_ssl.model.encoding import PEConfig
from omegaconf import DictConfig

from mmmv_ssl.model.encoding import PositionalEncoder


class TempMetaQuery(nn.Module):
    def __init__(self, pe_config, input_channels):
        super().__init__()
        if isinstance(pe_config, DictConfig | PEConfig):
            self.pe_encoding: PositionalEncoder = instantiate(
                pe_config, d=input_channels
            )
        elif isinstance(pe_config, PositionalEncoder):
            self.pe_encoding: PositionalEncoder = pe_config
        else:
            raise NotImplementedError

    def forward(self, q, doy):
        """

        Args:
            q (): d
            doy (): b,t

        Returns:

        """
        pe = self.pe_encoding(doy)  # b t,c
        query = repeat(q, " c -> b t c", b=pe.shape[0], t=pe.shape[1])
        return torch.cat([pe, query], dim=-1)
