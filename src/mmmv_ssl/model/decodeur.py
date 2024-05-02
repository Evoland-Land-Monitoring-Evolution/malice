import torch.nn as nn
from hydra.utils import instantiate
from mt_ssl.model.attention import FlexibleLearnedQMultiHeadAttention
from mt_ssl.model.transformer import Encoder
from omegaconf import DictConfig
from torch import Tensor


class MetaDecoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        input_channels: int,
        d_in: int,
        intermediate_layers: DictConfig | Encoder = None,
    ):
        super().__init__()
        self.cross_attn = FlexibleLearnedQMultiHeadAttention(
            n_head=num_heads, d_k=input_channels, d_in=d_in
        )
        self.input_channels = input_channels
        if intermediate_layers is None:
            self.intermediate_layers = None
        elif isinstance(intermediate_layers, Encoder):
            self.intermediate_layers = intermediate_layers
        elif isinstance(intermediate_layers, DictConfig):
            self.intermediate_layers = instantiate(
                intermediate_layers,
                input_channels=input_channels,
                output_channels=input_channels,
            )
        else:
            raise NotImplementedError

    def forward(self, mm_sits: Tensor, padd_mm: Tensor, mm_queries: Tensor):
        """

        Args:
            mm_sits (): b,t,c
            padd_mm (): b,t
            mm_queries (): b,t,c

        Returns:

        """
        out_mm = self.cross_attn(v=mm_sits, q=mm_queries, pad_mask=None)
        if self.intermediate_layers is not None:
            out_mm = self.intermediate_layers(
                out_mm, key_padding_mask=padd_mm
            )  # so the padded dates do not interfere during
            # SA
        return out_mm
