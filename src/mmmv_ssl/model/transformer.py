from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class TransformerBlockConfig:
    n_layers: int
    d_model: int
    d_in: int
    dropout = 0.1
    norm_first: bool = False
    n_head: int = 1


class TransformerBlock(nn.Module):
    """
    Transformer Encoder nn.Module
    """

    def __init__(self, config: TransformerBlockConfig):
        """
        Number of layers
        """
        super().__init__()
        self.nhead = config.n_head
        self.layer_stack = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    dim_feedforward=config.d_in,
                    nhead=config.n_head,
                    batch_first=True,
                    dropout=config.dropout,
                    norm_first=config.norm_first,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(
        self,
        data: Tensor,
        key_padding_mask: None | Tensor = None,
        src_mask: None | Tensor = None,
    ) -> Tensor:
        """
        input: b,n,c
        key_padding_mask: b,n or n : true means ignored
        src_mask: b*nhead,n,n
        """

        enc_output: Tensor = self.dropout(data)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                src_key_padding_mask=key_padding_mask,
                src_mask=src_mask,
            )
        return enc_output
