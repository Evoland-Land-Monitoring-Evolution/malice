"""
Transformer block for deep decoding in Malice.
Currently not used in the default model configuration
"""
import torch
from torch import nn

from mmmv_ssl.model.datatypes import TransformerBlockConfig


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
            data: torch.Tensor,
            key_padding_mask: None | torch.Tensor = None,
            src_mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        """
        input: b,n,c
        key_padding_mask: b,n or n : true means ignored
        src_mask: b*nhead,n,n
        """

        enc_output: torch.Tensor = self.dropout(data)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                src_key_padding_mask=key_padding_mask,
                src_mask=src_mask,
            )
        return enc_output
