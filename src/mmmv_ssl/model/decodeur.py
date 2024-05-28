import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor

from mmmv_ssl.model.transformer import TransformerBlock

my_logger = logging.getLogger(__name__)


class LearnedQMultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    The learnable queries has no temporal dimension
    """

    def __init__(self, n_head: int, d_k: int, d_in: int, d_q_in: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.fc1_k = nn.Linear(d_in, n_head * d_k, bias=False)
        self.fc1_q = nn.Linear(d_q_in, n_head * d_k, bias=False)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

    def forward(self, v, q, pad_mask=None):
        """

        Args:
            v (): b,t,c
            q: queries
            pad_mask (): b,t, true means the value should take part in attention

        Returns:

        """
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = v.size()
        # Q=self.Q[:,None,:].expand(-1,t,-1)
        # q = torch.stack([Q for _ in range(sz_b)], dim=1)
        my_logger.debug(f"query{q.shape}")
        # q = rearrange(q, "head b nq c -> b head nq c")
        my_logger.debug(f"in forward q {q.shape}")
        q = self.fc1_q(rearrange(q, " nh b t c -> b t (nh c)"))  # b h nq c
        my_logger.debug(f"in forward q {q.shape}")
        q = rearrange(q, "b t (nh c)-> b nh t c", nh=self.n_head)
        my_logger.debug(f"in forward q {q.shape}")
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        my_logger.debug(f"key {k.shape}")
        k = rearrange(k, "b t head c -> b head t c")
        my_logger.debug(f"key {k.shape}")
        if pad_mask is not None:
            pad_mask = pad_mask[..., None, None]  # b,t,1,1
            pad_mask = pad_mask.expand(-1, -1, n_head, self.n_q)
            pad_mask = rearrange(pad_mask, "b t head nq ->b head nq t")
            my_logger.debug(f"Pad mask shape {pad_mask.shape}")
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1))
        v = rearrange(v, "head b t c -> b head t c")

        my_logger.debug(f"value {v.shape}")
        my_logger.debug(f"key {k.shape}")
        my_logger.debug(f"query{q.shape}")
        # q=q.to(v)
        output = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=pad_mask)  # B,h,nq,d_in
        my_logger.debug(f"output {output.shape}")
        return rearrange(output, "b h nq c -> b nq (h c)")


class MetaDecoder(nn.Module):

    def __init__(
        self,
        num_heads: int,
        input_channels: int,
        d_k: int,
        d_q_in: int,
        intermediate_layers: DictConfig | TransformerBlock = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.cross_attn = LearnedQMultiHeadAttention(n_head=num_heads,
                                                     d_k=d_k,
                                                     d_in=input_channels,
                                                     d_q_in=d_q_in)
        self.input_channels = input_channels
        if intermediate_layers is None:
            self.intermediate_layers = None
        elif isinstance(intermediate_layers, TransformerBlock):
            self.intermediate_layers = intermediate_layers
        elif isinstance(intermediate_layers, DictConfig):
            transformer_config = instantiate(intermediate_layers.config)
            print(transformer_config)
            transformer_config.d_model = input_channels
            self.intermediate_layers = TransformerBlock(transformer_config)
        else:
            raise NotImplementedError

    def forward(self, mm_sits: Tensor, padd_mm: Tensor, mm_queries: Tensor):
        """

        Args:
            mm_sits (): b,t,c
            padd_mm (): b,t
            mm_queries (): nh, b,t,c

        Returns:

        """
        my_logger.debug(f"decodeur v={mm_sits.shape} q={mm_queries.shape}")
        out_mm = self.cross_attn(v=mm_sits, q=mm_queries, pad_mask=None)
        if self.intermediate_layers is not None:
            out_mm = self.intermediate_layers(
                out_mm,
                key_padding_mask=None  #padd_mm
            )  # so the padded dates do not interfere during
            # SA
        return out_mm
