# pylint: disable=invalid-name

"""
Temporal projector
"""

import logging

import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
from mmmv_ssl.model.dataclass import OutTempProjForward

my_logger = logging.getLogger(__name__)


class CrossAttn(nn.Module):
    """
    Cross attention from Temporal Projector
    """
    def __init__(self, n_head: int, d_k: int, d_in: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.fc1_k = nn.Linear(d_in, n_head * d_k, bias=False)

    def forward(self,
                v: torch.Tensor,
                q: torch.Tensor,
                pad_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """

        :param v: b,t,c
        :type v:
        :param q: nq,dk*head
        :type q:
        :param pad_mask: b,t, True means Value in attention
        :type pad_mask:
        :return:
        :rtype:
        """
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = v.size()
        # q = torch.stack([q for _ in range(sz_b)], dim=1)
        q = repeat(q, "head nq c -> head b nq c", b=sz_b)
        my_logger.debug(f"query{q.shape}")
        q = rearrange(q, "head b nq c -> b head nq c")
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        my_logger.debug(f"key {k.shape}")
        k = rearrange(k, "b t head c -> b head t c")
        my_logger.debug(f"key {k.shape}")
        if pad_mask is not None:
            pad_mask = repeat(
                pad_mask, "b t -> b nhead nq t", nhead=n_head, nq=q.shape[2]
            )
            my_logger.debug(f"Pad mask shape {pad_mask.shape}")

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1))
        v = rearrange(v, "head b t c -> b head t c")
        my_logger.debug(f"value {v.shape}")
        my_logger.debug(f"query{q.shape}")
        my_logger.debug(f"key {k.shape}")
        # q=q.to(v)
        output = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=pad_mask
        )  # B,h,nq,d_in
        my_logger.debug(f"output {output.shape}")
        return rearrange(output, "b h nq c -> b nq (h c)")


class TemporalProjector(nn.Module):
    """
    Temporal projector class. Module from Malice encoder.
    """
    def __init__(self, num_heads: int, input_channels: int, n_q: int):
        super().__init__()
        self.Q = nn.Parameter(
            torch.zeros((num_heads, n_q, input_channels))
        ).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (input_channels)))
        self.ca_s1 = CrossAttn(
            n_head=num_heads, d_k=input_channels, d_in=input_channels
        )
        self.ca_s2 = CrossAttn(
            n_head=num_heads, d_k=input_channels, d_in=input_channels
        )

    def forward(
            self,
            sits_s1: torch.Tensor,
            padd_s1: torch.Tensor | None,
            sits_s2: torch.Tensor,
            padd_s2: torch.Tensor | None,
    ) -> OutTempProjForward:
        """
        :param sits_s1: b,t1,c
        :type sits_s1:
        :param padd_s1: b,t1 means value in attention False when padded
        :type padd_s1:
        :param sits_s2: b,t2,c
        :type sits_s2:
        :param padd_s2: b,t2 means value in attention False when padded
        :type padd_s2:True
        :return:
        :rtype:
        """
        align_s1 = self.ca_s1(v=sits_s1, q=self.Q, pad_mask=padd_s1)  # b nq dv
        align_s2 = self.ca_s2(v=sits_s2, q=self.Q, pad_mask=padd_s2)  # b nq dv
        return OutTempProjForward(s1=align_s1, s2=align_s2)
