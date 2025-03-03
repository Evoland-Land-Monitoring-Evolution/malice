# pylint: disable=invalid-name

"""
Embedding projectors for invariance loss.
If no invarince loss is used, we utilize IdentityProj(),
otherwise AliseProj()
"""

import torch
from einops import rearrange
from torch import nn


class IdentityProj(nn.Module):
    """
    Projector of embeddings for invariance loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """ batch (): b,t,c,h,w  """
        return batch


class AliseProj(nn.Module):
    """Projector of embeddings for invariance loss"""
    def __init__(
            self,
            input_channels: int,
            out_channels: int,
            l_dim: list | None = None,
            freeze: bool = False,
    ):
        super().__init__()
        self.input_channels: input_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.l_dim = l_dim
        if self.l_dim is not None:
            self.l_dim = [input_channels] + self.l_dim + [out_channels]
        else:
            self.l_dim = [input_channels] + [out_channels]
        layers = []
        for i in range(len(self.l_dim) - 2):
            layers.extend(
                [
                    nn.Linear(self.l_dim[i], self.l_dim[i + 1]),
                    nn.BatchNorm1d(self.l_dim[i + 1]),
                    nn.ReLU(True),
                ]
            )
        layers.extend([nn.Linear(self.l_dim[-2], self.l_dim[-1], bias=False)])
        self.layers = nn.Sequential(*layers)
        if self.freeze:
            for param in self.layers.parameters():
                param.requires_grad = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        b = batch.shape[0]
        if self.freeze:
            with torch.no_grad():
                batch = self.layers(rearrange(batch, "b t c -> (b t ) c "))
        else:
            batch = self.layers(rearrange(batch, "b t c -> (b t ) c "))
        return rearrange(batch, "(b t ) c -> b t c", b=b)
