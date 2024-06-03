import torch
from einops import rearrange
from torch import Tensor
from torch import nn as nn


def MLP(d_in: int, d_out: int, l_dim: list, norm_layer="batch_norm"):
    """
    from https://github.com/facebookresearch/VICRegL/blob/main/utils.py
    :param mlp:
    :type mlp:
    :param embedding:
    :type embedding:
    :param norm_layer:
    :type norm_layer:
    :return:
    :rtype:
    """

    layers = []
    f = [d_in] + l_dim + [d_out]
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if norm_layer == "batch_norm":
            layers.append(nn.BatchNorm1d(f[i + 1]))
        elif norm_layer == "layer_norm":
            layers.append(nn.LayerNorm(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class MMProjector(nn.Module):
    def __init__(self, proj1: nn.Module, proj2: nn.Module):
        super().__init__()
        self.proj2 = proj2
        self.proj1 = proj1

    def forward(self, repr_1, repr_2):
        return self.proj1(repr_1), self.proj2(repr_2)


class ProjectorTemplate(nn.Module):
    def __init__(self, input_channels: int, out_channels: int):
        super().__init__()
        self.input_channels: input_channels
        self.out_channels = out_channels


class IdentityProj(ProjectorTemplate):
    def __init__(self, input_channels: int = 0, out_channels: int = 0):
        super().__init__(input_channels, out_channels)

    def forward(self, batch: Tensor):
        """

        Args:
            batch (): b,t,c,h,w

        Returns:

        """
        return batch


class AliseProj(ProjectorTemplate):
    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        l_dim: list = None,
        freeze=False,
    ):
        super().__init__(input_channels, out_channels)
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
                    nn.Linear(self.l_dim[-2], self.l_dim[-1], bias=False),
                ]
            )
        self.layers = nn.Sequential(*layers)
        if self.freeze:
            for param in self.layers.parameters():
                param.requires_grad = False

    def forward(self, batch: Tensor):
        b = batch.shape[0]
        if self.freeze:
            with torch.no_grad():
                batch = self.layers(rearrange(batch, "b t c -> (b t ) c "))
        else:
            batch = self.layers(rearrange(batch, "b t c -> (b t ) c "))
        return rearrange(batch, "(b t ) c -> b t c", b=b)
