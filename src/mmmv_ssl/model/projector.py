import torch.nn as nn


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
