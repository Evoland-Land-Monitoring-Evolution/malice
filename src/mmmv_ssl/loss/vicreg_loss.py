import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

my_logger = logging.getLogger(__name__)


class VarianceLoss(nn.Module):
    def forward(self, x: Tensor, y: Tensor):
        """
        Inspired by https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        Args:
            x (): tensor dim b, t, c ,... mean along the channel dim batch dimension should be 0
            y (): tensor dim b,t,c  mean along the channel dim batch dimension should be 0

        Returns:
        Tensor of shape t,...
        """
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        return (
            torch.mean(F.relu(1 - std_x)) / 2
            + torch.mean(F.relu(1 - std_y)) / 2
        )


class CovarianceLoss(nn.Module):
    def __init__(self, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features

    def forward(self, x: Tensor, y: Tensor, batch_size: int):
        """
        Inspired by https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        Args:
            x (): tensor dim b,t,c mean along the batch dimension should be 0
            y (): tensor dim b,t,c mean along the batch dimension should be 0

        Returns:

        """
        x_1 = rearrange(x, "b t c h w -> t h w  c b ")
        y_1 = rearrange(y, "b t c h w -> t h w c b")
        x_T = rearrange(x, "b t c h w -> t h w  b c")
        y_T = rearrange(y, "b t c h w -> t h w  b c")
        t, h, w, b, c = y_T.shape
        cov_x = (x_1 @ x_T) / (batch_size - 1)  # shape t,h,w,c,c
        cov_y = (y_1 @ y_T) / (batch_size - 1)  # shape t,h,w,c,c
        assert cov_x.shape[-1] == c
        my_logger.debug(f" cov mat {cov_x.shape}")
        return off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)


def off_diagonal(x):
    t, h, w, n, m = x.shape
    assert n == m
    x = rearrange(x, "t h w m n  -> (t h w) (m n )")[..., :-1]
    x = rearrange(x, "thw (m n )-> thw m n ", m=n - 1, n=n + 1)[:, :, 1:]
    return rearrange(x, "thw m n  -> thw (m n)")
    # return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@dataclass
class VicRegLoss:
    inv_loss: Tensor | None
    var_loss: Tensor | None
    cov_loss: Tensor | None
    w_inv: int = 25
    w_var: int = 25
    w_cov: int = 1

    def to_dict(self, suffix="train"):
        return {
            f"{suffix}_inv_loss": self.inv_loss.item(),
            f"{suffix}_var_loss": self.var_loss.item(),
            f"{suffix}_cov_loss": self.cov_loss.item(),
        }

    def total_loss(self):
        return (
            self.inv_loss * self.w_inv
            + self.var_loss * self.w_var
            + self.cov_loss * self.w_cov
        )


@dataclass
class TotalLoss:
    bottleneck_loss: VicRegLoss
    local_loss: VicRegLoss
    w_bottleneck: float
    w_local: float

    def to_dict(self, suffix: str = "train"):
        dict = self.bottleneck_loss.to_dict(suffix=f"{suffix}_bottleneck")
        dict.update(self.local_loss.to_dict(suffix=f"{suffix}_local"))
        return dict

    def total_loss(self):
        return (
            self.w_bottleneck * self.bottleneck_loss.total_loss()
            + self.w_local * self.local_loss.total_loss()
        )
