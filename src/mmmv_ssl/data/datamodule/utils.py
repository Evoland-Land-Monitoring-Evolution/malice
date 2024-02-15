import torch.nn as nn
from einops import rearrange
from torch import Tensor


def apply_transform_basic(batch_sits: Tensor, transform: nn.Module) -> Tensor:
    b, *_ = batch_sits.shape
    batch_sits = rearrange(batch_sits, " b t c h w -> c (b t )  h w")
    batch_sits = transform(batch_sits)
    batch_sits = rearrange(batch_sits, "c (b t )  h w -> b t c h w", b=b)
    return batch_sits
