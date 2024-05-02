from dataclasses import dataclass

from torch import Tensor


@dataclass
class OutMMAliseF:
    repr_s1: Tensor
    repr_s2: Tensor
    pred_s1: Tensor
    pred_s2: Tensor
