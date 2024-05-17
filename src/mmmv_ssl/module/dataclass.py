from dataclasses import dataclass

from torch import Tensor

from mmmv_ssl.module.loss import GlobalInvRecMMLoss


@dataclass
class LatRepr:
    s1a: Tensor
    s1b: Tensor
    s2a: Tensor
    s2b: Tensor


@dataclass
class RecWithOrigin:
    same_mod: Tensor
    other_mod: Tensor


@dataclass
class Rec:
    s1a: RecWithOrigin
    s1b: RecWithOrigin
    s2a: RecWithOrigin
    s2b: RecWithOrigin


@dataclass
class OutMMAliseF:
    repr: LatRepr
    rec: Rec


@dataclass
class OutMMAliseSharedStep:
    loss: GlobalInvRecMMLoss
    out_forward: OutMMAliseF
