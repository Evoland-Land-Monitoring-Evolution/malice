from dataclasses import dataclass

from torch import Tensor


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
    emb: LatRepr


@dataclass
class DespeckleS1:
    s1a: Tensor
    s1b: Tensor



@dataclass
class OneViewRecL:
    monom_rec: Tensor
    crossm_rec: Tensor

    def to_dict(self, suffix: str = "") -> dict:
        return {
            f"{suffix}_crossmrec": self.crossm_rec,
            f"{suffix}_monomrec": self.monom_rec,
        }


@dataclass
class TotalRecLoss:
    s2_a: OneViewRecL
    s2_b: OneViewRecL
    s1_a: OneViewRecL
    s1_b: OneViewRecL

    def to_dict(self):
        d = {}
        d.update(self.s1_a.to_dict(suffix="s1a"))
        d.update(self.s1_b.to_dict(suffix="s1b"))
        d.update(self.s2_a.to_dict(suffix="s2a"))
        d.update(self.s2_b.to_dict(suffix="s2b"))
        return d


@dataclass
class GlobalInvRecMMLoss:
    total_rec_loss: TotalRecLoss
    inv_loss: Tensor | None = None

@dataclass
class OutMMAliseSharedStep:
    loss: GlobalInvRecMMLoss
    out_forward: OutMMAliseF
    despeckle_s1: DespeckleS1
