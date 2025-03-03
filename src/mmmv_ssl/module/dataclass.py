"""Dataclasses for losses"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class LatRepr:
    """Latent representations of views"""
    s1a: Tensor
    s1b: Tensor
    s2a: Tensor
    s2b: Tensor


@dataclass
class RecWithOrigin:
    """Reconstruction of one view"""
    same_mod: Tensor
    other_mod: Tensor


@dataclass
class Rec:
    """All reconstructions of all views"""
    s1a: RecWithOrigin
    s1b: RecWithOrigin
    s2a: RecWithOrigin
    s2b: RecWithOrigin


@dataclass
class OutMMAliseF:
    """Output of malice module"""
    repr: LatRepr
    rec: Rec
    emb: LatRepr


@dataclass
class DespeckleS1:
    """
    Despeckled S1 views
    """
    s1a: Tensor
    s1b: Tensor



@dataclass
class OneViewRecL:
    """
    Reconstruction losses for each view
    """
    monom_rec: Tensor
    crossm_rec: Tensor

    def to_dict(self, suffix: str = "") -> dict:
        """Dataclass to dict"""
        return {
            f"{suffix}_crossmrec": self.crossm_rec,
            f"{suffix}_monomrec": self.monom_rec,
        }


@dataclass
class TotalRecLoss:
    """
    Total reconstruction loss for each view
    """
    s2_a: OneViewRecL
    s2_b: OneViewRecL
    s1_a: OneViewRecL
    s1_b: OneViewRecL

    def to_dict(self):
        """Dataclass to dictionary"""
        dictionary = {}
        dictionary.update(self.s1_a.to_dict(suffix="s1a"))
        dictionary.update(self.s1_b.to_dict(suffix="s1b"))
        dictionary.update(self.s2_a.to_dict(suffix="s2a"))
        dictionary.update(self.s2_b.to_dict(suffix="s2b"))
        return dictionary


@dataclass
class GlobalInvRecMMLoss:
    """Global loss"""
    total_rec_loss: TotalRecLoss
    inv_loss: Tensor | None = None

@dataclass
class OutMMAliseSharedStep:
    """Output shared step"""
    loss: GlobalInvRecMMLoss
    out_forward: OutMMAliseF
    despeckle_s1: DespeckleS1

@dataclass
class WeightClass:
    """Loss weights"""
    w_rec: float
    w_inv: float
    w_crossrec: float
