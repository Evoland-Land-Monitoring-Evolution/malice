from dataclasses import dataclass

from torch import Tensor


@dataclass
class OneViewRecL:
    monom_rec: Tensor
    crossm_rec: Tensor

    def to_dict(self, suffix: str = "") -> dict:
        return {
            f"{suffix}_crossmrec": self.crossm_rec.item(),
            f"{suffix}_monomrec": self.monom_rec.item(),
        }

    def compute(self, w_cross_rec) -> Tensor:
        return (1 -
                w_cross_rec) * self.monom_rec + w_cross_rec * self.crossm_rec


@dataclass
class TotalRecLoss:
    s2_a: OneViewRecL
    s2_b: OneViewRecL
    s1_a: OneViewRecL
    s1_b: OneViewRecL

    def to_dict(self, suffix="train"):
        d = {}
        d.update(self.s1_a.to_dict(suffix=f"{suffix}_s1a"))
        d.update(self.s1_b.to_dict(suffix=f"{suffix}_s1b"))
        d.update(self.s2_a.to_dict(suffix=f"{suffix}_s2a"))
        d.update(self.s2_b.to_dict(suffix=f"{suffix}_s2b"))
        return d

    def compute(self, w_cross_rec) -> Tensor:
        loss = self.s2_a.compute(w_cross_rec)
        loss += self.s2_b.compute(w_cross_rec)
        loss += self.s1_a.compute(w_cross_rec)
        loss += self.s1_b.compute(w_cross_rec)
        return loss / 8


@dataclass
class GlobalInvRecMMLoss:
    total_rec_loss: TotalRecLoss
    inv_loss: Tensor | None = None
    w_inv: float = 1
    w_rec: float = 1
    w_crossrec: float = 1

    def to_dict(self, suffix="train"):
        if self.inv_loss is not None:
            inv_loss = self.inv_loss.item()
        else:
            inv_loss = 0
        d = {f"{suffix}_invloss": inv_loss}
        d.update(self.total_rec_loss.to_dict(suffix=suffix))
        total = self.compute()
        d.update({f"{suffix}_total_loss": total.item()})
        return d

    def compute(self):
        if self.inv_loss is not None:
            loss = self.w_inv * self.inv_loss
        else:
            loss = 0
        loss += self.w_rec * self.total_rec_loss.compute(
            w_cross_rec=self.w_crossrec)
        return loss
