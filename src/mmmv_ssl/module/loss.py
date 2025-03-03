# pylint: disable=invalid-name

"""Loss functions"""

import torch
from einops import rearrange
from mt_ssl.module.loss import create_mask_loss
from torch import nn

from mmmv_ssl.data.dataclass import BatchOneMod, BatchMMSits
from mmmv_ssl.model.datatypes import DataInputChannels
from mmmv_ssl.module.dataclass import (
    RecWithOrigin, DespeckleS1, Rec, LatRepr,
    OneViewRecL, TotalRecLoss, GlobalInvRecMMLoss)
from mmmv_ssl.utils.speckle_filter import despeckle_batch


def despeckle(batch_sits: BatchOneMod) -> tuple[torch.Tensor, int]:
    """Despeckle S1 image for rec loss computation"""
    b, _, _, h, w = batch_sits.sits.shape  # pylint: disable=C0103
    despeckle_s1, margin = despeckle_batch(
        rearrange(batch_sits.sits, "b t c h w -> (b t ) c h w")
    )
    despeckle_s1 = rearrange(
        despeckle_s1, "(b t ) c h w -> b t c h w", b=b
    )[
                   ...,
                   margin: h - margin,
                   margin: w - margin,
                   ]
    return despeckle_s1, margin


class ReconstructionLoss:
    """"Reconstruction loss module"""

    def __init__(
            self,
            rec_loss: nn.Module = nn.MSELoss(),
            margin: int = 0,
            channels: DataInputChannels = DataInputChannels()
    ):
        self.rec_loss = rec_loss
        self.margin = margin
        self.channels = channels

    def compute_one_rec_loss(self,
                             rec_sits: RecWithOrigin,
                             sits: BatchOneMod,
                             original_sits: torch.Tensor,
                             return_desp: DespeckleS1 | None = None,
                             margin: int = 0,
                             mod: str = "s2"
                             ) -> OneViewRecL | tuple[None, DespeckleS1]:
        """
        Compute reconstruction loss for one view.
        Returns dataclass with direct reconstruction loss and cross reconstruction loss.
        """
        valid_mask = create_mask_loss(
            sits.padd_index, ~sits.mask
        )  # in .mask True means pixel valid
        h, w = sits.h, sits.w  # pylint: disable=C0103
        if mod == "s2":
            b = self.channels.s2
        else:
            b = self.channels.s1
        if torch.sum(valid_mask) != 0:
            valid_mask = valid_mask[
                         ...,
                         margin: h - margin,
                         margin: w - margin,
                         ]
            return OneViewRecL(
                monom_rec=self.rec_loss(
                    torch.masked_select(rec_sits.same_mod[
                                        ...,
                                        margin: h - margin,
                                        margin: w - margin,
                                        ],
                                        valid_mask),
                    torch.masked_select(original_sits[:, :, :b], valid_mask),
                ),
                crossm_rec=self.rec_loss(
                    torch.masked_select(rec_sits.other_mod[
                                        ...,
                                        margin: h - margin,
                                        margin: w - margin,
                                        ],
                                        valid_mask),
                    torch.masked_select(original_sits[:, :, :b], valid_mask),
                ),
            )
        return None, return_desp

    def compute_rec_loss(
            self, batch: BatchMMSits, rec: Rec
    ) -> tuple[TotalRecLoss, DespeckleS1] | tuple[None, None]:
        """
        Compute reconstruction loss for 4 views: s1a, s1b, s2a, s2b.
        Each view has 2 reconstruciton losses: direct and cross.
        """

        assert isinstance(batch, BatchMMSits), f"batch is {batch}"

        despeckle_s1a, margin = despeckle(batch.sits1a)
        despeckle_s1b, _ = despeckle(batch.sits1b)

        s1a_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s1a,
                                                 sits=batch.sits1a,
                                                 original_sits=despeckle_s1a,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a,
                                                                         s1b=despeckle_s1b),
                                                 margin=margin,
                                                 mod="s1")

        s1b_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s1b,
                                                 sits=batch.sits1b,
                                                 original_sits=despeckle_s1b,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a,
                                                                         s1b=despeckle_s1b),
                                                 margin=margin,
                                                 mod="s1")

        s2a_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s2a,
                                                 sits=batch.sits2a,
                                                 original_sits=batch.sits2a.sits,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a,
                                                                         s1b=despeckle_s1b))

        s2b_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s2b,
                                                 sits=batch.sits2b,
                                                 original_sits=batch.sits2b.sits,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a,
                                                                         s1b=despeckle_s1b))

        return TotalRecLoss(
            s1_a=s1a_rec_loss,
            s1_b=s1b_rec_loss,
            s2_a=s2a_rec_loss,
            s2_b=s2b_rec_loss,
        ), DespeckleS1(s1a=despeckle_s1a, s1b=despeckle_s1b)


class InvarianceLoss:
    """
    Class for invariance loss (latent space loss).
    """

    def __init__(
            self,
            inv_loss: nn.Module = nn.MSELoss(),
            same_mod_loss: bool = False,
            margin: int = 0
    ):
        self.inv_loss = inv_loss
        self.same_mod_loss = same_mod_loss
        # self.margin = margin
        if margin > 0:
            self.margin_mask = torch.zeros(2, 64, 64)
            self.margin_mask[:, margin: - margin, margin: - margin] = 1
        else:
            self.margin_mask = torch.ones(2, 64, 64)
        self.margin_mask = rearrange(
            self.margin_mask, "b h w -> (b h w)"
        ).bool()

    def compute_inv_loss(self, embeddings: LatRepr) -> torch.Tensor:
        """
        Compute invariance loss.
        It is not computed between 2 views of the same sensor,
        but between 2 views of different sensors.
        """

        creca = self.inv_loss(
            embeddings.s1a[self.margin_mask, :, :],
            embeddings.s2a[self.margin_mask, :, :]
        )
        crecb = self.inv_loss(
            embeddings.s1b[self.margin_mask, :, :],
            embeddings.s2b[self.margin_mask, :, :]
        )
        if self.same_mod_loss:
            crec1 = self.inv_loss(
                embeddings.s1a[self.margin_mask, :, :],
                embeddings.s1b[self.margin_mask, :, :]
            )
            crec2 = self.inv_loss(
                embeddings.s2a[self.margin_mask, :, :],
                embeddings.s2b[self.margin_mask, :, :]
            )
            return (crecb + creca + crec1 + crec2) / 4
        return (crecb + creca) / 2


class GlobalLoss:
    """
    Class that assembles all different losses
    and computes global loss as well.
    """

    def __init__(
            self,
            w_inv: float = 1,
            w_rec: float = 1,
            w_cross_rec: float = 1
    ):

        self.w_rec = w_rec
        self.w_inv = w_inv
        self.w_cross_rec = w_cross_rec

    @staticmethod
    def define_global_loss(tot_rec_loss: TotalRecLoss,
                           inv_loss: torch.Tensor
                           ) -> None | GlobalInvRecMMLoss:
        """
        Define global loss.
        """
        if tot_rec_loss is None:
            global_loss = None
        else:
            global_loss = GlobalInvRecMMLoss(
                total_rec_loss=tot_rec_loss,
                inv_loss=inv_loss,
            )
        return global_loss

    def compute_global_loss(self,
                            global_loss: GlobalInvRecMMLoss
                            ) -> torch.Tensor:
        """
        Compute global loss as weighted sum of
        reconstruction loss and invariance loss.
        """
        if global_loss.inv_loss is not None:
            loss = self.w_inv * global_loss.inv_loss
        else:
            loss = 0
        loss += self.w_rec * self.compute_weighted_rec_loss(global_loss.total_rec_loss)
        return loss

    def compute_weighted_rec_loss(self, total_rec_loss: TotalRecLoss) -> torch.Tensor:
        """
        Compute total weighted reconstruction loss
        (direct and cross reconstruction of 4 views).
        """
        loss = self.compute_weighted_one_rec_loss(total_rec_loss.s2_a)
        loss += self.compute_weighted_one_rec_loss(total_rec_loss.s2_b)
        loss += self.compute_weighted_one_rec_loss(total_rec_loss.s1_a)
        loss += self.compute_weighted_one_rec_loss(total_rec_loss.s1_b)
        return loss / 8

    def compute_weighted_one_rec_loss(
            self,
            one_rec_loss: OneViewRecL
    ) -> torch.Tensor:
        """
        Compute weighted reconstruction loss for one view.
        """
        return (1 -
                self.w_cross_rec) * one_rec_loss.monom_rec + \
               self.w_cross_rec * one_rec_loss.crossm_rec

    def all_losses_dict(self, global_loss: GlobalInvRecMMLoss) -> dict[str, torch.Tensor]:
        """
        Transforms all losses to dict for further logging.
        """
        if global_loss.inv_loss is not None:
            inv_loss = global_loss.inv_loss
        else:
            inv_loss = 0
        dictionary = {"invloss": inv_loss}
        dictionary.update(global_loss.total_rec_loss.to_dict())
        total = self.compute_global_loss(global_loss)
        dictionary.update({"total_loss": total})
        return dictionary
