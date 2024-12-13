import logging
from pathlib import Path

import torch

from mmmv_ssl.data.dataclass import BatchMMSits, BatchOneMod
from mmmv_ssl.model.malice_module import AliseMMModule
from mmmv_ssl.model.projector import IdentityProj
from mmmv_ssl.module.dataclass import (
    OutMMAliseF,
    OutMMAliseSharedStep,
    WeightClass
)
from mmmv_ssl.module.loss import ReconstructionLoss, InvarianceLoss, GlobalLoss
from mmmv_ssl.module.template import TemplateModule

my_logger = logging.getLogger(__name__)


class AliseMM(TemplateModule):
    """Pytorch Lightning class for MALICE algorithm"""

    def __init__(
            self,
            model: AliseMMModule,
            weights: WeightClass,
            lr: float,
            same_mod_loss: bool = False
    ):
        super().__init__(model, lr)

        self.margin = 3

        self.inv_loss = InvarianceLoss(torch.nn.MSELoss(), same_mod_loss=same_mod_loss, margin=self.margin)
        self.rec_loss = ReconstructionLoss(torch.nn.MSELoss(), margin=self.margin, channels=self.model.input_channels)
        self.global_loss = GlobalLoss(weights.w_inv, weights.w_rec, weights.w_crossrec)

        if weights.w_inv == 0:
            self.model.encoder.projector_emb = IdentityProj()  # TODO do better if possible

        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

        print(self.model)


    def forward(self, batch: BatchMMSits) -> OutMMAliseF:
        """Forward Malice"""
        return self.model.forward(batch)

    def shared_step(self, batch: BatchMMSits) -> OutMMAliseSharedStep:
        """Shared step for train/val/test"""
        out_model = self.forward(batch)
        assert isinstance(batch, BatchMMSits)
        tot_rec_loss, despeckle_s1 = self.rec_loss.compute_rec_loss(
            batch=batch, rec=out_model.rec
        )

        inv_loss = self.inv_loss.compute_inv_loss(out_model.emb)

        global_loss = self.global_loss.define_global_loss(tot_rec_loss, inv_loss)

        return OutMMAliseSharedStep(
            loss=global_loss, out_forward=out_model, despeckle_s1=despeckle_s1
        )

    def training_step(self, batch: BatchMMSits, batch_idx: int) -> None | torch.Tensor:
        """Training step"""
        out_shared_step = self.shared_step(batch)
        if out_shared_step.loss is None:
            return None

        loss_dict = self.global_loss.all_losses_dict(out_shared_step.loss)

        for loss_name, value in loss_dict.items():
            if value is not None:
                self.log(
                    f"train/{loss_name}",
                    value.item(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=self.bs
                )

        return loss_dict["total_loss"]

    def validation_step(self, batch: BatchMMSits, batch_idx: int) -> OutMMAliseSharedStep:
        """Validation step"""
        out_shared_step = self.shared_step(batch)

        loss_dict = self.global_loss.all_losses_dict(out_shared_step.loss)

        for loss_name, value in loss_dict.items():
            self.log(
                f"val/{loss_name}",
                value.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=self.bs
            )
        return out_shared_step

    def load_weights(self, path_ckpt: str | Path, strict: bool =True) -> None:
        """Load weights"""
        my_logger.info(f"We load state dict  from {path_ckpt}")
        if not torch.cuda.is_available():
            map_params = {"map_location": "cpu"}
        else:
            map_params = {}
        ckpt = torch.load(path_ckpt, **map_params)
        self.load_state_dict(ckpt["state_dict"], strict=strict)


def load_malice(pl_module: AliseMM, path_ckpt: str | Path) -> AliseMM:
    """Load malice from checkpoint"""
    if path_ckpt is not None:
        pl_module = pl_module.load_from_checkpoint(path_ckpt)

    return pl_module


def check_for_nans(tensor: torch.Tensor) -> None:
    """Check for nans in tensor"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("NaNs or Infs detected in data")
