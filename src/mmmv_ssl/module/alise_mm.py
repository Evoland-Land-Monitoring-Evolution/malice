import logging
from pathlib import Path

import torch
from einops import rearrange, repeat
from omegaconf import DictConfig

from mmmv_ssl.data.dataclass import BatchMMSits, BatchOneMod
from mmmv_ssl.model.malice_module import AliseMMModule
from mmmv_ssl.module.dataclass import (
    OutMMAliseF,
    OutMMAliseSharedStep,
)
from mmmv_ssl.module.loss import ReconstructionLoss, InvarianceLoss, GlobalLoss
from mmmv_ssl.module.template import TemplateModule

my_logger = logging.getLogger(__name__)



class AliseMM(TemplateModule):
    def __init__(
            self,
            model: AliseMMModule,
            weights,
            batch_size: int,
            lr: float,
    ):
        super().__init__(model, batch_size, lr)

        self.inv_loss = InvarianceLoss(torch.nn.MSELoss())
        self.rec_loss = ReconstructionLoss(torch.nn.MSELoss())
        self.global_loss = GlobalLoss(weights.w_inv, weights.w_rec, weights.w_crossrec)

        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

    def forward(self, batch: BatchMMSits) -> OutMMAliseF:
        return self.model.forward(batch)


    def shared_step(self, batch: BatchMMSits) -> OutMMAliseSharedStep:
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

    def training_step(self, batch: BatchMMSits, batch_idx: int):
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

    def validation_step(self, batch: BatchMMSits, batch_idx: int):
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


    def load_weights(self, path_ckpt, strict=True):
        my_logger.info(f"We load state dict  from {path_ckpt}")
        if not torch.cuda.is_available():
            map_params = {"map_location": "cpu"}
        else:
            map_params = {}
        ckpt = torch.load(path_ckpt, **map_params)
        self.load_state_dict(ckpt["state_dict"], strict=strict)


    def compute_query(self, batch_sits: BatchOneMod, sat: str = "s2"):
        """Compute query and reshape it"""
        query = repeat(
            self.query_builder(self.q_decod_s2 if sat == "s1" else self.q_decod_s1, batch_sits.input_doy),
            "b t c -> b t c h w",
            h=batch_sits.h,
            w=batch_sits.w,
        )
        return rearrange(
            query,
            "b t (nh c )h w -> nh (b h w) t c",
            nh=self.meta_decodeur.num_heads,
        )


def load_malice(pl_module: AliseMM, path_ckpt: str | Path):
    if path_ckpt is not None:
        pl_module = pl_module.load_from_checkpoint(path_ckpt)

    return pl_module


def check_for_nans(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("NaNs or Infs detected in data")
