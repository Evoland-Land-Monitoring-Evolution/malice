import logging

import lightning.pytorch as pl
import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from openeo_mmdc.dataset.dataclass import Stats

# from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mmmv_ssl.data.dataclass import BatchMMSits
from mmmv_ssl.loss.vicreg_loss import (
    CovarianceLoss,
    TotalLoss,
    VarianceLoss,
    VicRegLoss,
)
from mmmv_ssl.model.dataclass import OutUTAEForward
from mmmv_ssl.model.projector import MMProjector
from mmmv_ssl.model.utae import UTAE

my_logger = logging.getLogger(__name__)


class LVicRegModule(pl.LightningModule):
    def __init__(
        self,
        train_config,
        d_emb: int,
        d_model: int,
        stats: None | Stats,
        model1: UTAE,
        model2: UTAE,
        projector_bottleneck: MMProjector,
        projector_local: MMProjector,
    ):  # TODO define a dataclass for train_config
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.model1: UTAE = model1
        self.model2: UTAE = model2
        self.train_config = train_config
        self.learning_rate = train_config.lr
        self.scheduler = train_config.scheduler
        self.optimizer = self.train_config.optimizer
        self.bs = train_config.batch_size
        self.stats = stats
        self.metric_name = []
        self.model1.return_maps = True
        self.model2.return_maps = True
        self.w_inv = train_config.w_inv
        self.w_cov = train_config.w_cov
        self.w_var = train_config.w_var
        self.w_bottle = train_config.w_bottle
        self.w_local = train_config.w_local
        self.projector_bottleneck = projector_bottleneck
        self.projetor_local = projector_local
        self.var_loss = VarianceLoss()
        self.cov_loss = CovarianceLoss(self.d_emb)
        self.inv_loss = nn.MSELoss()
        my_logger.debug(f"UTAE return maps set to {self.model1.return_maps}")

    def forward(
        self, batch: BatchMMSits
    ) -> tuple[OutUTAEForward, OutUTAEForward]:
        repr_1 = self.model1(
            input=batch.sits1.sits,
            batch_positions=batch.sits1.input_doy,
            key_padding_mask=batch.sits1.padd_index,
        )
        repr_2 = self.model2(
            input=batch.sits2.sits,
            batch_positions=batch.sits2.input_doy,
            key_padding_mask=batch.sits2.padd_index,
        )
        return repr_1, repr_2

    def shared_step(
        self, batch: BatchMMSits
    ) -> tuple[TotalLoss, OutUTAEForward, OutUTAEForward]:
        out1, out2 = self.forward(
            batch
        )  # TODO, try using all_gather_ddp_if_available()
        botteleneck_repr_1 = rearrange(
            out1.feature_maps[0], "b c h w -> b 1 c h w"
        )
        botteleneck_repr_2 = rearrange(
            out2.feature_maps[0], "b c h w -> b 1 c h w"
        )
        my_logger.debug(
            f"botteclneck feature maps shape is {botteleneck_repr_1.shape}"
        )
        repr_1 = rearrange(out1.seg_map, "b h w c -> b 1 c h w")
        repr_2 = rearrange(out2.seg_map, "b h w c -> b 1 c h w")
        bottleneck_loss = self.apply_vic_reg_loss(
            botteleneck_repr_1, botteleneck_repr_2, self.projector_bottleneck
        )
        local_loss = self.apply_vic_reg_loss(
            repr_1, repr_2, self.projetor_local
        )
        total_loss = TotalLoss(
            bottleneck_loss=bottleneck_loss,
            local_loss=local_loss,
            w_bottleneck=self.w_bottle,
            w_local=self.w_local,
        )
        return total_loss, out1, out2

    def training_step(self, batch: BatchMMSits, batch_idx: int):
        total_loss, out1, out2 = self.shared_step(batch)
        loss = total_loss.total_loss()
        self.log_dict(
            total_loss.to_dict(suffix="train"),
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: BatchMMSits, batch_idx: int):
        total_loss, out1, out2 = self.shared_step(batch)
        self.log_dict(
            total_loss.to_dict(suffix="val"),
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
        )
        return out1, out2

    def test_step(self, batch: BatchMMSits, batch_idx: int):
        total_loss, out1, out2 = self.shared_step(batch)
        loss = total_loss.total_loss()
        self.log_dict(
            total_loss.to_dict(suffix="train"),
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
        )
        return loss

    def apply_vic_reg_loss(
        self, repr_1: Tensor, repr_2: Tensor, projector: MMProjector
    ) -> VicRegLoss:
        b, t, c, h, w = repr_1.shape
        my_logger.debug(f"t dim {t}")
        emb1, emb2 = projector.forward(
            rearrange(repr_1, "b t c h w -> (b t h w) c"),
            rearrange(repr_2, "b t c h w -> (b t h w) c"),
        )
        emb = torch.stack([emb1, emb2])
        emb = rearrange(
            emb,
            "r (b t h w ) c -> r b t c h w",
            r=2,
            b=b,
            t=t,
            h=h,
            w=w,
        )
        emb = emb - torch.mean(
            emb, dim=1, keepdim=True
        )  # mean over the batch dimension
        var_loss = self.var_loss(emb[0, ...], emb[1, ...])
        cov_loss = self.cov_loss(emb[0, ...], emb[1, ...], batch_size=self.bs)
        inv_loss = self.inv_loss(emb[0, ...], emb[1, ...])
        return VicRegLoss(
            inv_loss=inv_loss,
            var_loss=var_loss,
            cov_loss=cov_loss,
            w_inv=self.w_inv,
            w_var=self.w_var,
            w_cov=self.w_cov,
        )

    def configure_optimizers(self):
        if isinstance(self.optimizer, DictConfig):
            optimizer = instantiate(
                self.optimizer, params=self.parameters(), lr=self.learning_rate
            )
        elif isinstance(self.optimizer, Optimizer):
            optimizer = self.optimizer
        else:
            raise NotImplementedError
        if isinstance(self.scheduler, DictConfig):
            sch = instantiate(self.scheduler, optimizer=optimizer)
        elif isinstance(self.scheduler, LRScheduler):
            sch = self.scheduler
        else:
            raise NotImplementedError
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": self.train_config.optimizer_monitor,
                "strict": False,
            },
        }
