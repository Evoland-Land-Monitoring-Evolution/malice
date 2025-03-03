# pylint: disable=invalid-name

"""
Template Lightning Module
"""

import logging

import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from mmmv_ssl.model.malice_module import AliseMMModule
from mmmv_ssl.model.malice_module_aux import AliseMMModuleAux

my_logger = logging.getLogger(__name__)


class TemplateModule(pl.LightningModule):
    """
    Base model
    """

    def __init__(self, model: AliseMMModule, lr: float = 0.001):

        super().__init__()

        self.df_metrics = None
        self.border = None
        self.stats = None

        self.lr = lr
        self.bs = None
        self.metric_name = []

        if isinstance(model, DictConfig):
            self.model: AliseMMModule | AliseMMModuleAux = instantiate(
                model,
                _recursive_=False
            )
        else:
            self.model = model

        self.save_hyperparameters()

    def setup(self, stage: str | None):
        self.border = 0

    def configure_optimizers(self):
        """
        Training optimizer.
        """
        # optimizer = instantiate(
        #     self.optimizer, params=self.parameters(), lr=self.learning_rate
        # )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.lr  # , weight_decay=0.01
        )

        # scheduler = instantiate(self.scheduler, optimizer=optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2,
            T_mult=2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "strict": False,
            },
        }

    def on_test_epoch_start(self):
        """On test start"""
        self.df_metrics = pd.DataFrame(
            columns=self.metric_name + ["test_loss"]
        )

    def on_fit_start(self) -> None:
        """
        On fit start, get the stats, and set them into the model.
        Also get batch_size for the logger
        """
        self.stats = (self.trainer.datamodule.all_transform.s2.stats,
                      self.trainer.datamodule.all_transform.s1_asc.stats
                      )
        self.bs = self.trainer.datamodule.batch_size
