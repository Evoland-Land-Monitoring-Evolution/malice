import logging

import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from mt_ssl.data.transform import apply_transform_basic
from omegaconf import DictConfig
from openeo_mmdc.dataset.to_tensor import load_all_transforms
from torch.utils.data import DataLoader

from mmmv_ssl.data.dataclass import BatchMMSits
from mmmv_ssl.data.dataset.collate_fn import collate_fn_mm_dataset
from mmmv_ssl.data.dataset.pretraining_mm_mask import PretrainingMMMaskDataset
from mmmv_ssl.data.dataset.pretraining_mm_year_dataset import (
    ConfigPretrainingMMYearDataset,
)

my_logger = logging.getLogger(__name__)


class MMMaskDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config_dataset,
        batch_size: int = 2,
        num_workers: int = 2,
        path_dir_csv: None = None,
    ):
        super().__init__()
        self.train_dataset: ConfigPretrainingMMYearDataset | DictConfig = config_dataset.train
        self.val_dataset = config_dataset.val
        self.test_dataset = config_dataset.test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.path_dir_csv = path_dir_csv
        self.all_transform = load_all_transforms(self.path_dir_csv,
                                                 modalities=["s1_asc", "s2"])

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.data_train: PretrainingMMMaskDataset = instantiate(
                self.train_dataset,
                dataset_type="train",
            )
            self.data_val: PretrainingMMMaskDataset = instantiate(
                self.val_dataset,
                dataset_type="val",
            )
        else:
            self.data_test: PretrainingMMMaskDataset = instantiate(
                self.test_dataset,
                dataset_type="test",
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn_mm_dataset,
            drop_last=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn_mm_dataset,
            drop_last=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn_mm_dataset,
            drop_last=True
        )

    def on_after_batch_transfer(self, batch: BatchMMSits,
                                dataloader_idx: int) -> BatchMMSits:
        #print("in datamodule", batch.sits2a.sits[0, 0, 0, ...])
        sits1a = apply_transform_basic(
            batch.sits1a.sits, transform=self.all_transform.s1_asc.transform)
        sits1b = apply_transform_basic(
            batch.sits1b.sits, transform=self.all_transform.s1_asc.transform)
        sits2a = apply_transform_basic(
            batch.sits2a.sits, transform=self.all_transform.s2.transform)
        #print("in datamodule", batch.sits2a.sits[0, 0, 0, ...])
        sits2b = apply_transform_basic(
            batch.sits2b.sits, transform=self.all_transform.s2.transform)

        batch.sits2a.sits = sits2a
        batch.sits2b.sits = sits2b
        batch.sits1a.sits = sits1a
        batch.sits1b.sits = sits1b

        #print("in datamodule", batch.sits2b.sits[0, 0, 0, ...])

        return batch

    def transfer_batch_to_device(self, batch: BatchMMSits, device,
                                 dataloader_idx: int):

        return batch.to(device=device, dtype=None)
