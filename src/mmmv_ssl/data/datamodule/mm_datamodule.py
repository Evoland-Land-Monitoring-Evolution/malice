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

from mmmv_ssl.constant.dataset import S2_BAND
from mmmv_ssl.data.dataclass import BatchVicReg
from mmmv_ssl.data.dataset.collate_fn import collate_fn_mm_dataset
from mmmv_ssl.data.dataset.pretraining_mm_year_dataset import (
    ConfigPretrainingMMYearDataset,
    PretrainingMMYearDataset,
)

my_logger = logging.getLogger(__name__)


class TemplateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: DictConfig | ConfigPretrainingMMYearDataset,
        val_dataset: DictConfig | ConfigPretrainingMMYearDataset,
        test_dataset: DictConfig | ConfigPretrainingMMYearDataset,
        s2_band: list | None = None,
        batch_size: int = 2,
        crop_size: int = 64,
        num_workers: int = 2,
        path_dir_csv: None = None,
        prefetch_factor: int = 2,
        max_len=60,
    ):
        super().__init__()
        self.max_len = max_len
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        self.crop_size = crop_size
        if s2_band is None:
            s2_band = S2_BAND
        self.s2_band = s2_band
        self.batch_size = batch_size
        self.path_dir_csv = path_dir_csv
        self.prefetch_factor = prefetch_factor
        self.all_transform = load_all_transforms(
            self.path_dir_csv, modalities=["s1_asc", "s2"]
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.data_train: PretrainingMMYearDataset = instantiate(
                self.train_dataset,
                max_len=self.max_len,
                s2_band=self.s2_band,
                crop_size=self.crop_size,
                dataset_type="train",
                transform=None,
            )
            self.data_val: PretrainingMMYearDataset = instantiate(
                self.val_dataset,
                max_len=self.max_len,
                s2_band=self.s2_band,
                crop_size=self.crop_size,
                dataset_type="val",
                transform=None,
            )
        else:
            self.data_test: PretrainingMMYearDataset = instantiate(
                self.test_dataset,
                max_len=self.max_len,
                s2_band=self.s2_band,
                crop_size=self.crop_size,
                dataset_type="test",
                transform=None,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn_mm_dataset,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn_mm_dataset,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn_mm_dataset,
        )

    def on_after_batch_transfer(
        self, batch: BatchVicReg, dataloader_idx: int
    ) -> BatchVicReg:
        "apply transform on device"
        batch.sits1.sits = apply_transform_basic(
            batch.sits1.sits, transform=self.all_transform.s1_asc.transform
        )
        batch.sits2.sits = apply_transform_basic(
            batch.sits1.sits, transform=self.all_transform.s2.transform
        )
        return batch

    def transfer_batch_to_device(
        self, batch: BatchVicReg, device, dataloader_idx: int
    ):
        if dataloader_idx == 0:
            # skip device transfer for the first dataloader or anything you wish
            return batch
        return batch.to(device=device, dtype=None)
