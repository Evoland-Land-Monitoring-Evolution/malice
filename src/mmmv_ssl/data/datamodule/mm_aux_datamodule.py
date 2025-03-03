"""
Malice datamodule
"""

import logging

from hydra.utils import instantiate
from openeo_mmdc.dataset.to_tensor import load_all_transforms

from mmmv_ssl.data.dataclass import BatchMMSits
from mmmv_ssl.data.datamodule.mm_datamodule import MMMaskDataModule
from mmmv_ssl.data.datamodule.utils import apply_transform_basic
from mmmv_ssl.data.dataset.pretraining_mm_mask import PretrainingMMMaskDatasetAux

my_logger = logging.getLogger(__name__)


class MMMaskDataModuleAux(MMMaskDataModule):
    """Malice aux datamodule"""

    def __init__(
            self,
            config_dataset,
            batch_size: int = 2,
            num_workers: int = 2,
            path_dir_csv: None = None,
    ):
        super().__init__(config_dataset, batch_size, num_workers, path_dir_csv)

        self.all_transform = load_all_transforms(self.path_dir_csv,
                                                 modalities=["s1_asc", "s2", "agera5", "dem"])

    def setup(self, stage: str) -> None:
        """Setup"""
        if stage == "fit":
            self.data_train: PretrainingMMMaskDatasetAux = instantiate(
                self.train_dataset,
                dataset_type="train",
            )
            self.data_val: PretrainingMMMaskDatasetAux = instantiate(
                self.val_dataset,
                dataset_type="val",
            )
        else:
            self.data_test: PretrainingMMMaskDatasetAux = instantiate(
                self.test_dataset,
                dataset_type="test",
            )

    def on_after_batch_transfer(self, batch: BatchMMSits,
                                dataloader_idx: int) -> BatchMMSits:
        """
        Normalize on batch transfer.
        """
        # print("in datamodule", batch.sits2a.sits[0, 0, 0, ...])
        sits1a = apply_transform_basic(
            batch.sits1a.sits, transform=self.all_transform.s1_asc.transform)
        sits1b = apply_transform_basic(
            batch.sits1b.sits, transform=self.all_transform.s1_asc.transform)
        sits2a = apply_transform_basic(
            batch.sits2a.sits, transform=self.all_transform.s2.transform)
        # print("in datamodule", batch.sits2a.sits[0, 0, 0, ...])

        sits2b = apply_transform_basic(
            batch.sits2b.sits, transform=self.all_transform.s2.transform)
        sits1a_meteo = apply_transform_basic(
            batch.sits1a.meteo[..., None],
            transform=self.all_transform.agera5.transform).squeeze(-1)
        sits1b_meteo = apply_transform_basic(
            batch.sits1b.meteo[..., None],
            transform=self.all_transform.agera5.transform).squeeze(-1)

        sits2a_meteo = apply_transform_basic(
            batch.sits2a.meteo[..., None],
            transform=self.all_transform.agera5.transform).squeeze(-1)
        sits2b_meteo = apply_transform_basic(
            batch.sits2b.meteo[..., None],
            transform=self.all_transform.agera5.transform).squeeze(-1)
        dem = apply_transform_basic(batch.dem,
                                    transform=self.all_transform.dem.transform)

        batch.sits2a.sits = sits2a
        batch.sits2b.sits = sits2b
        batch.sits1a.sits = sits1a
        batch.sits1b.sits = sits1b

        batch.sits2a.meteo = sits2a_meteo
        batch.sits2b.meteo = sits2b_meteo
        batch.sits1a.meteo = sits1a_meteo
        batch.sits1b.meteo = sits1b_meteo

        batch.sits1b.dem = dem

        return batch
