from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from openeo_mmdc.dataset.dataclass import (
    MMDC_MAXLEN,
    PT_MMDC_DF,
    ItemTensorMMDC,
)
from openeo_mmdc.dataset.load_tensor import load_mmdc_sits
from torch.utils.data import Dataset

from mmmv_ssl.data.dataclass import MMSITS, SITSOneMod
from mmmv_ssl.data.dataset.utils import randomcropindex


class TemplateDataset(Dataset):
    def __init__(
        self,
        crop_size: int = 64,
        crop_type: Literal["Center", "Random"] = "Center",
        transform: None = None,
        dataset_type: Literal["train", "val", "test"] = "test",
    ):
        self.crop_size = crop_size
        self.dataset_type = dataset_type
        self.transform = transform
        self.crop_type = crop_type

    def get_crop_idx(self, rows, cols) -> tuple[int, int]:
        """return the coordinate, width and height of the window loaded
        by the SITS depending od the value of the attribute
        self.crop_type

        Args:
            rows: int
            cols: int
        """
        if self.crop_type == "Random":
            return randomcropindex(rows, cols, self.crop_size, self.crop_size)

        return int(rows // 2 - self.crop_size // 2), int(
            cols // 2 - self.crop_size // 2
        )


@dataclass
class ConfigPretrainingMMYearDataset:
    directory: str
    max_len: int
    path_dataset_info: str
    crop_size: int
    crop_type: Literal["Center", "Random"] = "Center"
    transform: Any = None
    dataset_type: Literal["train", "val", "test"] = "test"


class PretrainingMMYearDataset(TemplateDataset):
    def __init__(
        self,
        directory: str,
        max_len: int,
        path_dataset_info: str,
        crop_size: int = 64,
        crop_type: Literal["Center", "Random"] = "Center",
        transform: None = None,
        dataset_type: Literal["train", "val", "test"] = "test",
    ):
        super().__init__(
            crop_size=crop_size,
            crop_type=crop_type,
            transform=transform,
            dataset_type=dataset_type,
        )
        self.max_len = max_len
        self.directory = directory
        self.dataset_info: PT_MMDC_DF = pd.read_csv(
            path_dataset_info
        )  # dataframe whihc contains two columns "s1_path","s2_path"

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, item) -> MMSITS:
        seed = item
        if self.dataset_type == "train":
            seed = None
        out_item: ItemTensorMMDC = load_mmdc_sits(
            self.dataset_info,
            item=item,
            crop_size=self.crop_size,
            crop_type=self.crop_type,
            max_len=MMDC_MAXLEN(s2=self.max_len, s1_asc=self.max_len),
            opt="s1s2",
            all_transform=self.transform,
            seed=seed,
        )
        mask_s1 = out_item.s1_asc.mask.merge_mask()
        mask_s2 = out_item.s2.mask.merge_mask()
        s2 = SITSOneMod(
            sits=out_item.s2.sits,
            doy=out_item.s2.doy,
            true_doy=out_item.s2.true_doy,
            mask=mask_s2,
        )
        s1 = SITSOneMod(
            sits=out_item.s1_asc.sits,
            doy=out_item.s1_asc.doy,
            true_doy=out_item.s1_asc.true_doy,
            mask=mask_s1,
        )
        final_item = MMSITS(
            sits1=s1.apply_padding(max_len=self.max_len),
            sits2=s2.apply_padding(max_len=self.max_len),
        )
        return final_item
