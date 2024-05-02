from dataclasses import dataclass
from typing import Any, Literal

from mt_ssl.data.dataset.unlabeled_dataset import UnlabeledDataset
from openeo_mmdc.dataset.dataclass import ItemTensorMMDC

from mmmv_ssl.data.dataclass import MMSITS, SITSOneMod


@dataclass
class ConfigPretrainingMMYearDataset:
    directory: str
    max_len: int
    path_dataset_info: str
    crop_size: int
    crop_type: Literal["Center", "Random"] = "Center"
    transform: Any = None
    dataset_type: Literal["train", "val", "test"] = "test"


class PretrainingMMYearDataset(UnlabeledDataset):
    def __len__(self):
        return len(self.c_mmdc_df.s2)

    def __getitem__(self, item) -> MMSITS:
        """
        Getitem for multitask pre-training
        Args:
            item ():

        Returns:
        interpolItem.sits_in.sits shae is f,t,c,h,w
        """
        out_item: ItemTensorMMDC = self.mmdc_sits(item, opt="s1s2")  # t,c,h,w
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
