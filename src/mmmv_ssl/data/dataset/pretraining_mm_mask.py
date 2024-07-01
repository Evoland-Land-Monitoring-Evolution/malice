from dataclasses import dataclass
from typing import Any, Literal

import torch
from einops import rearrange, repeat
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


class PretrainingMMMaskDataset(UnlabeledDataset):

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
        mask_s1 = out_item.s1_asc.mask.mask_nan
        mask_s2 = out_item.s2.mask.merge_mask()

        s2 = SITSOneMod(
            sits=rearrange(out_item.s2.sits, "c t h w -> t c h w"),
            input_doy=out_item.s2.doy,
            true_doy=out_item.s2.true_doy,
            mask=rearrange(mask_s2, "c t h w -> t c h w"),
        )
        sits_s1 = rearrange(out_item.s1_asc.sits,
                            "c t h w-> t c h w")[:, [0, 1], ...]
        ratio = sits_s1[:, [0], ...] / sits_s1[:, [1], ...]
        sits_s1 = torch.log(torch.cat([sits_s1[:,[0,1],...], ratio], dim=1)) #first two bands vv,vh then incidence angle
        s1 = SITSOneMod(
            sits=sits_s1,
            input_doy=out_item.s1_asc.doy,
            true_doy=out_item.s1_asc.true_doy,
            mask=rearrange(mask_s1, "c t h w -> t c h w"),
        )
        s2a, s2b = split_one_mod(s2, max_len=self.max_len)
        s1a, s1b = split_one_mod(s1, max_len=self.max_len)
        final_item = MMSITS(sits1a=s1a, sits1b=s1b, sits2a=s2a, sits2b=s2b)
        return final_item


def split_one_mod(one_mode: SITSOneMod,
                  max_len) -> tuple[SITSOneMod, SITSOneMod]:
    """
    Randomly split SITS into two non overlapping SITS.
    Args:
        one_mode ():

    Returns:

    """
    T = one_mode.sits.shape[0]
    idx = torch.randperm(T)  # idx all elements
    split_idx = int(T // 2)
    idx_v1, _ = torch.sort(idx[:split_idx])
    idx_v2, _ = torch.sort(idx[split_idx:])
    one_mod_v1 = SITSOneMod(
        sits=one_mode.sits[idx_v1, ...],
        mask=one_mode.mask[idx_v1, ...],
        true_doy=None,
        input_doy=one_mode.input_doy[idx_v1, ...],
    )
    one_mod_v2 = SITSOneMod(
        sits=one_mode.sits[idx_v2, ...],
        mask=one_mode.mask[idx_v2, ...],
        true_doy=None,
        input_doy=one_mode.input_doy[idx_v2],
    )
    return one_mod_v1.apply_padding(max_len=max_len), one_mod_v2.apply_padding(
        max_len=max_len)
