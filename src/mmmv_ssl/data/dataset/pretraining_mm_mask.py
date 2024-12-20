from dataclasses import dataclass
from typing import Any, Literal
import logging
import torch
from einops import rearrange
from mmmv_ssl.data.dataset.load_tensor import create_mmcd_tensor_df, load_mmdc_sits
from torch.utils.data import Dataset
from openeo_mmdc.dataset.dataclass import (
    MMDC_MAXLEN,
)

from mmmv_ssl.data.dataclass import MMSITS, SITSOneMod, ItemTensorMMDC

my_logger = logging.getLogger(__name__)


@dataclass
class ConfigPretrainingMMYearDataset:
    directory: str
    max_len: int
    path_dataset_info: str
    crop_size: int
    crop_type: Literal["Center", "Random"] = "Center"
    transform: Any = None
    dataset_type: Literal["train", "val", "test"] = "test"


class PretrainingMMMaskDataset(Dataset):
    def __init__(
            self,
            directory: str,
            max_len: int,
            s2_tile: list | None = None,
            modalities: list[Literal["s2", "s1_asc", "s1_desc", "dem"]] = None,
            path_dir_csv: str | None = None,
            s2_max_ccp: None | float = None,
            crop_size=64,
            crop_type="Center",
            dataset_type="test",
            extract_true_doy: bool = False,
    ):
        """

        Args:
            directory ():  directory where the .nc images are located
            max_len (): the maximum number of date to select in a SITS
            split_in_two (): if set to True input data time series will be split in two.
        """
        super().__init__()

        self.crop_size = crop_size
        self.dataset_type = dataset_type
        self.crop_type = crop_type

        self.extract_true_doy = extract_true_doy
        self.s2_max_ccp = s2_max_ccp
        self.directory = directory

        self.modalities = modalities
        self.s2_tile = s2_tile

        print(f"Directory {directory}")
        self.c_mmdc_df = create_mmcd_tensor_df(
            path_directory=directory, s2_tile=s2_tile, modalities=modalities
        )
        print(f"DF {len(self.c_mmdc_df)}")

        self.max_len = max_len

        self.path_dir_csv = path_dir_csv

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
        sits_s1 = torch.log(
            torch.cat([sits_s1[:, [0, 1], ...], ratio], dim=1))  # first two bands vv,vh then incidence angle
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

    def mmdc_sits(
            self, item, opt: Literal["all", "s2", "s1", "s1s2", "sentinel", "aux"] = "all"
    ) -> ItemTensorMMDC:
        seed = item
        if self.dataset_type == "train":
            seed = None

        return load_mmdc_sits(
            self.c_mmdc_df,
            item=item,
            crop_size=self.crop_size,
            crop_type=self.crop_type,
            max_len=MMDC_MAXLEN(
                s2=self.max_len
            ),  # TODO imporve when working with multimodal data
            opt=opt,
            seed=seed,
        )


class PretrainingMMMaskDatasetAux(PretrainingMMMaskDataset):
    def __init__(
            self,
            directory: str,
            max_len: int,
            s2_tile: list | None = None,
            modalities: list[Literal["s2", "s1_asc", "s1_desc", "dem"]] = None,
            path_dir_csv: str | None = None,
            s2_max_ccp: None | float = None,
            crop_size: int = 64,
            crop_type: str = "Center",
            dataset_type: str = "test",
            extract_true_doy: bool = False,
            days_meteo: list[int] | None = None
    ):
        """

        Args:
            directory ():  directory where the .nc images are located
            max_len (): the maximum number of date to select in a SITS
            split_in_two (): if set to True input data time series will be split in two.
        """
        super().__init__(directory,
                         max_len,
                         s2_tile,
                         modalities,
                         path_dir_csv,
                         s2_max_ccp,
                         crop_size,
                         crop_type,
                         dataset_type,
                         extract_true_doy)
        self.days_meteo = days_meteo

    def __getitem__(self, item) -> MMSITS:
        """
        Getitem for multitask pre-training
        Args:
            item ():

        Returns:
        interpolItem.sits_in.sits shae is f,t,c,h,w
        """
        out_item: ItemTensorMMDC = self.mmdc_sits(item, opt="aux")  # t,c,h,w
        mask_s1 = out_item.s1_asc.mask.mask_nan
        mask_s2 = out_item.s2.mask.merge_mask()
        meteo_s2 = out_item.s2.meteo[:, :, self.days_meteo] \
            if self.days_meteo is not None else out_item.s2.meteo
        meteo_s1_asc = out_item.s1_asc.meteo[:, :, self.days_meteo] \
            if self.days_meteo is not None else out_item.s1_asc.meteo
        s2 = SITSOneMod(
            sits=rearrange(out_item.s2.sits, "c t h w -> t c h w"),
            input_doy=out_item.s2.doy,
            true_doy=out_item.s2.true_doy,
            mask=rearrange(mask_s2, "c t h w -> t c h w"),
            meteo=rearrange(meteo_s2, "c t d -> t c d"),
        )
        s1 = SITSOneMod(
            sits=rearrange(out_item.s1_asc.sits,
                           "c t h w-> t c h w"),
            input_doy=out_item.s1_asc.doy,
            true_doy=out_item.s1_asc.true_doy,
            mask=rearrange(mask_s1, "c t h w -> t c h w"),
            meteo=rearrange(meteo_s1_asc, "c t d -> t c d"),
        )
        s2a, s2b = split_one_mod(s2, max_len=self.max_len)
        s1a, s1b = split_one_mod(s1, max_len=self.max_len)

        final_item = MMSITS(sits1a=s1a, sits1b=s1b, sits2a=s2a, sits2b=s2b, dem=out_item.dem)
        return final_item


def split_one_mod(one_mode: SITSOneMod,
                  max_len: int) -> tuple[SITSOneMod, SITSOneMod]:
    """
    Randomly split SITS into two non overlapping SITS.
    Args:
        one_mode ():

    Returns:

    """
    T = one_mode.sits.shape[0]
    idx = torch.randperm(T)  # idx all elements

    if T == 1:
        idx_v1 = [0]
        idx_v2 = [0]
    else:
        split_idx = int(T // 2)
        idx_v1, _ = torch.sort(idx[:split_idx])
        idx_v2, _ = torch.sort(idx[split_idx:])
        my_logger.debug(f"T {T} split_idx {split_idx}")
    one_mod_v1 = SITSOneMod(
        sits=one_mode.sits[idx_v1, ...],
        mask=one_mode.mask[idx_v1, ...],
        true_doy=None,
        input_doy=one_mode.input_doy[idx_v1, ...],
        meteo=one_mode.meteo[idx_v1, ...],
    )
    one_mod_v2 = SITSOneMod(
        sits=one_mode.sits[idx_v2, ...],
        mask=one_mode.mask[idx_v2, ...],
        true_doy=None,
        input_doy=one_mode.input_doy[idx_v2],
        meteo=one_mode.meteo[idx_v2, ...]
    )
    return one_mod_v1.apply_padding(max_len=max_len), one_mod_v2.apply_padding(
        max_len=max_len)
