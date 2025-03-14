# pylint: disable=invalid-name
"""Different dataclasses for the input data"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from openeo_mmdc.dataset.dataclass import MaskMod
from openeo_mmdc.dataset.padding import apply_padding
from torch import Tensor

my_logger = logging.getLogger(__name__)


@dataclass
class OneMod:
    """One modality dataclass"""
    sits: Tensor
    doy: Tensor | None
    mask: MaskMod | None = MaskMod()
    true_doy: None | Tensor = None
    dates: None | np.datetime64 = None
    meteo: None | Tensor = None


@dataclass
class ItemTensorMMDC:
    """Batch element dataclass"""
    s2: OneMod | None = None
    s1_asc: OneMod | None = None
    s1_desc: OneMod | None = None
    dem: OneMod | None = None
    agera5: OneMod | None = None


@dataclass
class SITSOneMod:
    """One modality dataclass before collate_fn"""
    sits: Tensor
    input_doy: Tensor
    meteo: Tensor | None = None
    true_doy: Tensor | None = None
    padd_mask: Tensor | None = None
    mask: Tensor | None = None

    def apply_padding(self, max_len: int, allow_padd: bool = True):
        """Apply padding"""
        t = self.sits.shape[0]
        sits, doy, padd_index = apply_padding(allow_padd, max_len, t,
                                              self.sits, self.input_doy)
        my_logger.debug(f"t = {t} paddinx {padd_index}")
        padd_doy = (0, max_len - t)
        if self.true_doy is not None:
            true_doy = F.pad(self.true_doy, padd_doy)
        else:
            true_doy = None
        if self.mask is not None:
            padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
            mask = F.pad(self.mask, padd_tensor)
        else:
            mask = None
        if self.meteo is not None:
            padd_tensor = (0, 0, 0, 0, 0, max_len - t)
            meteo = F.pad(self.meteo, padd_tensor)
        else:
            meteo = None
        return SITSOneMod(
            sits=sits,
            input_doy=doy,
            true_doy=true_doy,
            mask=mask,
            padd_mask=padd_index,
            meteo=meteo
        )

    def remove_padded(self, max_len: int):
        """Remove extra padding"""
        if self.sits.shape[0] >= max_len:
            self.sits = self.sits[:max_len]
            self.input_doy = self.input_doy[:max_len]
            if self.true_doy is not None:
                self.true_doy = self.true_doy[:max_len]
            if self.padd_mask is not None:
                self.padd_mask = self.padd_mask[:max_len]
            if self.mask is not None:
                self.mask = self.mask[:max_len]
            if self.meteo is not None:
                self.meteo = self.meteo[:max_len]
            return self
        return self.apply_padding(self.sits.shape[0] + 1)


@dataclass
class MMSITS:
    """Batch dataclass before collate_fn"""
    sits1a: SITSOneMod
    sits1b: SITSOneMod
    sits2a: SITSOneMod
    sits2b: SITSOneMod
    dem: torch.Tensor | None = None


class BatchOneMod:
    """One modality in batch"""
    def __init__(
            self,
            sits: Tensor,
            input_doy: Tensor,
            true_doy: Tensor | None = None,
            meteo: Tensor | None = None,
            padd_index: Tensor | None = None,
            mask: Tensor | None = None,
    ):
        self.true_doy = true_doy
        assert len(sits.shape) == 5, f"Incorrect sits shape {sits.shape}"
        self.sits = sits
        assert (len(
            input_doy.shape) == 2), f"Incorrect doy shape {input_doy.shape}"
        self.input_doy = input_doy
        assert input_doy.shape[1] == sits.shape[1]
        if padd_index is not None:
            assert (len(padd_index.shape) == 2
                    ), f"Incorrect padd_doy {padd_index.shape}"
        self.padd_index = padd_index
        self.mask = mask
        self.meteo = meteo
        self.b = sits.shape[0]
        self.t = sits.shape[1]
        self.c = sits.shape[2]
        self.h = sits.shape[3]
        self.w = sits.shape[4]

    def pin_memory(self):
        """Enable pin memory"""
        self.sits = self.sits.pin_memory()
        self.input_doy = self.input_doy.pin_memory()
        if self.true_doy is not None:
            self.true_doy = self.true_doy.pin_memory()
        if self.padd_index is not None:
            self.padd_index = self.padd_index.pin_memory()
        if self.mask is not None:
            self.mask = self.mask.pin_memory()
        if self.meteo is not None:
            self.meteo = self.meteo.pin_memory()
        return self

    def to(self, device: torch.device | None = None,
           dtype: torch.dtype | None = None):
        """To device and to dtype"""
        self.sits = self.sits.to(device=device, dtype=dtype)
        self.input_doy = self.input_doy.to(device, dtype=dtype)
        if self.true_doy is not None:
            self.true_doy = self.true_doy.to(device, dtype=dtype)
        if self.padd_index is not None:
            self.padd_index = self.padd_index.to(device=device, dtype=dtype)
        if self.mask is not None:
            self.mask = self.mask.to(device=device, dtype=dtype)
        if self.meteo is not None:
            self.meteo = self.meteo.to(device=device, dtype=dtype)
        return self


@dataclass
class BatchMMSits:
    """Batch dataclass after collate_fn"""
    sits1a: BatchOneMod
    sits1b: BatchOneMod
    sits2a: BatchOneMod
    sits2b: BatchOneMod
    dem: torch.Tensor | None = None

    def pin_memory(self):
        """Enable pin memory"""
        self.sits1a = self.sits1a.pin_memory()
        self.sits2a = self.sits2a.pin_memory()
        self.sits1b = self.sits1b.pin_memory()
        self.sits2b = self.sits2b.pin_memory()
        self.dem = self.dem.pin_memory() if self.dem is not None else None
        return self

    def to(self, device: torch.device | None, dtype: torch.dtype | None):
        """To device"""
        self.sits1a = self.sits1a.to(device=device, dtype=dtype)
        self.sits1b = self.sits1b.to(device=device, dtype=dtype)
        self.sits2a = self.sits2a.to(device=device, dtype=dtype)
        self.sits2b = self.sits2b.to(device=device, dtype=dtype)
        self.dem = self.dem.to(device=device, dtype=dtype) if self.dem is not None else None
        return self


@dataclass
class MMChannels:
    """Input modality channels"""
    s1_channels: int = 3
    s2_channels: int = 10
    s1_aux_channels: int = 1
    s2_aux_channels: int = 1
    s1_meteo_channels: int = 8
    s2_meteo_channels: int = 8
    dem_channels: int | None = None


def merge2views(viewa: BatchOneMod, viewb: BatchOneMod) -> BatchOneMod:
    """Merge 2 views in a batch"""
    sits = torch.cat([viewa.sits, viewb.sits])
    input_doy = torch.cat([viewa.input_doy, viewb.input_doy])
    if viewa.true_doy is not None:
        true_doy = torch.cat([viewa.true_doy, viewb.true_doy])
    else:
        true_doy = None
    if viewa.mask is not None:
        mask = torch.cat([viewa.mask, viewb.mask])
    else:
        mask = None
    padd_mask = torch.cat([viewa.padd_index, viewb.padd_index])
    if viewa.meteo is not None:
        meteo = torch.cat([viewa.meteo, viewb.meteo])
    else:
        meteo = None
    return BatchOneMod(
        sits=sits,
        input_doy=input_doy,
        true_doy=true_doy,
        padd_index=padd_mask,
        mask=mask,
        meteo=meteo
    )
