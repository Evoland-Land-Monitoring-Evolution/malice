from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from openeo_mmdc.dataset.padding import apply_padding
from torch import Tensor


@dataclass
class SITSOneMod:
    sits: Tensor
    input_doy: Tensor
    true_doy: Tensor
    padd_mask: Tensor | None = None
    mask: Tensor | None = None

    def apply_padding(self, max_len: int, allow_padd=True):
        sits = rearrange(self.sits, "c t h w -> t c h w")
        t = sits.shape[0]
        sits, doy, padd_index = apply_padding(
            allow_padd, max_len, t, sits, self.input_doy
        )
        padd_doy = (0, max_len - t)
        true_doy = F.pad(self.true_doy, padd_doy)
        if self.padd_mask is not None:
            padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
            mask = F.pad(self.padd_mask, padd_tensor)
        else:
            mask = None
        return SITSOneMod(
            sits=sits,
            input_doy=doy,
            true_doy=true_doy,
            mask=mask,
            padd_mask=padd_index,
        )


@dataclass
class MMSITS:
    sits1: SITSOneMod
    sits2: SITSOneMod


class BatchOneMod:
    def __init__(
        self,
        sits: Tensor,
        input_doy: Tensor,
        true_doy: Tensor,
        padd_index: Tensor = None,
        mask: Tensor | None = None,
    ):
        self.true_doy = true_doy
        assert len(sits.shape) == 5, f"Incorrect sits shape {sits.shape}"
        self.sits = sits
        assert (
            len(input_doy.shape) == 2
        ), f"Incorrect doy shape {input_doy.shape}"
        self.input_doy = input_doy
        assert input_doy.shape[1] == sits.shape[1]
        if padd_index is not None:
            assert (
                len(padd_index.shape) == 2
            ), f"Incorrect padd_doy {padd_index.shape}"
        self.padd_index = padd_index
        self.mask = mask
        self.b = sits.shape[0]
        self.t = sits.shape[1]
        self.c = sits.shape[2]
        self.h = sits.shape[3]
        self.w = sits.shape[4]

    def pin_memory(self):
        self.sits = self.sits.pin_memory()
        self.input_doy = self.input_doy.pin_memory()
        self.true_doy = self.true_doy.pin_memory()
        if self.padd_index is not None:
            self.padd_index = self.padd_index.pin_memory()
        if self.mask is not None:
            self.mask = self.mask.pin_memory()
        return self

    def to(self, device: torch.device | None, dtype: torch.dtype | None):
        self.sits = self.sits.to(device=device, dtype=dtype)
        self.input_doy = self.input_doy.to(device, dtype=dtype)
        self.true_doy = self.true_doy.to(device, dtype=dtype)
        if self.padd_index is not None:
            self.padd_index = self.padd_index.to(device=device, dtype=dtype)
        if self.mask is not None:
            self.mask = self.mask.to(device=device, dtype=dtype)
        return self


@dataclass
class BatchMMSits:
    sits1: BatchOneMod
    sits2: BatchOneMod

    def pin_memory(self):
        self.sits1 = self.sits1.pin_memory()
        self.sits2 = self.sits2.pin_memory()

    def to(self, device: torch.device | None, dtype: torch.dtype | None):
        self.sits1 = self.sits1.to(device=device, dtype=dtype)
        self.sits2 = self.sits2.to(device=device, dtype=dtype)
        return self


@dataclass
class MMChannels:
    s1_channels: int = 3
    s2_channels: int = 10
