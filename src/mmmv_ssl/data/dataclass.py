from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from openeo_mmdc.dataset.padding import apply_padding
from torch import Tensor


@dataclass
class SITSOneMod:
    sits: Tensor
    doy: Tensor
    true_doy: Tensor
    padd_mask: Tensor | None = None
    mask: Tensor | None = None

    def apply_padding(self, max_len: int, allow_padd=True):
        sits = rearrange(self.sits, "c t h w -> t c h w")
        t = sits.shape[0]
        sits, doy, padd_index = apply_padding(
            allow_padd, max_len, t, sits, self.doy
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
            doy=doy,
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
        doy: Tensor,
        true_doy: Tensor,
        padd_mask: Tensor = None,
        mask: Tensor | None = None,
    ):
        self.true_doy = true_doy
        assert len(sits.shape) == 5, f"Incorrect sits shape {sits.shape}"
        self.sits = sits
        assert len(doy.shape) == 2, f"Incorrect doy shape {doy.shape}"
        self.doy = doy
        if padd_mask is not None:
            assert (
                len(padd_mask.shape) == 2
            ), f"Incorrect padd_doy {padd_mask.shape}"
        self.padd_mask = padd_mask
        self.mask = mask

    def pin_memory(self):
        self.sits = self.sits.pin_memory()
        self.doy = self.doy.pin_memory()
        self.true_doy = self.true_doy.pin_memory()
        if self.padd_mask is not None:
            self.padd_mask = self.padd_mask.pin_memory()
        if self.mask is not None:
            self.mask = self.mask.pin_memory()
        return self

    def to(self, device: torch.device | None, dtype: torch.dtype | None):
        self.sits = self.sits.to(device=device, dtype=dtype)
        self.doy = self.doy.to(device, dtype=dtype)
        self.true_doy = self.true_doy.to(device, dtype=dtype)
        if self.padd_mask is not None:
            self.padd_mask = self.padd_mask.to(device=device, dtype=dtype)
        if self.mask is not None:
            self.mask = self.mask.to(device=device, dtype=dtype)
        return self


@dataclass
class BatchVicReg:
    sits1: BatchOneMod
    sits2: BatchOneMod

    def pin_memory(self):
        self.sits1 = self.sits1.pin_memory()
        self.sits2 = self.sits2.pin_memory()

    def to(self, device: torch.device | None, dtype: torch.dtype | None):
        self.sits1 = self.sits1.to(device=device, dtype=dtype)
        self.sits2 = self.sits2.to(device=device, dtype=dtype)
        return self
