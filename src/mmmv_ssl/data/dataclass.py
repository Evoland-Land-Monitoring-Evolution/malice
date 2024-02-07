from dataclasses import dataclass

import torch
from torch import Tensor


class BatchOneMod:
    def __init__(self, sits: Tensor, doy: Tensor, padd_mask: Tensor = None):
        assert len(sits.shape) == 5, f"Incorrect sits shape {sits.shape}"
        self.sits = sits
        assert len(doy.shape) == 2, f"Incorrect doy shape {doy.shape}"
        self.doy = doy
        if padd_mask is not None:
            assert (
                len(padd_mask.shape) == 2
            ), f"Incorrect padd_doy {padd_mask.shape}"
        self.padd_mask = padd_mask

    def pin_memory(self):
        self.sits = self.sits.pin_memory()
        self.doy = self.doy.pin_memory()
        if self.padd_mask is not None:
            self.padd_mask = self.padd_mask.pin_memory()
        return self

    def to(self, device: torch.device | None, dtype: torch.dtype | None):
        self.sits = self.sits.to(device=device, dtype=dtype)
        self.doy = self.doy.to(device, dtype=dtype)
        if self.padd_mask is not None:
            self.padd_mask = self.padd_mask.to(device=device, dtype=dtype)
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
