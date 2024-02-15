from collections.abc import Iterable

import torch

from mmmv_ssl.data.dataclass import (
    MMSITS,
    BatchOneMod,
    BatchVicReg,
    SITSOneMod,
)


def collate_fn_mm_dataset(batch: Iterable[MMSITS]) -> BatchVicReg:
    sits1 = collate_one_mode([b.sits1 for b in batch])
    sits2 = collate_one_mode([b.sits2 for b in batch])
    return BatchVicReg(sits1=sits1, sits2=sits2)


def collate_one_mode(batch: Iterable[SITSOneMod]) -> BatchOneMod:
    b_sits = torch.stack([b.sits for b in batch])
    b_doy = torch.stack([b.doy for b in batch])
    b_true_doy = torch.stack([b.true_doy for b in batch])
    b_padd_mask = torch.stack([b.padd_mask for b in batch])
    if batch[0].mask is not None:
        b_mask = torch.stack([b.padd_mask for b in batch])
    else:
        b_mask = None
    return BatchOneMod(
        sits=b_sits,
        doy=b_doy,
        true_doy=b_true_doy,
        mask=b_mask,
        padd_mask=b_padd_mask,
    )
