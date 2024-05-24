from collections.abc import Iterable

import torch

from mmmv_ssl.data.dataclass import (
    MMSITS,
    BatchMMSits,
    BatchOneMod,
    SITSOneMod,
)


def collate_fn_mm_dataset(batch: Iterable[MMSITS]) -> BatchMMSits:
    sits1a = collate_one_mode([b.sits1a for b in batch])
    sits1b = collate_one_mode([b.sits1b for b in batch])
    sits2a = collate_one_mode([b.sits2a for b in batch])
    sits2b = collate_one_mode([b.sits2b for b in batch])
    return BatchMMSits(sits1a=sits1a,
                       sits1b=sits1b,
                       sits2a=sits2a,
                       sits2b=sits2b)


def collate_one_mode(batch: Iterable[SITSOneMod]) -> BatchOneMod:
    b_sits = torch.stack([b.sits for b in batch])
    b_doy = torch.stack([b.input_doy for b in batch])

    b_padd_mask = torch.stack([b.padd_mask for b in batch])
    if batch[0].mask is not None:
        b_mask = torch.stack([b.mask for b in batch])
    else:
        b_mask = None
    if batch[0].true_doy is not None:
        b_true_doy = torch.stack([b.true_doy for b in batch])
    else:
        b_true_doy = None
    return BatchOneMod(
        sits=b_sits,
        input_doy=b_doy,
        true_doy=b_true_doy,
        mask=b_mask,
        padd_index=b_padd_mask,
    )
