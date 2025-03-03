"""
Functions to collate the elements of the batch
"""

from collections.abc import Iterable

import torch

from mmmv_ssl.data.dataclass import (
    MMSITS,
    BatchMMSits,
    BatchOneMod,
    SITSOneMod,
)


def collate_fn_mm_dataset(batch: Iterable[MMSITS]) -> BatchMMSits:
    """
    Collate elements for each mod view.
    Also remove unnecessary padding
    (as we padd to len=60 by default at the beginning)
    """
    # Compute get length of the longest view
    max_len_batch = max(*[(~b.sits1a.padd_mask).sum() for b in batch],
                        *[(~b.sits1b.padd_mask).sum() for b in batch],
                        *[(~b.sits2a.padd_mask).sum() for b in batch],
                        *[(~b.sits2b.padd_mask).sum() for b in batch])
    # If we have DEM in modalities, add one more padding,
    # as DEM will be added at the end of the sits later
    if batch[0].dem is not None:
        max_len_batch += 1
    sits1a = collate_one_mode([b.sits1a.remove_padded(max_len_batch) for b in batch])
    sits1b = collate_one_mode([b.sits1b.remove_padded(max_len_batch) for b in batch])
    sits2a = collate_one_mode([b.sits2a.remove_padded(max_len_batch) for b in batch])
    sits2b = collate_one_mode([b.sits2b.remove_padded(max_len_batch) for b in batch])
    dem = torch.stack([b.dem for b in batch]).unsqueeze(1) if batch[0].dem is not None else None
    return BatchMMSits(sits1a=sits1a,
                       sits1b=sits1b,
                       sits2a=sits2a,
                       sits2b=sits2b,
                       dem=dem)


def collate_one_mode(batch: Iterable[SITSOneMod]) -> BatchOneMod:
    """
    Collate all the elements of one mode:
    SITS, DOY, mask, meteo, padding
    """
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
    if batch[0].meteo is not None:
        meteo = torch.stack([b.meteo for b in batch])
    else:
        meteo = None
    return BatchOneMod(
        sits=b_sits,
        input_doy=b_doy,
        true_doy=b_true_doy,
        mask=b_mask,
        padd_index=b_padd_mask,
        meteo=meteo
    )
