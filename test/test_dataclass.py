import torch
from einops import repeat

from mmmv_ssl.data.dataclass import BatchOneMod, SITSOneMod, merge2views
from mmmv_ssl.data.dataset.pretraining_mm_mask import split_one_mod


def test_merge2views():
    T = 20
    input_val = torch.arange(T)
    sits = repeat(input_val, "t -> t c h w", c=10, h=16, w=16)
    sits_one_mod = SITSOneMod(
        sits=sits, input_doy=input_val, true_doy=input_val, mask=sits < 10
    )
    sits1, sits2 = split_one_mod(sits_one_mod, max_len=10)
    b1 = BatchOneMod(
        sits=sits1.sits[None, ...],
        input_doy=sits1.input_doy[None, ...],
        true_doy=sits1.true_doy[None, ...],
        padd_index=torch.zeros(int(T // 2)).bool()[None, ...],
        mask=sits1.mask[None, ...],
    )
    b2 = BatchOneMod(
        sits=sits2.sits[None, ...],
        input_doy=sits2.input_doy[None, ...],
        true_doy=sits2.true_doy[None, ...],
        padd_index=torch.zeros(int(T // 2)).bool()[None, ...],
        mask=sits2.mask[None, ...],
    )
    s = merge2views(b1, b2)
    assert s.sits.shape == (2, 10, 10, 16, 16)
    assert s.input_doy.shape == (2, 10)
    assert s.mask.shape == (2, 10, 10, 16, 16)
    assert s.true_doy.shape == (2, 10)
