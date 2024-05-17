import torch
from einops import repeat

from mmmv_ssl.data.dataclass import SITSOneMod
from mmmv_ssl.data.dataset.pretraining_mm_mask import split_one_mod


def test_mask_one_mod():
    T = 20
    input_val = torch.arange(T)
    mask = torch.randn(T) < 0.5
    sits = repeat(input_val, "t -> t c h w", c=10, h=16, w=16)
    sits_one_mod = SITSOneMod(
        sits=sits, input_doy=input_val, true_doy=input_val, mask=mask[:, None]
    )
    sits1, sits2 = split_one_mod(sits_one_mod, max_len=10)
    assert sits1.sits.shape == (10, 10, 16, 16)
    assert torch.equal(sits1.sits[:, 0, 0, 0], sits1.input_doy)
    assert torch.equal(sits1.sits[:, 0, 0, 0], sits1.true_doy)
    assert sits2.sits.shape == (10, 10, 16, 16)
    assert torch.equal(sits2.sits[:, 0, 0, 0], sits2.input_doy)
    assert torch.equal(sits2.sits[:, 0, 0, 0], sits2.true_doy)
