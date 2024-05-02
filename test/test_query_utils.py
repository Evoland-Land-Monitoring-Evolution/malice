import torch
from einops import repeat

from mmmv_ssl.model.encoding import PositionalEncoder
from mmmv_ssl.model.query_utils import TempMetaQuery


def test_forward():
    positional_encoding = PositionalEncoder(d=16)
    meta_q = TempMetaQuery(pe_config=positional_encoding, input_channels=32)
    out_q = meta_q(
        q=torch.randn(16), doy=repeat(torch.arange(0, 12), "t-> b t ", b=2)
    )
    assert out_q.shape == (2, 12, 32)
