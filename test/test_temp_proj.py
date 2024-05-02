import torch

from mmmv_ssl.model.temp_proj import TemporalProjector


def test_forward():
    nh, nq = 2, 10
    b, t1, t2, c, h, w = 1, 5, 7, 10, 8, 8
    temp_proj = TemporalProjector(num_heads=nh, input_channels=c, n_q=nq)
    s1 = torch.rand(b * h * w, t1, c)
    padd_s1 = torch.ones(b * h * w, t1).bool()
    padd_s2 = torch.ones(b * h * w, t2).bool()
    s2 = torch.rand(b * h * w, t2, c)
    out = temp_proj(s1, padd_s1, s2, padd_s2)
    assert out.s1.shape == (b * h * w, nq, c)
    assert out.s2.shape == (b * h * w, nq, c)
