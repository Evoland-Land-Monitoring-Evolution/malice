import torch

from mmmv_ssl.model.decodeur import MetaDecoder
from mmmv_ssl.model.transformer import TransformerBlock, TransformerBlockConfig


def test_forward():
    (
        nh,
        c,
        b,
    ) = (
        4,
        8,
        2,
    )
    decodeur = MetaDecoder(
        num_heads=4,
        input_channels=8,
        d_k=4,
        intermediate_layers=None,
        d_q_in=nh * c,
    )
    inp = torch.rand(2, 10, 8)  # b,t,c
    padd = torch.ones(2, 10)
    queries = torch.rand(nh, b, 16, c)  # nh,b,t,c
    out = decodeur(mm_sits=inp, padd_mm=padd, mm_queries=queries)
    assert out.shape == (2, 16, 8)


def test_forward_intermediate_layer():
    (
        nh,
        c,
        b,
    ) = (
        4,
        8,
        2,
    )
    tr_config = TransformerBlockConfig(
        n_layers=1, d_model=c, d_in=32, n_head=4
    )
    layers = TransformerBlock(tr_config)
    # layers = Encoder(n_layers=2, d_model=c, d_in=16, block_name="pff", nhead=2)
    decodeur = MetaDecoder(
        num_heads=4,
        input_channels=8,
        d_k=4,
        intermediate_layers=layers,
        d_q_in=nh * c,
    )

    inp = torch.rand(2, 10, 8)  # b,t,c
    padd = torch.ones(2, 16)
    queries = torch.rand(nh, b, 16, c)  # nh,b,t,c
    out = decodeur(mm_sits=inp, padd_mm=padd, mm_queries=queries)
    assert out.shape == (2, 16, 8)
