import torch
from einops import repeat
from hydra.utils import instantiate
from mt_ssl.utils.open import open_yaml
from omegaconf import DictConfig

from mmmv_ssl.data.dataclass import BatchMMSits, BatchOneMod, MMChannels
from mmmv_ssl.model.clean_ubarn import CleanUBarn
from mmmv_ssl.model.clean_ubarn_repr_encoder import CleanUBarnReprEncoder
from mmmv_ssl.model.decodeur import MetaDecoder
from mmmv_ssl.model.encoding import PositionalEncoder
from mmmv_ssl.model.temp_proj import TemporalProjector
from mmmv_ssl.model.transformer import TransformerBlock
from mmmv_ssl.model.datatypes import TransformerBlockConfig, CleanUBarnConfig
from mmmv_ssl.module.alise_mm import AliseMM
from mmmv_ssl.module.dataclass import OutMMAliseF


def generate_input_mod(b, t, c, h, w, opt=None):
    if opt == "log":
        sits = (
                torch.rand(b, t, c, h, w)
                + torch.arange(h)[None, None, None, :, None] / 100
                + torch.arange(w)[None, None, None, None, :] / 100
        )
    else:
        sits = torch.rand(b, t, c, h, w)
    padd_index = torch.zeros(b, t)
    padd_index[0, -1] = 1
    padd_index[0, -3] = 1
    padd_index[0, -2] = 1
    return BatchOneMod(
        sits=sits,
        input_doy=repeat(torch.arange(t), "t -> b t", b=b),
        true_doy=repeat(torch.arange(t), "t -> b t", b=b),
        padd_index=padd_index.bool(),
        mask=torch.zeros(b, t, c, h, w).bool(),
    )


def generate_mm_input(b, t1, t2, h, w):
    b_s1_a = generate_input_mod(b, t1, 3, h, w, "log")
    b_s1_b = generate_input_mod(b, t1, 3, h, w, "log")
    b_s2_a = generate_input_mod(b, t2, 10, h, w)
    b_s2_b = generate_input_mod(b, t2, 10, h, w)
    return BatchMMSits(
        sits2a=b_s2_a, sits2b=b_s2_b, sits1a=b_s1_a, sits1b=b_s1_b
    )


# def test_forward2():
#     d_repr = 16
#     nh = 2
#     nq = 10
#     nh_decod = 4
#     query_s1s2 = 32
#     pe_c = 32
#     s1_ub = CleanUBarn(
#         ne_layers=1, d_model=d_repr, input_channels=3, use_transformer=True
#     )
#     s2_ub = CleanUBarn(
#         ne_layers=1, d_model=d_repr, input_channels=10, use_transformer=True
#     )
#     s1_sste = CleanUBarnReprEncoder(
#         ubarn=s1_ub,
#         d_model=d_repr,
#         input_channels=3,
#         use_pytorch_transformer=False,
#     )
#     s2_sste = CleanUBarnReprEncoder(
#         ubarn=s2_ub,
#         d_model=d_repr,
#         input_channels=10,
#         use_pytorch_transformer=False,
#     )
#     common_temp_proj = TemporalProjector(
#         num_heads=nh, input_channels=d_repr, n_q=10
#     )
#     decodeur = MetaDecoder(
#         num_heads=nh_decod,
#         input_channels=d_repr,
#         d_k=4,
#         intermediate_layers=None,
#         d_q_in=pe_c + query_s1s2,
#     )
#     train_config = DictConfig(open_yaml("../config/train/pretrain_ssl.yaml"))
#     mm_channels = MMChannels(s1_channels=3, s2_channels=10)
#     pe_config = PositionalEncoder(d=pe_c)
#     module = AliseMM(
#         encodeur_s1=s1_sste,
#         encodeur_s2=s2_sste,
#         common_temp_proj=common_temp_proj,
#         decodeur=decodeur,
#         train_config=train_config,
#         input_channels=mm_channels,
#         pe_config=pe_config,
#         d_repr=d_repr,
#         query_s1s2_d=query_s1s2,
#         pe_channels=pe_c,
#     )
#     input_batch = generate_mm_input(1, 4, 4, 64, 64)
#     out = module.forward(input_batch)
#     print(out.repr.s1a.shape)
#     assert out.repr.s1a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s2b.shape
#     assert out.repr.s1a.shape == (1, nq, d_repr, 64, 64)
#     assert out.rec.s1a.same_mod.shape == out.rec.s1a.same_mod.shape
#     assert out.rec.s1b.same_mod.shape == (1, 4, 3, 64, 64)
#     assert out.rec.s2a.same_mod.shape == (1, 4, 10, 64, 64)
#     assert out.rec.s2b.same_mod.shape == out.rec.s2b.other_mod.shape


# def test_forward():
#     d_repr = 16
#     nh = 2
#     nq = 10
#     nh_decod = 4
#     query_s1s2 = 32
#     pe_c = 32
#
#
#     s1_ub = CleanUBarnConfig(
#         d_model=d_repr,
#         nhead=nh,
#         ne_layers=1,
#         input_channels=3,
#         use_transformer=True
#     )
#     s2_ub = CleanUBarnConfig(
#         d_model=d_repr,
#         nhead=nh,
#         ne_layers=1,
#         input_channels=10,
#         use_transformer=True
#     )
#
#     s1_sste = CleanUBarnReprEncoder(
#         ubarn_config=s1_ub,
#         d_model=d_repr,
#         input_channels=3,
#         use_pytorch_transformer=True
#     )
#     s2_sste = CleanUBarnReprEncoder(
#         ubarn_config=s2_ub,        d_model=d_repr,
#         input_channels=10,
#         use_pytorch_transformer=True
#     )
#     common_temp_proj = TemporalProjector(
#         num_heads=nh, input_channels=d_repr, n_q=nq
#     )
#     decodeur = MetaDecoder(
#         num_heads=nh_decod,
#         input_channels=d_repr,
#         d_k=nh_decod,
#         intermediate_layers=None,
#         d_q_in=pe_c + query_s1s2,
#     )
#     train_config = DictConfig(open_yaml("../config/train/pretrain_ssl.yaml"))
#     mm_channels = MMChannels(s1_channels=3, s2_channels=10)
#     pe_config = PositionalEncoder(d=pe_c)
#     module = AliseMM(
#         encodeur_s1=s1_sste,
#         encodeur_s2=s2_sste,
#         common_temp_proj=common_temp_proj,
#         decodeur=decodeur,
#         train_config=train_config,
#         input_channels=mm_channels,
#         pe_config=pe_config,
#         d_repr=d_repr,
#         query_s1s2_d=query_s1s2,
#         pe_channels=pe_c,
#     )
#     input_batch = generate_mm_input(1, 4, 4, 64, 64)
#     out = module.forward(input_batch)
#     print(out.repr.s1a.shape)
#     assert out.repr.s1a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s2b.shape
#     assert out.repr.s1a.shape == (1, nq, d_repr, 64, 64)
#     assert out.rec.s1a.same_mod.shape == out.rec.s1a.same_mod.shape
#     assert out.rec.s1b.same_mod.shape == (1, 4, 3, 64, 64)
#     assert out.rec.s2a.same_mod.shape == (1, 4, 10, 64, 64)
#     assert out.rec.s2b.same_mod.shape == out.rec.s2b.other_mod.shape


# def test_forward_with_deep_decoder():
#     d_repr = 16
#     nh = 2
#     nq = 10
#     nh_decod = 4
#     query_s1s2 = 32
#     pe_c = 32
#     s1_ub = CleanUBarn(
#         ne_layers=1, d_model=d_repr, input_channels=3, use_transformer=True
#     )
#     s2_ub = CleanUBarn(
#         ne_layers=1, d_model=d_repr, input_channels=10, use_transformer=True
#     )
#     s1_sste = CleanUBarnReprEncoder(
#         ubarn=s1_ub, d_model=d_repr, input_channels=3
#     )
#     s2_sste = CleanUBarnReprEncoder(
#         ubarn=s2_ub, d_model=d_repr, input_channels=10
#     )
#     common_temp_proj = TemporalProjector(
#         num_heads=nh, input_channels=d_repr, n_q=10
#     )
#     tr_config = TransformerBlockConfig(
#         n_layers=1, d_model=d_repr, d_in=32, n_head=4
#     )
#     layers = TransformerBlock(tr_config)
#     decodeur = MetaDecoder(
#         num_heads=nh_decod,
#         input_channels=d_repr,
#         d_k=4,
#         intermediate_layers=layers,
#         d_q_in=pe_c + query_s1s2,
#     )
#     train_config = DictConfig(open_yaml("../config/train/pretrain_ssl.yaml"))
#     mm_channels = MMChannels(s1_channels=3, s2_channels=10)
#     pe_config = PositionalEncoder(d=pe_c)
#     module = AliseMM(
#         encodeur_s1=s1_sste,
#         encodeur_s2=s2_sste,
#         common_temp_proj=common_temp_proj,
#         decodeur=decodeur,
#         train_config=train_config,
#         input_channels=mm_channels,
#         pe_config=pe_config,
#         d_repr=d_repr,
#         query_s1s2_d=query_s1s2,
#         pe_channels=pe_c,
#     )
#     input_batch = generate_mm_input(1, 8, 2, 64, 64)
#     out = module.forward(input_batch)
#     assert out.repr.s1a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s2b.shape
#     assert out.repr.s1a.shape == (1, nq, d_repr, 64, 64)
#     assert out.rec.s1a.same_mod.shape == out.rec.s1a.same_mod.shape
#     assert out.rec.s1b.same_mod.shape == (1, 2, 3, 64, 64)
#     assert out.rec.s2a.same_mod.shape == (1, 2, 10, 64, 64)
#     assert out.rec.s2b.same_mod.shape == out.rec.s2b.other_mod.shape


def test_instantiate():
    module_config = DictConfig(open_yaml("../config/module/alise_mm_proj.yaml"))
    module = instantiate(
        module_config,
        _recursive_=False,
    )
    nq = module_config["model"]["encoder"]["common_temp_proj"]["n_q"]
    d_repr = module_config["model"]["d_repr"]
    input_batch = generate_mm_input(1, 4, 4, 64, 64)

    out: OutMMAliseF = module.forward(input_batch)
    assert out.repr.s1a.shape == out.repr.s1b.shape
    assert out.repr.s2a.shape == out.repr.s1b.shape
    assert out.repr.s2a.shape == out.repr.s2b.shape
    assert out.repr.s1a.shape == (1, nq, d_repr, 64, 64)
    assert out.rec.s1a.same_mod.shape == out.rec.s1a.same_mod.shape
    assert out.rec.s2a.same_mod.shape == (1, 4, 10, 64, 64)
    assert out.rec.s1b.same_mod.shape == (1, 4, 3, 64, 64)

    assert out.rec.s2b.same_mod.shape == out.rec.s2b.other_mod.shape


# def test_instantiate_deepdeocder():
#     nq = 10
#     d_repr = 8
#     module_config = DictConfig(
#         open_yaml("../config/model/alise_mm_deepdecod.yaml")
#     )
#     train_config = DictConfig(open_yaml("../config/train/pretrain_ssl.yaml"))
#     mm_channels = MMChannels(s1_channels=3, s2_channels=10)
#     module = instantiate(
#         module_config,
#         train_config=train_config,
#         _recursive_=False,
#         input_channels=mm_channels,
#         d_repr=d_repr,
#     )
#     input_batch = generate_mm_input(1, 2, 2, 64, 64)
#     out = module.forward(input_batch)
#     assert out.repr.s1a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s1b.shape
#     assert out.repr.s2a.shape == out.repr.s2b.shape
#     assert out.repr.s1a.shape == (1, nq, d_repr, 64, 64)
#     assert out.rec.s1a.same_mod.shape == out.rec.s1a.same_mod.shape
#     assert out.rec.s1b.same_mod.shape == (1, 2, 3, 64, 64)
#     assert out.rec.s2a.same_mod.shape == (1, 2, 10, 64, 64)
#     assert out.rec.s2b.same_mod.shape == out.rec.s2b.other_mod.shape
#
#
# def test_instantiate_deepdeocder_training_step():
#     d_repr = 8
#     module_config = DictConfig(
#         open_yaml("../config/module/alise_mm_deepdecod.yaml")
#     )
#     train_config = DictConfig(open_yaml("../config/train/pretrain_ssl.yaml"))
#     mm_channels = MMChannels(s1_channels=3, s2_channels=10)
#     module = instantiate(
#         module_config,
#         train_config=train_config,
#         _recursive_=False,
#         input_channels=mm_channels,
#         d_repr=d_repr,
#     )
#     input_batch = generate_mm_input(2, 4, 3, 64, 64)
#     out_module = module.shared_step(input_batch)
#     print(out_module.loss.to_dict())
#
#
# def test_instantiate_deepdeocder_validation_step():
#     d_repr = 8
#     module_config = DictConfig(
#         open_yaml("../config/module/alise_mm_deepdecod.yaml")
#     )
#     train_config = DictConfig(open_yaml("../config/train/pretrain_ssl.yaml"))
#     mm_channels = MMChannels(s1_channels=3, s2_channels=10)
#     module = instantiate(
#         module_config,
#         train_config=train_config,
#         _recursive_=False,
#         input_channels=mm_channels,
#         d_repr=d_repr,
#     )
#     input_batch = generate_mm_input(1, 2, 2, 64, 64)
#     out_module = module.validation_step(input_batch, 0)
#     print(out_module.loss.to_dict(suffix="val"))
#     # print(out_module.loss.to_dict())


def test_instantiate_proj_training_step():
    module_config = DictConfig(
        open_yaml("../config/module/alise_mm_proj.yaml")
    )
    module = instantiate(
        module_config,
        _recursive_=False,
    )
    module.bs = 3
    module.patch_size = 48
    input_batch = generate_mm_input(3, 4, 4, 48, 48)
    out_module = module.shared_step(input_batch)

    print(out_module.loss)
