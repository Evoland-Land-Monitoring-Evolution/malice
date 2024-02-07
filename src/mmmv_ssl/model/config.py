from dataclasses import dataclass
from typing import Any

from mmmv_ssl.model.projector import MLP
from mmmv_ssl.model.utae import UTAE


@dataclass
class UTAEConfig:
    _target_: Any = UTAE
    encoder_widths: Any = None
    decoder_widths: Any = None
    out_conv: Any = None
    str_conv_k: int = 4
    str_conv_s: int = 2
    str_conv_p: int = 1
    agg_mode: str = "att_group"
    encoder_norm: str = "group"
    n_head: int = 16
    d_model: int = 256
    d_k: int = 4
    encoder: bool = False
    return_maps: bool = True
    pad_value: int = 0
    padding_mode: str = "reflect"


@dataclass
class ProjectorConfig:
    _target_: Any = MLP
    d_in: int | None = None
    d_out: int | None = None
    l_dim: Any | None = None
    norm_layer: str = "batch_norm"
