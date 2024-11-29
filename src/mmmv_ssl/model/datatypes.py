from dataclasses import dataclass
from typing import Literal


@dataclass
class UnetConfig:
    encoder_widths: list = (64, 64, 64, 128)
    decoder_widths: list = (32, 32, 64, 128)
    encoder_norm: Literal["group", "batch"] = "group"
    padding_mode: str = "reflect"
    decoding_norm: Literal["group", "batch"] = "group"
    return_maps: bool = False
    str_conv_k: int = 4
    str_conv_s: int = 2
    str_conv_p: int = 1
    border_size: int = 0
    skip_conv_norm: Literal["group", "batch"] = "group"


@dataclass
class CommonTempProjConfig:
    num_heads: int
    n_q: int


@dataclass
class AliseProjConfig:
    out_channels: int
    l_dim: list[int]
    freeze: bool = True


@dataclass
class CleanUBarnConfig:
    ne_layers: int
    d_model: int = 256
    d_hidden: int = 512
    dropout: float = 0.1
    block_name: Literal["se", "basicconv", "pff", "last"] = "pff"
    norm_first: bool = False
    input_channels: int = 10
    nhead: int = 4
    attn_dropout: float = 0.1,
    encoding_config: UnetConfig = UnetConfig()
    pe_cst: int = 10000
    use_transformer: bool = True
    use_pytorch_transformer: bool = True


@dataclass
class EncoderConfig:
    encoder_s1: CleanUBarnConfig
    encoder_s2: CleanUBarnConfig
    common_temp_proj: CommonTempProjConfig
    projector: AliseProjConfig | None = None


@dataclass
class MetaDecoderConfig:
    num_heads: int
    d_k: int
    intermediate_layers: None


@dataclass
class DecoderConfig:
    meta_decoder: MetaDecoderConfig
    query_s1s2_d: int = 64
    pe_channels: int = 64


@dataclass
class DataInputChannels:
    s1: int = 3
    s2: int = 10
