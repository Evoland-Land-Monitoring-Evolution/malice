"""
Configuration dataclasses for Malice model
"""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class UnetConfig:
    # pylint: disable=R0902
    """Unet class initialization config"""
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
class MeteoConfig:
    """
    Meteo Encoder class initialization config
    """
    widths: list[int] = (64, 64, 32)
    activation: str = "relu"
    common_s1_s2: bool = False
    # if we concatenate encoded meteo to images before or after we pass them to unet
    concat_before_unet: bool = True


@dataclass
class CommonTempProjConfig:
    """
    Common Temporal Projector class initialization config
    """
    num_heads: int
    n_q: int


@dataclass
class AliseProjConfig:
    """
    Embedding projector config
    """
    out_channels: int
    l_dim: list[int]
    freeze: bool = True


@dataclass
class CleanUBarnConfig:
    # pylint: disable=R0902
    """
    Clean UBarn config
    """
    ne_layers: int
    d_model: int = 256
    d_hidden: int = 512
    dropout: float = 0.1
    block_name: Literal["se", "basicconv", "pff", "last"] = "pff"
    norm_first: bool = False
    input_channels: int = 10
    nhead: int = 4
    attn_dropout: float = 0.1
    encoding_config: UnetConfig = UnetConfig()
    pe_cst: int = 10000
    use_transformer: bool = True
    use_pytorch_transformer: bool = True


@dataclass
class EncoderConfig:
    """
    Malice Encoder config
    """
    encoder_s1: CleanUBarnConfig
    encoder_s2: CleanUBarnConfig
    common_temp_proj: CommonTempProjConfig
    encoder_meteo: MeteoConfig | None = None
    projector: AliseProjConfig | None = None
    encoder_dem: CleanUBarnConfig | None = None


@dataclass
class MetaDecoderConfig:
    """
    Meta-decoder config
    """
    num_heads: int
    d_k: int
    intermediate_layers: None


@dataclass
class DecoderConfig:
    """
    Malice Decoder config
    """
    meta_decoder: MetaDecoderConfig
    query_s1s2_d: int = 64
    pe_channels: int = 64


@dataclass
class TransformerBlockConfig:
    """Transformer block configuration"""
    n_layers: int
    d_model: int
    d_in: int
    dropout = 0.1
    norm_first: bool = False
    n_head: int = 1


@dataclass
class DataInputChannels:
    """
    Model Input channels
    """
    s1: int = 3  # pylint: disable=C0103
    s2: int = 10  # pylint: disable=C0103
    s1_aux: int | None = 9
    s2_aux: int | None = 12
    s1_meteo: int | None = 8
    s2_meteo: int | None = 8
    dem: int | None = 4


@dataclass
class BOutputUBarn:
    """
    Ubarn output
    """
    output: torch.Tensor
    padd_index: torch.Tensor | None = None


@dataclass
class BOutputReprEncoder:
    """
    Class which encapsulate the output of the representation encoder
    """
    repr: torch.Tensor  # shape for mt_ssl (b f h w ) k c
    doy: torch.Tensor
    padd_index: None | torch.Tensor = None
