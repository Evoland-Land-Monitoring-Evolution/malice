import logging
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from mt_ssl.constant.dataset import ENCODE_PERMUTATED
from mt_ssl.data.bert_batch import InputBERTBatch
from mt_ssl.data.mask import permutate_flagged_patches
from mt_ssl.data.mt_batch import BInput5d, BOutputUBarn
from mt_ssl.model.attention import MultiHeadAttention
from mt_ssl.model.convolutionalblock import ConvBlock
from mt_ssl.model.norm import AdaptedLayerNorm
from mt_ssl.model.utae_unet import Unet
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mmmv_ssl.model.datatypes import UnetConfig
from mmmv_ssl.model.encoding import PositionalEncoder

my_logger = logging.getLogger(__name__)


class EncoderTransformerLayer2(nn.Module):
    """Transformer encoder layer"""

    def __init__(
            self,
            d_model: int,
            d_in: int,
            nhead: int = 1,
            norm_first: bool = False,
            block_name: Literal["se", "basicconv", "pff", "last", "cnn11"] = "se",
            dropout=0.1,
            attn_dropout: float = 0,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=attn_dropout
        )
        self.conv_block = ConvBlock(
            inplanes=d_model,
            planes=d_in,
            block_name=block_name,
            dropout=dropout,
        )
        self.norm = AdaptedLayerNorm(d_model, change_shape=True)
        self.norm2 = AdaptedLayerNorm(d_model, change_shape=True)
        self.norm_first = norm_first
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            input,
            src_mask=None,
            key_padding_mask=None,
            is_causal=False,
            src_key_padding_mask=None,
    ):
        """

        Args:
            input ():
            src_mask ():
            key_padding_mask (): True indicates which elements within ``key``
            to ignore for the purpose of attention

        Returns:

        """

        b, n, c, h, w = input.shape
        # input is (b,n,c,h,w)

        if self.norm_first:
            x = input
            output, enc_slf_attn = self._sa_block(
                self.norm(input),
                mask=src_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + output
            x = x + self._conv_block(self.norm2(x))
        else:
            x = input
            output, enc_slf_attn = self._sa_block(
                self.norm(input),
                mask=src_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm(x + output)
            x = self.norm2(x + self._conv_block(x))

        return x, enc_slf_attn

    def _sa_block(self, x, mask, key_padding_mask):
        x, p_attn = self.attn(
            x, x, x, mask=mask, key_padding_mask=key_padding_mask
        )
        return self.dropout(x), p_attn

    def _conv_block(self, x):
        return self.conv_block(x)


class Encoder2(nn.Module):
    """Inspired from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob
    /132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Models.py#L48"""

    def __init__(
            self,
            n_layers: int,
            d_model: int,
            d_in: int,
            dropout=0.1,
            block_name: Literal["se", "pff", "basicconv"] = "se",
            norm_first: bool = False,
            nhead: int = 1,
            attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.nhead = nhead
        self.layer_stack = nn.ModuleList(
            [
                EncoderTransformerLayer2(
                    d_model=d_model,
                    d_in=d_in,
                    nhead=nhead,
                    norm_first=norm_first,
                    block_name=block_name,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
            self,
            input: torch.Tensor,
            return_attns: bool = False,
            src_key_padding_mask: None | torch.Tensor = None,
            src_mask: None | torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        my_logger.debug(f"input enc   {input.shape}")
        enc_slf_attn_list = []
        # enc_output = input
        enc_output = self.dropout(input)
        # some values are not there (b,n enc_output=self.layer_norm(ind_modelput) #not sure layer norm is adapted to our
        # issue ...
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                key_padding_mask=src_key_padding_mask,
                src_mask=src_mask,
            )
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        # if return_attns:
        #     return enc_output, enc_slf_attn_list

        return enc_output


class CleanUBarn(nn.Module):
    def __init__(
        self,
        ne_layers: int,
        d_model: int = 256,
        d_hidden: int = 512,
        dropout: float = 0.1,
        block_name: Literal["se", "basicconv", "pff", "last"] = "pff",
        norm_first: bool = False,
        input_channels: int = 10,
        nhead: int = 4,
        attn_dropout: float = 0.1,
        encoding_config: UnetConfig = UnetConfig(),
        # max_len_pe: int = 3000,
        pe_cst: int = 10000,
        use_transformer: bool = True,
        use_pytorch_transformer:bool =True,
    ):
        super().__init__()
        # self.ne_layers = ne_layers
        # self.d_model = d_model
        # self.d_hidden = d_hidden
        # self.dropout = dropout
        # self.block_name = block_name
        # self.norm_first = norm_first
        # self.input_channels = input_channels
        # self.nhead = nhead
        # self.attn_dropout = attn_dropout
        # self.encoding_config = encoding_config
        # self.max_len_pe = max_len_pe
        # self.pe_cst = pe_cst

        self.patch_encoding = InputEncoding(
            inplanes=input_channels,
            planes=d_model,
            unet_config=encoding_config,
            pe_cst=pe_cst,
        )

        if use_transformer:
            if use_pytorch_transformer:
                encoder_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_hidden,
                    dropout=dropout,
                    norm_first=norm_first,
                    batch_first=True,
                )
                self.temporal_encoder = TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=ne_layers
                )
            else:
                self.temporal_encoder = Encoder2(
                    n_layers=ne_layers,
                    d_model=d_model,
                    d_in=d_hidden,
                    dropout=dropout,
                    block_name=block_name,
                    norm_first=norm_first,
                    nhead=nhead,
                    attn_dropout=attn_dropout,
                )
        else:
            self.temporal_encoder = None

    def forward(
        self,
        batch_input: BInput5d | InputBERTBatch,
        return_attns: bool,
        apply_corruption: bool = False,
    ) -> BOutputUBarn:
        """

        Args:
            batch_input ():
            return_attns ():

        Returns:

        """
        x, doy_encoding = self.patch_encoding(
            batch_input.sits, batch_input.input_doy, batch_input.true_doy, mask=batch_input.padd_index
        )
        if apply_corruption and isinstance(batch_input, InputBERTBatch):
            x = self.masking(
                x,
                mask_encoding=batch_input.corruption_mask,
                pad_index=batch_input.padd_index,
            )
        my_logger.debug(f"x{x.shape} doy {doy_encoding.shape}")
        if self.temporal_encoder is not None:
            x = x + doy_encoding
            # print(batch_input.padd_index.shape)
            # print(x.shape)
            b, t, c, h, w = x.shape
            if isinstance(self.temporal_encoder, TransformerEncoder):
                padd_index = repeat(
                    batch_input.padd_index, "b t -> b t h w ", h=h, w=w
                )
                padd_index = rearrange(padd_index, " b t h w -> (b h w ) t")
                x = rearrange(x, "b t c h w -> (b h w ) t c")
            else:
                padd_index = batch_input.padd_index
            my_logger.debug(f"before transformer {x.shape}")
            x = self.temporal_encoder(
                x,
                src_key_padding_mask=padd_index,
            )
            my_logger.debug(f"padd_index {padd_index[0, :]}")
            my_logger.debug(f"output ubarn clean {x.shape}")
            if isinstance(self.temporal_encoder, TransformerEncoder):
                x = rearrange(x, "(b h w ) t c -> b t c h w ", b=b, h=h, w=w)
            my_logger.debug(f"output ubarn clean {x.shape}")
            return BOutputUBarn(x)

        return BOutputUBarn(x, None)

    def masking(
            self, myinput: Tensor, mask_encoding: Tensor, pad_index: Tensor
    ):
        """

        Args:
            myinput ():
            mask_encoding ():
            pad_index ():

        Returns: apply the [MASK]ing strategy of the BERT: permutates the pixel value for the selected patch
        """
        x = myinput.clone()
        padd_val_dim = torch.count_nonzero(pad_index, dim=1)
        pad_val = torch.max(padd_val_dim)  # TODO is it really useful ??
        if pad_val != 0:
            nopad_x = x[:, :-pad_val, :, :, :]
        else:
            nopad_x = x
        nopad_x_shape = torch.Tensor(list(nopad_x.shape)).to(
            device=x.device, dtype=torch.int32
        )
        b, n, hdim, h, w = nopad_x_shape
        mask_encoding = torch.repeat_interleave(
            mask_encoding, repeats=hdim, dim=2
        )  # b t c h w "

        return permutate_flagged_patches(
            x, (mask_encoding == ENCODE_PERMUTATED)
        )


@dataclass
class ConfigUbarn:
    _target_: str = "mt_ssl.model.ubarn.UBarn"
    ne_layers: int = 3
    d_model: int = 256
    d_hidden: int = (512,)
    dropout: float = (0.1,)
    block_name: Literal["se", "basicconv", "pff", "last"] = ("pff",)
    norm_first: bool = (False,)
    input_channels: int = (10,)
    nhead: int = (4,)
    attn_dropout: float = (0.1,)
    encoding_config: DictConfig | None = (None,)
    pe_module: PositionalEncoder | None = None
    max_len_pe: int = (3000,)
    pe_cst: int = 10000
    use_transformer: bool = True
    args: list = None
    kwargs: dict = None


class InputEncoding(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            unet_config: UnetConfig,
            pe_cst: int = 1000,
    ):
        super().__init__()
        self.hss_encoding = HSSEncoding(
            input_channels=inplanes, d_model=planes, model_config=unet_config
        )
        self.pe_encoding = PositionalEncoder(d=planes, T=pe_cst)  # TODO: do smth here

    def forward(
            self, x: Tensor, doy_list: Tensor, true_doy_list: Tensor | None = None, mask: Tensor | None = None
    ):
        input_encodding = self.hss_encoding(x, mask)
        doy_encoding = self.pe_encoding(doy_list)
        doy_encoding = rearrange(doy_encoding, " b n c-> b n c 1 1")
        return input_encodding, doy_encoding.to(input_encodding)


class HSSEncoding(nn.Module):  # Hyperspectral & spatial encoding
    def __init__(
            self,
            input_channels,
            d_model: int = 32,
            model_config: UnetConfig | None = None,
    ):
        super().__init__()
        my_logger.debug(f"Input encoding planes is {d_model}")
        self.d_model = d_model

        self.my_model = Unet(
            inplanes=input_channels,
            planes=d_model,
            encoder_widths=model_config.encoder_widths,
            decoder_widths=model_config.decoder_widths,
            encoder_norm=model_config.encoder_norm,
            padding_mode=model_config.padding_mode,
            decoding_norm=model_config.decoding_norm,
            return_maps=model_config.return_maps,
            str_conv_k=model_config.str_conv_k,
            str_conv_s=model_config.str_conv_s,
            str_conv_p=model_config.str_conv_p,
            border_size=model_config.border_size,
            skip_conv_norm=model_config.skip_conv_norm)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, n, c, h, w = x.shape
        x = rearrange(x, "b n c h w -> (b n ) c h w")
        if mask is not None:
            mask = rearrange(mask, "b n -> (b n )")
            x = self.my_model(x[~mask])
            x_res = torch.zeros(b * n, x.shape[1], h, w)
            x_res[~mask] = x
            x = x_res
        else:
            x = self.my_model(x)
        x = rearrange(x, "( b n ) c h w -> b n c h w ", b=b)
        return x
