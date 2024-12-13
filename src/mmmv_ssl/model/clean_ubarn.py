# pylint: disable=invalid-name

"""
Classes relevant to Ubarn module that performs S2 or S1 encoding.
Part of Malice Encoder
"""

import logging
from typing import Literal

import torch
from torch import nn
from einops import rearrange, repeat

from mt_ssl.data.mt_batch import BInput5d, BOutputUBarn
from mt_ssl.model.attention import MultiHeadAttention
from mt_ssl.model.convolutionalblock import ConvBlock
from mt_ssl.model.norm import AdaptedLayerNorm
from mt_ssl.model.utae_unet import Unet
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
            batch: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch]:
        """

        Args:
            batch ():
            src_mask ():
            key_padding_mask (): True indicates which elements within ``key``
            to ignore for the purpose of attention

        Returns:

        """

        if self.norm_first:
            x = batch
            output, enc_slf_attn = self._sa_block(
                self.norm(batch),
                mask=src_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + output
            x = x + self._conv_block(self.norm2(x))
        else:
            x = batch
            output, enc_slf_attn = self._sa_block(
                self.norm(batch),
                mask=src_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm(x + output)
            x = self.norm2(x + self._conv_block(x))

        return x, enc_slf_attn

    def _sa_block(self,
                  x: torch.Tensor,
                  mask: torch.Tensor,
                  key_padding_mask: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        x, p_attn = self.attn(
            x, x, x, mask=mask, key_padding_mask=key_padding_mask
        )
        return self.dropout(x), p_attn

    def _conv_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class Encoder2(nn.Module):
    """Inspired from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob
    /132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Models.py#L48"""

    def __init__(
            self,
            n_layers: int,
            d_model: int,
            d_in: int,
            dropout: float = 0.1,
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
            batch: torch.Tensor,
            return_attns: bool = False,
            src_key_padding_mask: None | torch.Tensor = None,
            src_mask: None | torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Forward pass"""
        my_logger.debug(f"input enc {batch.shape}")
        enc_slf_attn_list = []
        # enc_output = batch
        enc_output = self.dropout(batch)
        # some values are not there (b,n enc_output=self.layer_norm(ind_modelput)
        # #not sure layer norm is adapted to our
        # issue ...
        for enc_layer in self.layer_stack:
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
    """
    Clean UBarn module for S1/S2 encoders
    """

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
            use_pytorch_transformer: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_channels = input_channels

        self.patch_encoding = InputEncoding(
            inplanes=input_channels,
            planes=d_model,
            unet_config=encoding_config,
            pe_cst=pe_cst,
        )


        if use_transformer:
            if use_pytorch_transformer:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_hidden,
                    dropout=dropout,
                    norm_first=norm_first,
                    batch_first=True,
                )
                self.temporal_encoder = nn.TransformerEncoder(
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
            batch_input: BInput5d,
    ) -> BOutputUBarn:
        """
        Forward pass
        """
        x, doy_encoding = self.patch_encoding(
            batch_input.sits, batch_input.input_doy, mask=batch_input.padd_index
        )

        my_logger.debug(f"x{x.shape} doy {doy_encoding.shape}")
        if self.temporal_encoder is not None:
            x = x + doy_encoding
            # print(batch_input.padd_index.shape)
            # print(x.shape)
            b, _, _, h, w = x.shape
            if isinstance(self.temporal_encoder, nn.TransformerEncoder):
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
            if isinstance(self.temporal_encoder, nn.TransformerEncoder):
                x = rearrange(x, "(b h w ) t c -> b t c h w ", b=b, h=h, w=w)
            my_logger.debug(f"output ubarn clean {x.shape}")
            return BOutputUBarn(x)

        return BOutputUBarn(x)


class InputEncoding(nn.Module):
    """
    Input encoding class:
    spectro-spatial encoding + positional encoding
    """

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
            self, x: torch.Tensor, doy_list: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        input_encodding = self.hss_encoding(x, mask)
        doy_encoding = self.pe_encoding(doy_list)
        doy_encoding = rearrange(doy_encoding, " b n c-> b n c 1 1")
        return input_encodding, doy_encoding.to(input_encodding)


class HSSEncoding(nn.Module):
    """Hyperspectral & spatial encoding """

    def __init__(
            self,
            input_channels: int,
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass"""
        b, n, _, h, w = x.shape
        x = rearrange(x, "b n c h w -> (b n ) c h w")
        if mask is not None:
            mask = rearrange(mask, "b n -> (b n )")
            x = self.my_model(x[~mask])
            x_res = torch.zeros(b * n, x.shape[1], h, w).to(x.device)
            x_res[~mask] = x
            x = x_res
        else:
            x = self.my_model(x)
        x = rearrange(x, "( b n ) c h w -> b n c h w ", b=b)
        return x
