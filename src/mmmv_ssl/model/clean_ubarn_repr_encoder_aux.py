# pylint: disable=invalid-name
"""Ubarn Aux encoder"""

import logging
from typing import Literal

import torch
from einops import repeat, rearrange
from torch import nn

from mmmv_ssl.data.dataclass import BatchOneMod
from mmmv_ssl.model.clean_ubarn import CleanUBarn, HSSEncoding
from mmmv_ssl.model.datatypes import \
    CleanUBarnConfig, UnetConfig, MeteoConfig, BOutputUBarn, BOutputReprEncoder

my_logger = logging.getLogger(__name__)


class MeteoEncoder(nn.Module):
    """MLP encoder for meteo vectors"""

    def __init__(
            self,
            config: MeteoConfig,
            input_channels: int
    ):
        super().__init__()
        width = [input_channels] + config.widths
        layers = []
        for i in range(len(width) - 1):
            layers.extend([nn.Linear(width[i], width[i + 1]),
                           nn.ReLU()])
        self.encoder = nn.Sequential(*layers)
        self.before_unet = config.concat_before_unet

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
        """Forward pass for meteo data. MLP layers"""
        b, n, _, _ = x.shape
        x = rearrange(x, "b n c d -> (b n) (c d) ")
        if mask is not None:
            # we ignore padded values
            mask = rearrange(mask, "b n -> (b n )")
            x = self.encoder(x[~mask])
            x_res = torch.zeros(b * n, x.shape[1]).to(x.device)
            x_res[~mask] = x
            x = x_res
        else:
            x = self.encoder(x)
        x = rearrange(x, "( b n ) cd  -> b n cd ", b=b)
        return x


class CleanUBarnAux(CleanUBarn):
    """
    Clean UBarn Aux module to encode S1/S2
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
            pe_cst: int = 10000,
            use_transformer: bool = True,
            use_pytorch_transformer: bool = True,
    ):
        super().__init__(ne_layers,
                         d_model,
                         d_hidden,
                         dropout,
                         block_name,
                         norm_first,
                         input_channels,
                         nhead,
                         attn_dropout,
                         encoding_config,
                         pe_cst,
                         use_transformer,
                         use_pytorch_transformer)

        # dem encoder is instanced in MALICE Encoder module,
        # as it is common for S1 and S2 encoder
        self.encoder_dem: HSSEncoding  # TODO do smth better

        # for the moment meteo encoder is different for S1 and S2
        self.encoder_meteo: MeteoEncoder

    def forward(
            self,
            batch_input: BatchOneMod,
            dem: torch.Tensor,
    ) -> BOutputUBarn:
        """
        Forward pass
        """

        padd_index = batch_input.padd_index

        # We encode dem and expand it to two views
        x_dem = self.encoder_dem(dem)
        if batch_input.sits.shape[0] / dem.shape[0] == 2:
            # We keep this option for inference when we don't have views
            x_dem = torch.cat((x_dem, x_dem), dim=0)

        meteo_encoded = self.encoder_meteo(
            batch_input.meteo, mask=padd_index
        )[:, :, :, None, None].expand(-1, -1, -1, *batch_input.sits.shape[-2:])
        if self.encoder_meteo.before_unet:
            x, doy_encoding = self.patch_encoding(
                torch.concat([batch_input.sits, meteo_encoded], 2),
                batch_input.input_doy,
                mask=padd_index
            )
        else:
            raise NotImplementedError

        # We find the first "no data" padding index to attribute dem value to it
        to_replace = (~batch_input.padd_index).sum(1)

        my_logger.debug(f"x{x.shape} doy {doy_encoding.shape}")
        if self.temporal_encoder is not None:
            x = x + doy_encoding
            padd_index[torch.arange(x.shape[0]), to_replace.int()] = False
            x[torch.arange(x.shape[0]), to_replace.int()] = x_dem.squeeze(1).to(x.device)
            padd = padd_index
            b, _, _, h, w = x.shape
            if isinstance(self.temporal_encoder, nn.TransformerEncoder):
                padd_index = repeat(
                    padd_index, "b t -> b t h w ", h=h, w=w
                )
                padd_index = rearrange(padd_index, " b t h w -> (b h w ) t")
                x = rearrange(x, "b t c h w -> (b h w ) t c")

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
            return BOutputUBarn(x, padd_index=padd)
        padd_index[torch.arange(x.shape[0]), to_replace.int()] = False
        x[torch.arange(x.shape[0]), to_replace.int()] = x_dem.squeeze(1).to(x.device)
        return BOutputUBarn(x, padd_index=padd_index)


class CleanUBarnReprEncoderAux(nn.Module):
    """Entry for Ubarn aux encoder"""
    def __init__(
            self,
            ubarn_config: CleanUBarnConfig,
            d_model: int,
            input_channels: int = 10,
            use_pytorch_transformer: bool = False,
    ):
        super().__init__()
        self.ubarn = CleanUBarnAux(
            ne_layers=ubarn_config.ne_layers,
            d_model=d_model,
            d_hidden=ubarn_config.d_hidden,
            dropout=ubarn_config.dropout,
            block_name=ubarn_config.block_name,
            norm_first=ubarn_config.norm_first,
            input_channels=input_channels,
            nhead=ubarn_config.nhead,
            attn_dropout=ubarn_config.attn_dropout,
            encoding_config=ubarn_config.encoding_config,
            use_pytorch_transformer=use_pytorch_transformer,
        )

    def forward(
            self,
            batch_input: BatchOneMod,
            dem: torch.Tensor,
    ) -> BOutputReprEncoder:
        """Forward pass"""
        batch_output = self.ubarn(batch_input, dem=dem)

        return BOutputReprEncoder(
            repr=batch_output.output,
            doy=batch_input.input_doy,
            padd_index=batch_output.padd_index,
        )
