import logging
from imaplib import Literal

import torch.nn as nn
from einops import repeat, rearrange
from mmmv_ssl.model.clean_ubarn import CleanUBarn, HSSEncoding
from mmmv_ssl.model.datatypes import CleanUBarnConfig, UnetConfig
from mt_ssl.data.mt_batch import BInput5d, BOutputReprEnco, BOutputUBarn



my_logger = logging.getLogger(__name__)


class CleanUBarnAux(CleanUBarn):
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
            encoding_dem_config: UnetConfig = UnetConfig()
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

        self.dem_encoding = HSSEncoding(
            input_channels=4, d_model=d_model, model_config=encoding_dem_config
        )


    def forward(
            self,
            batch_input: BInput5d,
            return_attns: bool,
    ) -> BOutputUBarn:
        """
        Forward pass
        """
        x, doy_encoding = self.patch_encoding(
            batch_input.sits, batch_input.input_doy, mask=batch_input.padd_index
        )
        x_dem = self.dem_encoding(batch_input.sits)

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

        return BOutputUBarn(x, None)


class CleanUBarnReprEncoder(nn.Module):
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
            batch_input: BInput5d,
            return_attns=True,
            mtan_grad: bool = True,
    ) -> BOutputReprEnco:
        batch_output = self.ubarn(batch_input, return_attns=return_attns)

        return BOutputReprEnco(
            repr=batch_output.output,
            doy=batch_input.input_doy,
            attn_ubarn=batch_output.attn,
        )

    def forward_keep_input_dim(
            self, batch_input: BInput5d, return_attns=True
    ) -> BOutputReprEnco:
        return self.forward(batch_input)
    #
    # def load_ubarn(self, path_ckpt):
    #     my_logger.info(f"We load state dict  from {path_ckpt}")
    #     if not torch.cuda.is_available():
    #         map_params = {"map_location": "cpu"}
    #     else:
    #         map_params = {}
    #     ckpt = torch.load(path_ckpt, **map_params)
    #     self.ubarn.load_state_dict(ckpt["ubarn_state_dict"])
