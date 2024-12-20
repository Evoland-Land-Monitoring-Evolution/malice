import logging

import torch.nn as nn

from mmmv_ssl.data.dataclass import BatchOneMod
from mmmv_ssl.model.clean_ubarn import CleanUBarn
from mmmv_ssl.model.datatypes import CleanUBarnConfig
from mt_ssl.data.mt_batch import BOutputReprEnco

my_logger = logging.getLogger(__name__)


class CleanUBarnReprEncoder(nn.Module):
    def __init__(
            self,
            ubarn_config: CleanUBarnConfig,
            d_model: int,
            input_channels: int = 10,
            use_pytorch_transformer: bool = False,
    ):
        super().__init__()
        self.ubarn = CleanUBarn(
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
    ) -> BOutputReprEnco:
        batch_output = self.ubarn(batch_input)

        return BOutputReprEnco(
            repr=batch_output.output,
            doy=batch_input.input_doy,
            attn_ubarn=batch_output.attn,
        )
