import logging

import torch
import torch.nn as nn
from hydra.utils import instantiate
from mt_ssl.data.mt_batch import BInput5d, BOutputReprEnco
from mt_ssl.model.fc_encoding import FCEncoding
from mt_ssl.model.template_repr_encoder import BaseReprEncoder
from mt_ssl.model.ubarn import UBarn
from omegaconf import DictConfig

from mmmv_ssl.model.clean_ubarn import CleanUBarn

my_logger = logging.getLogger(__name__)


class CleanTemplateReprEncoder(BaseReprEncoder):
    def __init__(
        self,
        ubarn: UBarn | FCEncoding | DictConfig,
        d_model,
        input_channels,
        pe_module: nn.Module | None = None,
        use_pytorch_transformer=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(ubarn, CleanUBarn):
            self.ubarn = ubarn
        else:
            self.ubarn: UBarn = instantiate(
                ubarn,
                input_channels=input_channels,
                _recursive_=False,
                d_model=d_model,
                use_pytorch_transformer=use_pytorch_transformer,
            )
        self.d_model = d_model

    def load_ubarn(self, path_ckpt):
        my_logger.info(f"We load state dict  from {path_ckpt}")
        if not torch.cuda.is_available():
            map_params = {"map_location": "cpu"}
        else:
            map_params = {}
        ckpt = torch.load(path_ckpt, **map_params)
        self.ubarn.load_state_dict(ckpt["ubarn_state_dict"])


class CleanUBarnReprEncoder(CleanTemplateReprEncoder):
    def __init__(
        self,
        ubarn: DictConfig | CleanUBarn,
        d_model: int,
        input_channels=10,
        reference_time_points=None,
        override_pe: bool = True,
        use_pytorch_transformer=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            ubarn,
            d_model,
            input_channels,
            use_pytorch_transformer=use_pytorch_transformer,
            *args,
            **kwargs,
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
