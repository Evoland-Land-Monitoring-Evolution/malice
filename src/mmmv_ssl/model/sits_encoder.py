import logging
from abc import ABC

from einops import rearrange, repeat
from mt_ssl.data.classif_class import ClassifBInput
from mt_ssl.data.mt_batch import BOutputReprEnco
from mt_ssl.model.template_repr_encoder import TemplateReprEncoder
from mt_ssl.model.ubarn import UBarn
from omegaconf import DictConfig
from torch import nn

from mmmv_ssl.model.temp_proj import CrossAttn

my_logger = logging.getLogger(__name__)


class MonoSITSEncoder(TemplateReprEncoder, ABC):
    def __init__(
        self,
        ubarn: UBarn | DictConfig,
        d_model,
        input_channels,
        temp_proj_one_mod: CrossAttn,
        query: nn.Parameter,
        *args,
        **kwargs,
    ):
        super().__init__(ubarn, d_model, input_channels, *args, **kwargs)
        self.query = query
        self.temp_proj = temp_proj_one_mod
        self.nq = self.query.shape[1]

    def forward(
        self, batch_input: ClassifBInput, return_attns=False
    ) -> BOutputReprEnco:
        batch_output = self.ubarn(batch_input, return_attns=False)
        padd_mask = repeat(
            batch_input.padd_index,
            "b n -> b n h w ",
            h=batch_input.h,
            w=batch_input.w,
        )
        padd_mask = rearrange(padd_mask, " b n h w -> (b h w ) n ")

        repr = self.temp_proj(
            v=rearrange(
                batch_output.output,
                "B n c h w -> (B h w ) n c",
            ),
            q=self.query,
            pad_mask=~padd_mask,
        )
        my_logger.debug(f"repr shape{repr.shape}")
        return BOutputReprEnco(
            repr=repr, doy=batch_input.input_doy, attn_ubarn=batch_output.attn
        )

    def forward_keep_input_dim(
        self, batch_input: ClassifBInput, returns_attns: bool = False
    ) -> BOutputReprEnco:
        b, n, c, h, w = batch_input.sits.shape
        out = self.forward(batch_input)
        repr = rearrange(
            out.repr,
            "(B h w ) n c -> B n c h w",
            B=b,
            c=self.d_model,
            h=h,
            w=w,
        )
        return BOutputReprEnco(
            repr=repr,
            doy=batch_input.input_doy,
            attn_ubarn=out.attn_ubarn,
        )
