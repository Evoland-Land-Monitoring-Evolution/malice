import logging

from mt_ssl.data.mt_batch import BInput5d, BOutputReprEnco
from mt_ssl.model.template_repr_encoder import TemplateReprEncoder
from omegaconf import DictConfig

from mmmv_ssl.model.clean_ubarn import CleanUBarn

my_logger = logging.getLogger(__name__)


class CleanUBarnReprEncoder(TemplateReprEncoder):
    def __init__(
        self,
        ubarn: DictConfig | CleanUBarn,
        d_model: int,
        input_channels=10,
        reference_time_points=None,
        override_pe: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            ubarn,
            d_model,
            input_channels,
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
