# pylint: disable=invalid-name
"""
Encoder class for inference.
Takes one modality and produces its embeddings.
"""

import logging

import torch
from einops import rearrange, repeat
from torch import nn

from mmmv_ssl.data.dataclass import BatchOneMod
from mmmv_ssl.model.clean_ubarn_repr_encoder import CleanUBarnReprEncoder
from mmmv_ssl.model.datatypes import BOutputReprEncoder
from mmmv_ssl.model.temp_proj import CrossAttn

my_logger = logging.getLogger(__name__)


class MonoSITSEncoder(nn.Module):
    """
    Encoder class for inference.
    Takes one modality and produces its embeddings.
    """

    def __init__(
            self,
            encoder: CleanUBarnReprEncoder,
            temp_proj_one_mod: CrossAttn,
            query: nn.Parameter,
    ):
        super().__init__()
        self.query = query
        self.temp_proj = temp_proj_one_mod
        self.nq = self.query.shape[1]
        self.encoder = encoder

    def forward(
            self, batch_input: BatchOneMod,
    ) -> BOutputReprEncoder:
        """Forward pass: SSTE encoder + temporal projector"""
        batch_output: BOutputReprEncoder = self.encoder(batch_input)
        mask_tp = repeat(~batch_input.padd_index.bool(),
                         "b t -> b t h w", h=batch_input.h, w=batch_input.w)
        padd = rearrange(mask_tp, "b t h w -> (b h w) t ")

        repro = self.temp_proj(
            v=rearrange(
                batch_output.repr,
                "B n c h w -> (B h w ) n c",
            ),
            q=self.query,
            pad_mask=padd,
        )
        my_logger.debug(f"repr shape{repro.shape}")
        return BOutputReprEncoder(
            repr=repro, doy=batch_output.doy
        )

    def reshape(self,
                batch_input: BatchOneMod,
                batch_output: BOutputReprEncoder
                ) -> BOutputReprEncoder:
        """
        Reshape SITS to get spatial dims.
        """
        b, _, _, h, w = batch_input.sits.shape
        repro = rearrange(
            batch_output.repr,
            "(B h w ) n c -> B n c h w",
            B=b,
            c=self.encoder.ubarn.d_model,
            h=h,
            w=w,
        )
        return BOutputReprEncoder(
            repr=repro,
            doy=batch_input.input_doy,
        )

    def forward_keep_input_dim(
            self, batch_input: BatchOneMod
    ) -> BOutputReprEncoder:
        """Forward pass. Keep input dims"""
        out = self.forward(batch_input)
        return self.reshape(batch_input, out)


class MonoSITSAuxEncoder(MonoSITSEncoder):
    """
    Mono-modal malice encoder with auxillairy data
    """


    def forward(
            self, batch_input: BatchOneMod,
            dem: torch.Tensor
    ) -> BOutputReprEncoder:
        """
        Forward pas. Encoder images and project temporally
        """
        batch_output: BOutputReprEncoder = self.encoder(batch_input, dem)
        h, w = batch_output.repr.shape[-2:]
        mask_tp = repeat(~batch_output.padd_index.bool(), "b t -> b t h w", h=h, w=w)
        padd = rearrange(mask_tp, "b t h w -> (b h w) t ")

        repro = self.temp_proj(
            v=rearrange(
                batch_output.repr,
                "B n c h w -> (B h w ) n c",
            ),
            q=self.query,
            pad_mask=padd,
        )
        my_logger.debug(f"repr shape{repro.shape}")
        return BOutputReprEncoder(
            repr=repro, doy=batch_output.doy
        )

    def forward_keep_input_dim(
            self, batch_input: BatchOneMod, dem: torch.Tensor
    ) -> BOutputReprEncoder:
        """Forward pass. Keep input dims"""
        out = self.forward(batch_input, dem)
        return self.reshape(batch_input, out)
