# pylint: disable=invalid-name

"""
Malice torch module with general Malice class
end encoder and decoder classes
"""

import logging

import numpy as np
import torch
from einops import rearrange, repeat
from mmmv_ssl.model.clean_ubarn import HSSEncoding
from mmmv_ssl.model.malice_module import MaliceEncoder, AliseMMModule, MaliceDecoder
from torch import nn

from mt_ssl.data.mt_batch import BOutputReprEnco

from mmmv_ssl.data.dataclass import merge2views, BatchMMSits
from mmmv_ssl.model.clean_ubarn_repr_encoder import CleanUBarnReprEncoder
from mmmv_ssl.model.dataclass import OutTempProjForward
from mmmv_ssl.model.datatypes import EncoderConfig, DecoderConfig, DataInputChannels
from mmmv_ssl.module.dataclass import LatRepr, OutMMAliseF

my_logger = logging.getLogger(__name__)


class AliseMMModuleAux(AliseMMModule):
    """
    Malice Module composed of encoder and decoder
    """

    def __init__(self,
                 encoder: EncoderConfig,
                 decoder: DecoderConfig,
                 input_channels: DataInputChannels = DataInputChannels(),
                 d_repr: int = 64,
                 ):
        super().__init__(encoder, decoder, input_channels, d_repr)
        self.encoder = MaliceEncoderAux(encoder,
                                     input_channels=input_channels,
                                     d_repr=d_repr)
        self.decoder = MaliceDecoder(decoder,
                                     input_channels=input_channels,
                                     d_repr=d_repr)

    def forward(self, batch: BatchMMSits) -> OutMMAliseF:
        """Forward pass"""
        reprojected, out_emb, mm_embedding = self.encoder(batch)
        reconstructions = self.decoder(batch, mm_embedding)
        return OutMMAliseF(repr=reprojected, rec=reconstructions, emb=out_emb)


class MaliceEncoderAux(MaliceEncoder):
    """
    Encoder module.
    Contains two Ubarn encoders for S1 and S2, common temporal projector
    and embedding projector (for invariance loss)
    """

    def __init__(self,
                 encoder: EncoderConfig,
                 input_channels: DataInputChannels = DataInputChannels(),
                 d_repr: int = 64,
                 ):
        super().__init__(encoder, input_channels, d_repr)

        self.encoder_s1 = CleanUBarnReprEncoder(ubarn_config=encoder.encoder_s1,
                                                d_model=d_repr,
                                                input_channels=input_channels.s1 + input_channels.s1_aux,
                                                )
        self.encoder_s2 = CleanUBarnReprEncoder(ubarn_config=encoder.encoder_s2,
                                                d_model=d_repr,
                                                input_channels=input_channels.s2 + input_channels.s2_aux,
                                                )
        self.encoder_dem = HSSEncoding(
            input_channels=input_channels.dem, d_model=d_repr, model_config=unet_config
        )

    def encode_views(self, batch: BatchMMSits, sat: str) -> tuple[BOutputReprEnco, torch.Tensor]:
        """Get two view of one satellite and encode them with encoder"""
        if "1" in sat:
            view1, view2 = batch.sits1a, batch.sits1b
        else:
            view1, view2 = batch.sits2a, batch.sits2b
        h = view1.h
        w = view1.w
        merged_views = merge2views(view1, view2)

        out = self.encoder_s1.forward_keep_input_dim(merged_views) if "1" in sat \
            else self.encoder_s2.forward_keep_input_dim(merged_views)

        dem_emb = self.encoder_dem()

        mask_tp = repeat(~merged_views.padd_index.bool(), "b t -> b t h w", h=h, w=w)
        my_logger.debug(f"{sat} repr {out.repr.shape}")
        if isinstance(
                self.encoder_s1.ubarn.temporal_encoder, nn.TransformerEncoderLayer
        ):
            padd = None
        else:
            padd = rearrange(mask_tp, "b t h w -> (b h w) t ")
        return out, padd

    def forward(self, batch: BatchMMSits) -> tuple[LatRepr, LatRepr, torch.Tensor]:
        """
        Malice Encoder forward step.
        """
        out_s1, padd_s1 = self.encode_views(batch, sat="s1")
        out_s2, padd_s2 = self.encode_views(batch, sat="s2")

        aligned_repr: OutTempProjForward = self.common_temp_proj(
            sits_s1=rearrange(out_s1.repr, "b t c h w -> (b h w ) t c"),
            padd_s1=padd_s1,
            sits_s2=rearrange(out_s2.repr, "b t c h w -> (b h w) t c"),
            padd_s2=padd_s2,
        )

        reprojected, out_emb, mm_embedding = self.compute_mm_embeddings(
            aligned_repr, batch.sits1a.h, batch.sits1a.w
        )

        return reprojected, out_emb, mm_embedding

