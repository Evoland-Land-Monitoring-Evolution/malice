import logging

import numpy as np
import torch
from einops import rearrange, repeat
from mt_ssl.data.mt_batch import BOutputReprEnco
from torch import nn

from mmmv_ssl.data.dataclass import merge2views, BatchMMSits, BatchOneMod
from mmmv_ssl.model.clean_ubarn_repr_encoder import CleanUBarnReprEncoder
from mmmv_ssl.model.dataclass import OutTempProjForward
from mmmv_ssl.model.datatypes import EncoderConfig, DecoderConfig, DataInputChannels
from mmmv_ssl.model.decodeur import MetaDecoder
from mmmv_ssl.model.projector import IdentityProj, AliseProj
from mmmv_ssl.model.query_utils import TempMetaQuery
from mmmv_ssl.model.temp_proj import TemporalProjector
from mmmv_ssl.module.dataclass import LatRepr, Rec, RecWithOrigin, OutMMAliseF

my_logger = logging.getLogger(__name__)


class AliseMMModule(nn.Module):
    def __init__(self,
                 encoder: EncoderConfig,
                 decoder: DecoderConfig,
                 input_channels: DataInputChannels = DataInputChannels(),
                 d_repr: int = 64,
                 ):
        super().__init__()
        self.encoder = MaliceEncoder(encoder,
                                     input_channels=input_channels,
                                     d_repr=d_repr)

        self.decoder = MaliceDecoder(decoder,
                                     input_channels=input_channels,
                                     d_repr=d_repr)

    def forward(self, batch):
        repr, out_emb, mm_embedding = self.encoder(batch)
        reconstructions = self.decoder(batch, mm_embedding)
        return OutMMAliseF(repr=repr, rec=reconstructions, emb=out_emb)


class MaliceEncoder(nn.Module):
    def __init__(self,
                 encoder: EncoderConfig,
                 input_channels: DataInputChannels = DataInputChannels(),
                 d_repr: int = 64,
                 ):
        """
        Encoder module.
        Contains two Ubarn encoders for S1 and S2, common temporal projector
        and embedding projector (for invariance loss)
        """
        super().__init__()

        self.encoder_s1 = CleanUBarnReprEncoder(ubarn_config=encoder.encoder_s1,
                                                d_model=d_repr,
                                                input_channels=input_channels.s1,
                                                )
        self.encoder_s2 = CleanUBarnReprEncoder(ubarn_config=encoder.encoder_s2,
                                                d_model=d_repr,
                                                input_channels=input_channels.s2,
                                                )

        self.common_temp_proj = TemporalProjector(
            input_channels=d_repr,
            num_heads=encoder.common_temp_proj.num_heads,
            n_q=encoder.common_temp_proj.n_q,
        )

        if encoder.projector is None:
            self.projector_emb = IdentityProj()
        else:
            self.projector_emb = AliseProj(input_channels=d_repr,
                                           out_channels=encoder.projector.out_channels,
                                           l_dim=encoder.projector.l_dim,
                                           freeze=encoder.projector.freeze)

    def compute_mm_embeddings(self, aligned_repr: OutTempProjForward, h: int, w: int):
        """
        Compute multimodal embeddings for Invariance loss
        """
        embedding_s1 = rearrange(
            aligned_repr.s1, "(view bhw) t c-> view bhw t c", view=2
        )

        embedding_s2 = rearrange(
            aligned_repr.s2, "(view bhw) t c-> view bhw t c", view=2
        )
        embeddings = rearrange(
            torch.cat([embedding_s2, embedding_s1], dim=0),
            "view bhw t c -> (view bhw) t c",
        )
        embeddings = self.projector_emb(embeddings)
        embeddings = rearrange(
            embeddings, "(view bhw) t c -> view bhw t c", view=4
        )
        mm_embedding = torch.cat(
            [
                embedding_s2[0, ...],
                embedding_s2[0, ...],
                embedding_s2[1, ...],
                embedding_s2[1, ...],
                embedding_s1[0, ...],
                embedding_s1[0, ...],
                embedding_s1[1, ...],
                embedding_s1[1, ...],
            ],
            dim=0,
        )  # should be s2a,s2a,s2b,s2b,s1a,s1a,s1b,s1b

        rearrange_embeddings = lambda x: rearrange(
            x, "(b h w) t c-> b t c h w", h=h, w=w
        )

        repr = LatRepr(*
                       [rearrange_embeddings(emb) for emb in
                        [
                            embedding_s1[0, ...],
                            embedding_s1[1, ...],
                            embedding_s2[0, ...],
                            embedding_s2[1, ...],
                        ]
                        ])
        out_emb = LatRepr(
            s2a=embeddings[0, ...],
            s2b=embeddings[1, ...],
            s1a=embeddings[2, ...],
            s1b=embeddings[3, ...],
        )

        return repr, out_emb, mm_embedding

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

        mask_tp = repeat(~merged_views.padd_index.bool(), "b t -> b t h w", h=h, w=w)
        my_logger.debug(f"{sat} repr {out.repr.shape}")
        if isinstance(
                self.encoder_s1.ubarn.temporal_encoder, nn.TransformerEncoderLayer
        ):
            padd = None
        else:
            padd = rearrange(mask_tp, "b t h w -> (b h w) t ")
        return out, padd

    def forward(self, batch: BatchMMSits):
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

        repr, out_emb, mm_embedding = self.compute_mm_embeddings(aligned_repr, batch.sits1a.h, batch.sits1a.w)

        return repr, out_emb, mm_embedding


class MaliceDecoder(nn.Module):
    def __init__(self, decoder: DecoderConfig,
                 input_channels: DataInputChannels = DataInputChannels(),
                 d_repr: int = 64,
                 ):
        super().__init__()

        self.meta_decoder = MetaDecoder(
            num_heads=decoder.meta_decoder.num_heads,
            input_channels=d_repr,
            d_q_in=(decoder.query_s1s2_d + decoder.pe_channels),
            d_k=decoder.meta_decoder.d_k,
            intermediate_layers=decoder.meta_decoder.intermediate_layers
        )

        self.query_builder = TempMetaQuery(
            pe_config=None, input_channels=decoder.pe_channels
        )

        self.q_decod_s1 = self.define_query_decoder(decoder.query_s1s2_d)
        self.q_decod_s2 = self.define_query_decoder(decoder.query_s1s2_d)

        self.proj_s1 = torch.nn.Linear(
            self.meta_decoder.input_channels, input_channels.s1
        )

        self.proj_s2 = torch.nn.Linear(
            self.meta_decoder.input_channels, input_channels.s2
        )

    def compute_query(self, batch_sits: BatchOneMod, sat: str = "s2"):
        """Compute query and reshape it"""
        query = repeat(
            self.query_builder(self.q_decod_s2 if "2" in sat else self.q_decod_s1, batch_sits.input_doy),
            "b t c -> b t c h w",
            h=batch_sits.h,
            w=batch_sits.w,
        )
        return rearrange(
            query,
            "b t (nh c )h w -> nh (b h w) t c",
            nh=self.meta_decoder.num_heads,
        )

    @staticmethod
    def define_query_decoder(query: int) -> nn.Parameter:
        q_decod = nn.Parameter(
            torch.zeros(query)
        ).requires_grad_(True)
        nn.init.normal_(
            q_decod, mean=0, std=np.sqrt(2.0 / query)
        )  # TODO check that
        return q_decod

    def forward(self, batch: BatchMMSits, mm_embedding: torch.Tensor) -> Rec:
        # Decoder!

        query_s2a = self.compute_query(batch_sits=batch.sits2a, sat="s2")
        query_s2b = self.compute_query(batch_sits=batch.sits2b, sat="s2")
        query_s1a = self.compute_query(batch_sits=batch.sits1a, sat="s1")
        query_s1b = self.compute_query(batch_sits=batch.sits1b, sat="s1")
        my_logger.debug(f"query decoder s2 {query_s2b.shape}")

        mm_queries = torch.cat(
            [
                query_s2b, query_s1a, query_s2a, query_s1b,
                query_s1b, query_s2a, query_s1a, query_s2b,
            ],
            dim=1,
        )
        my_logger.debug(f"queries{mm_queries.shape}")

        h, w = batch.sits2a.h, batch.sits2a.w
        padd_mm = torch.cat(
            [
                repeat(batch.sits2b.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits2a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1b.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1b.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits2a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits2b.padd_index, "b t -> (b h w) t ", h=h, w=w),
            ],
            dim=0,
        )  # 2(b h w) t
        # print(f"padd mask {padd_mm.shape}")

        out = self.meta_decoder.forward(
            mm_sits=mm_embedding, padd_mm=padd_mm, mm_queries=mm_queries
        )  # (2bhw) t d

        out = rearrange(
            out,
            "(mod b h w) t c-> mod b t h w c",
            mod=8,
            b=batch.sits1a.b,
            h=batch.sits1a.h,
            w=batch.sits1a.w,
        )
        s1_rec = self.proj_s1(
            rearrange(
                out[[1, 3, 4, 6], ...], "mod b t c h w -> (mod b ) t c h w"
            )
        )  #
        s2_rec = self.proj_s2(
            rearrange(
                out[[0, 2, 5, 7], ...], "mod b t c h w -> (mod b) t c h w"
            )
        )
        s1_rec = rearrange(s1_rec, "(mod b) t h w c -> mod b t c h w", mod=4)
        s2_rec = rearrange(s2_rec, "(mod b) t h w c -> mod b t c h w", mod=4)

        rec = Rec(
            s1a=RecWithOrigin(
                same_mod=s1_rec[3, ...], other_mod=s1_rec[0, ...]
            ),
            s1b=RecWithOrigin(
                same_mod=s1_rec[2, ...], other_mod=s1_rec[1, ...]
            ),
            s2a=RecWithOrigin(
                same_mod=s2_rec[1, ...], other_mod=s2_rec[2, ...]
            ),
            s2b=RecWithOrigin(
                same_mod=s2_rec[0, ...], other_mod=s2_rec[3, ...]
            ),
        )

        return rec
