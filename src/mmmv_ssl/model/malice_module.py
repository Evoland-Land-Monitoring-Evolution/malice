import logging

import numpy as np
import torch
from einops import rearrange, repeat
from hydra.utils import instantiate
from mt_ssl.data.mt_batch import BOutputReprEnco
from omegaconf import DictConfig
from torch import nn

from mmmv_ssl.data.dataclass import merge2views, BatchMMSits, BatchOneMod
from mmmv_ssl.model.clean_ubarn_repr_encoder import CleanUBarnReprEncoder
from mmmv_ssl.model.dataclass import OutTempProjForward
from mmmv_ssl.model.decodeur import MetaDecoder
from mmmv_ssl.model.encoding import PositionalEncoder
from mmmv_ssl.model.projector import IdentityProj, ProjectorTemplate
from mmmv_ssl.model.query_utils import TempMetaQuery
from mmmv_ssl.model.temp_proj import TemporalProjector
from mmmv_ssl.module.dataclass import LatRepr, Rec, RecWithOrigin, OutMMAliseF

my_logger = logging.getLogger(__name__)


class AliseMMModule(nn.Module):
    def __init__(self, encoder, decoder, input_channels,
                 d_repr: int = 64,

                 ):
        super().__init__()
        if isinstance(encoder, DictConfig):
            self.encoder: MaliceEncoder = instantiate(encoder,
                                                      input_channels=input_channels,
                                                      d_repr=d_repr,
                                                      _recursive_=False)
        else:
            self.encoder = encoder

        if isinstance(decoder, DictConfig):
            self.decoder: MaliceDecoder = instantiate(decoder,
                                                      input_channels=input_channels,
                                                      d_repr=d_repr,
                                                      _recursive_=False)
        else:
            self.decoder = decoder

    def forward(self, batch):
        repr, out_emb, mm_embedding = self.encoder(batch)
        reconstructions = self.decoder(batch, mm_embedding)
        return OutMMAliseF(repr=repr, rec=reconstructions, emb=out_emb)


class MaliceEncoder(nn.Module):
    def __init__(self,
                 encodeur_s1,
                 encodeur_s2,
                 common_temp_proj,
                 projector: DictConfig | ProjectorTemplate = None,
                 input_channels: dict[str, int] = {"s2": 10, "s1": 3},
                 d_repr: int = 64,
                 ):
        super().__init__()

        if isinstance(encodeur_s1, DictConfig):
            print("load here")
            self.encodeur_s1: CleanUBarnReprEncoder = instantiate(
                encodeur_s1,
                d_model=d_repr,
                input_channels=input_channels["s1"],
                _recursive_=False

            )
        else:
            self.encodeur_s1 = encodeur_s1
        if isinstance(encodeur_s2, CleanUBarnReprEncoder):
            self.encodeur_s2 = encodeur_s2
        else:
            self.encodeur_s2: CleanUBarnReprEncoder = instantiate(
                encodeur_s2,
                d_model=d_repr,
                input_channels=input_channels["s2"],
                _recursive_=False

            )

        if isinstance(common_temp_proj, TemporalProjector):
            self.common_temp_proj = common_temp_proj
        else:
            self.common_temp_proj: TemporalProjector = instantiate(
                common_temp_proj, input_channels=self.encodeur_s2.ubarn.d_model
            )

        if isinstance(projector, DictConfig):
            self.projector_emb = instantiate(projector, input_channels=d_repr)
        elif (projector is None) or (self.w_inv == 0):
            self.projector_emb = IdentityProj()
        else:
            self.projector_emb = projector

    def compute_mm_embeddings(self, aligned_repr: OutTempProjForward, h: int, w: int):
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

        out = self.encodeur_s1.forward_keep_input_dim(merged_views) if "1" in sat \
            else self.encodeur_s2.forward_keep_input_dim(merged_views)

        mask_tp = repeat(~merged_views.padd_index.bool(), "b t -> b t h w", h=h, w=w)
        my_logger.debug(f"{sat} repr {out.repr.shape}")
        if isinstance(
                self.encodeur_s1.ubarn.temporal_encoder, nn.TransformerEncoderLayer
        ):
            padd = None
        else:
            padd = rearrange(mask_tp, "b t h w -> (b h w) t ")
        return out, padd

    def forward(self, batch: BatchMMSits):
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
    def __init__(self, decodeur,
                 pe_config: DictConfig | PositionalEncoder,
                 query_s1s2_d: int = 64,
                 pe_channels: int = 64,
                 input_channels: dict[str, int] = {"s2": 10, "s1": 3},
                 d_repr: int = 64,
                 ):
        super().__init__()

        if isinstance(decodeur, DictConfig):
            self.meta_decodeur = instantiate(
                decodeur,
                input_channels=d_repr,
                d_q_in=(query_s1s2_d + pe_channels),
                _recursive_=False,
            )
            # print(pe_channels+query_s1s2_d)
        else:
            self.meta_decodeur: MetaDecoder = decodeur

        self.query_builder = TempMetaQuery(
            pe_config=pe_config, input_channels=pe_channels
        )

        self.q_decod_s1 = self.define_query_decoder(query_s1s2_d)
        self.q_decod_s2 = self.define_query_decoder(query_s1s2_d)

        self.proj_s1 = torch.nn.Linear(
            self.meta_decodeur.input_channels, input_channels["s1"]
        )

        self.proj_s2 = torch.nn.Linear(
            self.meta_decodeur.input_channels, input_channels["s2"]
        )

    def compute_query(self, batch_sits: BatchOneMod, sat: str = "s2"):
        """Compute query and reshape it"""
        query = repeat(
            self.query_builder(self.q_decod_s2 if sat == "s1" else self.q_decod_s1, batch_sits.input_doy),
            "b t c -> b t c h w",
            h=batch_sits.h,
            w=batch_sits.w,
        )
        return rearrange(
            query,
            "b t (nh c )h w -> nh (b h w) t c",
            nh=self.meta_decodeur.num_heads,
        )

    @staticmethod
    def define_query_decoder(query):
        q_decod = nn.Parameter(
            torch.zeros(query)
        ).requires_grad_(True)
        nn.init.normal_(
            q_decod, mean=0, std=np.sqrt(2.0 / (query))
        )  # TODO check that
        return q_decod

    def forward(self, batch: BatchMMSits, mm_embedding: torch.Tensor) -> Rec:
        # Decoder!

        query_s2a = self.compute_query(batch_sits=batch.sits2a, sat="s2")
        query_s2b = self.compute_query(batch_sits=batch.sits2b, sat="s2")
        query_s1a = self.compute_query(batch_sits=batch.sits1a, sat="s1")
        query_s1b = self.compute_query(batch_sits=batch.sits1b, sat="s1")

        my_logger.debug(f"query decodeur s2 {query_s2b.shape}")

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

        out = self.meta_decodeur.forward(
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
