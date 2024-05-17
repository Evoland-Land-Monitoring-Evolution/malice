import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from lightning import LightningModule
from mt_ssl.model.ubarn_repr_encoder import UBarnReprEncoder
from mt_ssl.module.template import TemplateModule
from omegaconf import DictConfig
from openeo_mmdc.dataset.dataclass import Stats

from mmmv_ssl.data.dataclass import BatchMMSits, MMChannels, merge2views
from mmmv_ssl.model.dataclass import OutTempProjForward
from mmmv_ssl.model.decodeur import MetaDecoder
from mmmv_ssl.model.encoding import PositionalEncoder
from mmmv_ssl.model.query_utils import TempMetaQuery
from mmmv_ssl.model.temp_proj import TemporalProjector
from mmmv_ssl.module.dataclass import LatRepr, OutMMAliseF, Rec, RecWithOrigin

my_logger = logging.getLogger(__name__)


class AliseMM(TemplateModule, LightningModule):
    def __init__(
        self,
        encodeur_s1: UBarnReprEncoder | DictConfig,
        encodeur_s2: UBarnReprEncoder | DictConfig,
        common_temp_proj: nn.Module,
        decodeur: DictConfig | MetaDecoder,
        train_config,
        input_channels: MMChannels,
        pe_config: DictConfig | PositionalEncoder,
        stats: None | Stats = None,
        d_repr: int = 64,
        query_s1s2_d: int = 64,
        pe_channels: int = 64,
    ):
        super().__init__(train_config)

        self.d_repr = d_repr
        self.stats = stats
        self.input_channels = input_channels
        self.query_builder = TempMetaQuery(
            pe_config=pe_config, input_channels=pe_channels
        )
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

        self.common_temp_proj = common_temp_proj
        if isinstance(encodeur_s1, DictConfig):
            print("load here")
            self.encodeur_s1: UBarnReprEncoder = instantiate(
                encodeur_s1,
                d_model=d_repr,
                input_channels=input_channels.s1_channels,
                _recursive_=False,
            )
        else:
            self.encodeur_s1 = encodeur_s1
        if isinstance(encodeur_s2, UBarnReprEncoder):
            self.encodeur_s2 = encodeur_s2
        else:
            self.encodeur_s2: UBarnReprEncoder = instantiate(
                encodeur_s2,
                d_model=d_repr,
                input_channels=input_channels.s2_channels,
                _recursive_=False,
            )
        if isinstance(common_temp_proj, TemporalProjector):
            self.common_temp_proj = common_temp_proj
        else:
            self.common_temp_proj: TemporalProjector = instantiate(
                common_temp_proj, input_channels=self.encodeur_s2.ubarn.d_model
            )
        self.proj_s1 = torch.nn.Linear(
            self.meta_decodeur.input_channels, input_channels.s1_channels
        )
        self.proj_s2 = torch.nn.Linear(
            self.meta_decodeur.input_channels, input_channels.s2_channels
        )
        self.query_s1s2_d = query_s1s2_d
        assert (
            query_s1s2_d + pe_channels
        ) % self.meta_decodeur.num_heads == 0, (
            f"decoder query shape : {query_s1s2_d + pe_channels} decodeur"
            f" heads {self.meta_decodeur.num_heads}"
        )
        self.q_decod_s1 = nn.Parameter(
            torch.zeros(query_s1s2_d)
        ).requires_grad_(True)
        nn.init.normal_(
            self.q_decod_s1, mean=0, std=np.sqrt(2.0 / (query_s1s2_d))
        )  # TODO check that
        self.q_decod_s2 = nn.Parameter(
            torch.zeros(query_s1s2_d)
        ).requires_grad_(
            True
        )  # self.meta_decodeur.num_heads,
        nn.init.normal_(
            self.q_decod_s2, mean=0, std=np.sqrt(2.0 / (query_s1s2_d))
        )  # TODO check that

    def forward(self, batch: BatchMMSits) -> OutMMAliseF:
        s1 = merge2views(batch.sits1a, batch.sits1b)
        s2 = merge2views(batch.sits2a, batch.sits2b)
        out_s1 = self.encodeur_s1.forward_keep_input_dim(s1)
        out_s2 = self.encodeur_s2.forward_keep_input_dim(s2)
        # mm_out=torch.cat([out_s1.repr,out_s2.repr],dim=2)
        embedding: OutTempProjForward = self.common_temp_proj(
            sits_s1=rearrange(out_s1.repr, "b t c h w -> (b h w ) t c"),
            padd_s1=s1.padd_index,
            sits_s2=rearrange(out_s2.repr, "b t c h w -> (b h w) t c"),
            padd_s2=s2.padd_index,
        )
        query_s2a = repeat(
            self.query_builder(self.q_decod_s2, batch.sits2a.input_doy),
            "b t c -> b t c h w",
            h=batch.sits2a.h,
            w=batch.sits2a.w,
        )  # used to decode s2a dates from s1a or from s2b
        query_s2b = repeat(
            self.query_builder(self.q_decod_s2, batch.sits2b.input_doy),
            "b t c -> b t c h w",
            h=batch.sits2a.h,
            w=batch.sits2a.w,
        )  # used to decode s2b dates from s1b or from s2a
        my_logger.debug(f"query decodeur s2 {query_s2b.shape}")
        query_s1a = repeat(
            self.query_builder(self.q_decod_s1, batch.sits1a.input_doy),
            "b t c -> b t c h w",
            h=batch.sits1a.h,
            w=batch.sits1a.w,
        )  # used to decode s1a dates from s2a and s1b embeddings
        query_s1b = repeat(
            self.query_builder(self.q_decod_s1, batch.sits1b.input_doy),
            "b t c -> b t c h w",
            h=batch.sits1b.h,
            w=batch.sits1b.w,
        )  # used to decode s1b dates from s2b and s1a embeddings

        mm_queries = torch.cat(
            [
                rearrange(
                    query_s2b,
                    "b t (nh c) h w ->nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
                rearrange(
                    query_s1a,
                    "b t (nh c )h w -> nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
                rearrange(
                    query_s2a,
                    "b t (nh c) h w ->nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
                rearrange(
                    query_s1b,
                    "b t (nh c )h w -> nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
                rearrange(
                    query_s1b,
                    "b t (nh c )h w -> nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
                rearrange(
                    query_s2a,
                    "b t (nh c) h w ->nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
                rearrange(
                    query_s1a,
                    "b t (nh c )h w -> nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
                rearrange(
                    query_s2b,
                    "b t (nh c) h w ->nh (b h w) t c",
                    nh=self.meta_decodeur.num_heads,
                ),
            ],
            dim=1,
        )
        #
        # mm_queries = torch.cat(
        #     [
        #         rearrange(
        #             query_s1a,
        #             "b t (nh c )h w -> nh (b h w) t c",
        #             nh=self.meta_decodeur.num_heads,
        #         ),
        #         rearrange(
        #             query_s1b,
        #             "b t (nh c )h w -> nh (b h w) t c",
        #             nh=self.meta_decodeur.num_heads,
        #         ),
        #         rearrange(
        #             query_s2a,
        #             "b t (nh c) h w ->nh (b h w) t c",
        #             nh=self.meta_decodeur.num_heads,
        #         ),
        #         rearrange(
        #             query_s2a,
        #             "b t (nh c) h w ->nh (b h w) t c",
        #             nh=self.meta_decodeur.num_heads,
        #         ),
        #     ],
        #     dim=1,
        # )  # concat across batch dimension? each modalities treated independently be decoder
        embedding_s1 = rearrange(
            embedding.s1, "(view bhw) t c-> view bhw t c", view=2
        )
        embedding_s2 = rearrange(
            embedding.s2, "(view bhw) t c-> view bhw t c", view=2
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
        my_logger.debug(f"queries{mm_queries.shape}")
        h, w = batch.sits2a.h, batch.sits2a.w
        padd_mm = torch.cat(
            [
                repeat(batch.sits2a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits2a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits2b.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits2b.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1a.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1b.padd_index, "b t -> (b h w) t ", h=h, w=w),
                repeat(batch.sits1b.padd_index, "b t -> (b h w) t ", h=h, w=w),
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
        repr = LatRepr(
            s1a=rearrange(
                embedding_s1[0, ...], "(b h w) t c-> b t c h w", h=h, w=w
            ),
            s1b=rearrange(
                embedding_s1[1, ...], "(b h w) t c -> b t c h w", h=h, w=w
            ),
            s2a=rearrange(
                embedding_s2[0, ...], "(b h w) t c -> b t c h w", h=h, w=w
            ),
            s2b=rearrange(
                embedding_s2[1, ...], "(b h w) t c -> b t c h w", h=h, w=w
            ),
        )
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

        return OutMMAliseF(repr=repr, rec=rec)
