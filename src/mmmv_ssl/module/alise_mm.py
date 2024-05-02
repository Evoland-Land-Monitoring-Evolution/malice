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

from mmmv_ssl.data.dataclass import BatchMMSits, MMChannels
from mmmv_ssl.model.dataclass import OutTempProjForward
from mmmv_ssl.model.decodeur import MetaDecoder
from mmmv_ssl.model.encoding import PositionalEncoder
from mmmv_ssl.model.query_utils import TempMetaQuery
from mmmv_ssl.model.temp_proj import TemporalProjector
from mmmv_ssl.module.dataclass import OutMMAliseF


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
        query_s2_d: int = 64,
        query_s1_d: int = 64,
        pe_channels: int = 64,
    ):
        super().__init__(train_config)
        self.query_s1_d = query_s1_d
        self.query_s2_d = query_s2_d
        self.q_decod_s1 = nn.Parameter(torch.zeros(query_s1_d)).requires_grad_(
            True
        )
        nn.init.normal_(
            self.q_decod_s1, mean=0, std=np.sqrt(2.0 / (query_s1_d))
        )  # TODO check that
        self.q_decod_s2 = nn.Parameter(torch.zeros(query_s2_d)).requires_grad_(
            True
        )
        nn.init.normal_(
            self.q_decod_s2, mean=0, std=np.sqrt(2.0 / (query_s2_d))
        )  # TODO check that
        self.query_builder = TempMetaQuery(
            pe_config=pe_config, input_channels=pe_channels
        )
        self.d_repr = d_repr
        self.stats = stats
        self.input_channels = input_channels
        if isinstance(decodeur, DictConfig):
            self.meta_decodeur = instantiate(decodeur, input_channels=d_repr)
        else:
            self.meta_decodeur: MetaDecoder = decodeur

        self.common_temp_proj = common_temp_proj
        if isinstance(encodeur_s1, UBarnReprEncoder):
            self.encodeur_s1 = encodeur_s1
        else:
            self.encodeur_s1: UBarnReprEncoder = instantiate(
                encodeur_s1,
                input_channels=input_channels.s1_channels,
                d_model=d_repr,
            )
        if isinstance(encodeur_s2, UBarnReprEncoder):
            self.encodeur_s2 = encodeur_s2
        else:
            self.encodeur_s2: UBarnReprEncoder = instantiate(
                encodeur_s2,
                input_channels=input_channels.s2_channels,
                d_model=d_repr,
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

    def forward(self, batch: BatchMMSits) -> OutMMAliseF:
        out_s1 = self.encodeur_s1.forward_keep_input_dim(batch.sits1)
        out_s2 = self.encodeur_s2.forward_keep_input_dim(batch.sits2)
        # mm_out=torch.cat([out_s1.repr,out_s2.repr],dim=2)
        embedding: OutTempProjForward = self.common_temp_proj(
            sits_1=rearrange(out_s1.repr, "b t c h w -> (b h w ) t c"),
            padd_s1=batch.sits1.padd_index,
            sits_s2=rearrange(out_s2.repr, "b t c h w -> (b h w) t c"),
            padd_s2=batch.sits2.padd_index,
        )
        query_s2 = repeat(
            self.query_builder(self.q_decod_s2, batch.sits2.input_doy),
            "b t c -> b t c h w",
            h=batch.sits2.h,
            w=batch.sits2.w,
        )  # used to decode s2 dates from s1 embeddings
        query_s1 = repeat(
            self.query_builder(self.q_decod_s1, batch.sits1.input_doy),
            "b t c -> b t c h w",
            h=batch.sits1.h,
            w=batch.sits1.w,
        )  # used to decode s1 dates from s2 embeddings
        mm_queries = torch.cat(
            [
                rearrange(query_s1, "b t c h w -> (b h w) t c"),
                rearrange(query_s2, "b t c h w -> (b h w) t c"),
            ],
            dim=0,
        )  # concat across batch dimension? each modalities treated independently be decoder
        mm_embedding = torch.cat(
            [embedding.s2, embedding.s1], dim=0
        )  # 2(b h w) t d

        padd_mm = torch.cat(
            [
                repeat(
                    batch.sits1.padd_index,
                    "b t -> (b h w) t ",
                    h=batch.sits1.h,
                    w=batch.sits1.w,
                ),
                repeat(
                    batch.sits2.padd_index,
                    "b t -> (b h w) t ",
                    h=batch.sits2.h,
                    w=batch.sits2.w,
                ),
            ],
            dim=0,
        )  # 2(b h w) t

        out = self.meta_decodeur.forward(
            mm_sits=mm_embedding, padd_mm=padd_mm, mm_queries=mm_queries
        )  # (2bhw) t d
        out = rearrange(
            out,
            "(mod b h w) t c-> mod b t h w c",
            mod=2,
            b=batch.sits1.b,
            h=batch.sits1.h,
            w=batch.sits1.w,
        )
        s1_rec = self.proj_s1(out[0, ...])
        s2_rec = self.proj_s2(out[1, ...])
        return OutMMAliseF(
            repr_s1=rearrange(
                embedding.s1,
                "(b h w) t c -> b t c h w",
                h=batch.sits1.h,
                w=batch.sits1.w,
            ),
            repr_s2=rearrange(
                embedding.s2,
                "(b h w) t c -> b t c h w",
                h=batch.sits2.h,
                w=batch.sits2.w,
            ),
            pred_s1=rearrange(s1_rec, "b t h w c -> b t c h w"),
            pred_s2=rearrange(s2_rec, "b t h w c -> b t c h w"),
        )
