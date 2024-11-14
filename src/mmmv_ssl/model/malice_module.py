import logging

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
from mmmv_ssl.model.query_utils import TempMetaQuery

my_logger = logging.getLogger(__name__)

class AliseMMModule(nn.Module):
    def __init__(self, encodeur_s1, encodeur_s2, input_channels, common_temp_proj, decodeur,
                 pe_config: DictConfig | PositionalEncoder,
                 d_repr: int = 64,
                 query_s1s2_d: int = 64,
                 pe_channels: int = 64,
                 ):
        super().__init__()

        if isinstance(encodeur_s1, DictConfig):
            print("load here")
            self.encodeur_s1: CleanUBarnReprEncoder = instantiate(
                encodeur_s1,
                d_model=d_repr,
                input_channels=input_channels.s1_channels,
                _recursive_=False,
            )
        else:
            self.encodeur_s1 = encodeur_s1
        if isinstance(encodeur_s2, CleanUBarnReprEncoder):
            self.encodeur_s2 = encodeur_s2
        else:
            self.encodeur_s2: CleanUBarnReprEncoder = instantiate(
                encodeur_s2,
                d_model=d_repr,
                input_channels=input_channels.s2_channels,
                _recursive_=False,
            )

        self.common_temp_proj = common_temp_proj

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

        self.q_decod_s1 = self.define_query_decoder(query_s1s2_d)
        self.q_decod_s2 = self.define_query_decoder(query_s1s2_d)

    def forward(self, batch):
        self.encode()
        self.decode()

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

    def encode(self, batch):
        out_s1, padd_s1 = self.encode_views(batch, sat="s1")
        out_s2, padd_s2 = self.encode_views(batch, sat="s2")

        aligned_repr: OutTempProjForward = self.common_temp_proj(
            sits_s1=rearrange(out_s1.repr, "b t c h w -> (b h w ) t c"),
            padd_s1=padd_s1,
            sits_s2=rearrange(out_s2.repr, "b t c h w -> (b h w) t c"),
            padd_s2=padd_s2,
        )

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

        repr, out_emb, mm_embedding = self.compute_mm_embeddings(aligned_repr, batch.sits1a.h, batch.sits1a.w)

    def decode(self):