import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from lightning import LightningModule
from mt_ssl.data.mt_batch import BOutputReprEnco
from mt_ssl.module.loss import create_mask_loss
from mt_ssl.module.template import TemplateModule
from omegaconf import DictConfig
from openeo_mmdc.dataset.dataclass import Stats

from mmmv_ssl.data.dataclass import BatchMMSits, MMChannels, merge2views, BatchOneMod
from mmmv_ssl.model.clean_ubarn_repr_encoder import CleanUBarnReprEncoder
from mmmv_ssl.model.dataclass import OutTempProjForward
from mmmv_ssl.model.decodeur import MetaDecoder
from mmmv_ssl.model.encoding import PositionalEncoder
from mmmv_ssl.model.projector import IdentityProj, ProjectorTemplate
from mmmv_ssl.model.query_utils import TempMetaQuery
from mmmv_ssl.model.temp_proj import TemporalProjector
from mmmv_ssl.module.dataclass import (
    DespeckleS1,
    LatRepr,
    OutMMAliseF,
    OutMMAliseSharedStep,
    Rec,
    RecWithOrigin,
)
from mmmv_ssl.module.loss import GlobalInvRecMMLoss, OneViewRecL, TotalRecLoss
from mmmv_ssl.utils.speckle_filter import despeckle_batch

my_logger = logging.getLogger(__name__)


def despeckle(batch_sits: BatchOneMod) -> tuple[torch.Tensor, int]:
    b, t, _, h, w = batch_sits.sits.shape
    despeckle_s1, margin = despeckle_batch(
        rearrange(batch_sits.sits, "b t c h w -> (b t ) c h w")
    )
    despeckle_s1 = rearrange(
        despeckle_s1, "(b t ) c h w -> b t c h w", b=b
    )[
                   ...,
                   margin: h - margin,
                   margin: w - margin,
                   ]
    return despeckle_s1, margin


class AliseMM(TemplateModule, LightningModule):
    def __init__(
            self,
            encodeur_s1: CleanUBarnReprEncoder | DictConfig,
            encodeur_s2: CleanUBarnReprEncoder | DictConfig,
            common_temp_proj: nn.Module,
            decodeur: DictConfig | MetaDecoder,
            train_config,
            input_channels: MMChannels,
            pe_config: DictConfig | PositionalEncoder,
            stats: None | tuple[Stats, Stats] = None,
            d_repr: int = 64,
            query_s1s2_d: int = 64,
            pe_channels: int = 64,
            projector: DictConfig | ProjectorTemplate = None,
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
        self.inv_loss = torch.nn.MSELoss()
        self.rec_loss = torch.nn.MSELoss()
        self.w_inv = train_config.w_inv
        self.w_rec = train_config.w_rec
        self.w_crossrec = train_config.w_crossrec
        if isinstance(projector, DictConfig):
            self.projector_emb = instantiate(projector, input_channels=d_repr)
        elif (projector is None) or (self.w_inv == 0):
            self.projector_emb = IdentityProj()
        else:
            self.projector_emb = projector
        self.save_hyperparameters()

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

    def decoder(self, batch, mm_embedding: torch.Tensor, padd_mm: torch.Tensor, mm_queries: torch.Tensor) -> Rec:
        # Decoder!
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

    def forward(self, batch: BatchMMSits) -> OutMMAliseF:
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

        reconstructions = self.decoder(batch, mm_embedding, padd_mm, mm_queries)

        return OutMMAliseF(repr=repr, rec=reconstructions, emb=out_emb)

    def shared_step(self, batch: BatchMMSits) -> OutMMAliseSharedStep:
        out_model = self.forward(batch)
        assert isinstance(batch, BatchMMSits)
        tot_rec_loss, despeckle_s1 = self.compute_rec_loss(
            batch=batch, rec=out_model.rec
        )

        inv_loss = self.invariance_loss(out_model.emb)
        if tot_rec_loss is None:
            global_loss = None
        else:
            global_loss = GlobalInvRecMMLoss(
                total_rec_loss=tot_rec_loss,
                inv_loss=inv_loss,
                w_rec=self.w_rec,
                w_inv=self.w_inv,
                w_crossrec=self.w_crossrec,
            )
        return OutMMAliseSharedStep(
            loss=global_loss, out_forward=out_model, despeckle_s1=despeckle_s1
        )

    def training_step(self, batch: BatchMMSits, batch_idx: int):
        out_shared_step = self.shared_step(batch)
        if out_shared_step.loss is None:
            return None

        self.log_dict(
            out_shared_step.loss.to_dict(suffix="train"),
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
        )

        # self.log(
        #     f"train/{self.losses_list[0]}",
        #     out_shared_step.loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=False,
        # )

        return out_shared_step.loss.compute()

    def validation_step(self, batch: BatchMMSits, batch_idx: int):
        out_shared_step = self.shared_step(batch)

        self.log_dict(
            out_shared_step.loss.to_dict(suffix="val"),
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
            sync_dist=False,
        )
        return out_shared_step

    def compute_rec_loss(
            self, batch: BatchMMSits, rec: Rec
    ) -> tuple[TotalRecLoss, DespeckleS1] | tuple[None, None]:
        assert isinstance(batch, BatchMMSits), f"batch is {batch}"

        despeckle_s1a, margin = despeckle(batch.sits1a)
        despeckle_s1b, _ = despeckle(batch.sits1b)

        s1a_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s1a,
                                                 sits=batch.sits1a,
                                                 original_sits=despeckle_s1a,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a, s1b=despeckle_s1b),
                                                 margin=margin)

        s1b_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s1b,
                                                 sits=batch.sits1b,
                                                 original_sits=despeckle_s1b,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a, s1b=despeckle_s1b),
                                                 margin=margin)

        s2a_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s2a,
                                                 sits=batch.sits2a,
                                                 original_sits=batch.sits2a.sits,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a, s1b=despeckle_s1b))

        s2b_rec_loss = self.compute_one_rec_loss(rec_sits=rec.s2b,
                                                 sits=batch.sits1b,
                                                 original_sits=batch.sits2b.sits,
                                                 return_desp=DespeckleS1(s1a=despeckle_s1a, s1b=despeckle_s1b))

        return TotalRecLoss(
            s1_a=s1a_rec_loss,
            s1_b=s1b_rec_loss,
            s2_a=s2a_rec_loss,
            s2_b=s2b_rec_loss,
        ), DespeckleS1(s1a=despeckle_s1a, s1b=despeckle_s1b)

    def invariance_loss(self, embeddings: LatRepr):
        creca = self.inv_loss(embeddings.s1a, embeddings.s2a)
        crecb = self.inv_loss(embeddings.s1b, embeddings.s2b)
        return 1 / 2 * (crecb + creca)

    def load_weights(self, path_ckpt, strict=True):
        my_logger.info(f"We load state dict  from {path_ckpt}")
        if not torch.cuda.is_available():
            map_params = {"map_location": "cpu"}
        else:
            map_params = {}
        ckpt = torch.load(path_ckpt, **map_params)
        self.load_state_dict(ckpt["state_dict"], strict=strict)

    def compute_one_rec_loss(self, rec_sits: RecWithOrigin,
                             sits: BatchOneMod,
                             original_sits: torch.Tensor,
                             return_desp: DespeckleS1 | None = None, margin: int = 0
                             ) -> OneViewRecL | tuple[None, DespeckleS1]:
        valid_mask = create_mask_loss(
            sits.padd_index, ~sits.mask
        )  # in .mask True means pixel valid
        h, w = sits.h, sits.w
        if torch.sum(valid_mask) != 0:
            valid_mask = valid_mask[
                         ...,
                         margin: h - margin,
                         margin: w - margin,
                         ]
            return OneViewRecL(
                monom_rec=self.rec_loss(
                    torch.masked_select(rec_sits.same_mod[
                                        ...,
                                        margin: h - margin,
                                        margin: w - margin,
                                        ],
                                        valid_mask),
                    torch.masked_select(original_sits, valid_mask),
                ),
                crossm_rec=self.rec_loss(
                    torch.masked_select(rec_sits.other_mod[
                                        ...,
                                        margin: h - margin,
                                        margin: w - margin,
                                        ],
                                        valid_mask),
                    torch.masked_select(original_sits, valid_mask),
                ),
            )
        else:
            return None, return_desp

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


def load_malice(pl_module: AliseMM, path_ckpt, params_module: DictConfig):
    if path_ckpt is not None:
        pl_module = pl_module.load_from_checkpoint(path_ckpt)

    return pl_module


def check_for_nans(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("NaNs or Infs detected in data")
