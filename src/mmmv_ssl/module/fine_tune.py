import logging
from pathlib import Path
from typing import Literal

import torch
from einops import rearrange
from hydra.utils import instantiate
from mt_ssl.data.classif_class import ClassifBInput
from mt_ssl.data.mt_batch import BOutputReprEnco
from mt_ssl.hydra_config.model import ConfigDecoder, ConfigShallowClassifier
from mt_ssl.metrics.classif_metrics import init_classif_metrics
from mt_ssl.model.convolutionalblock import LastBlock
from mt_ssl.model.mask_repr_encoder import BERTReprEncoder
from mt_ssl.model.shallow_classifier import MasterQueryDecoding
from mt_ssl.module.dataclass import OutFTForward, OutFTSharedStep
from mt_ssl.module.template import TemplateClassifModule
from mt_ssl.module.template_ft_module import FTParams
from omegaconf import DictConfig
from openeo_mmdc.dataset.dataclass import Stats
from openeo_mmdc.dataset.padding import apply_padding
from openeo_mmdc.dataset.to_tensor import load_transform_one_mod
from torch import nn
from torchmetrics import MetricCollection

from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
from mmmv_ssl.model.utils import build_encoder
from mmmv_ssl.module.alise_mm import AliseMM, load_malice

my_logger = logging.getLogger(__name__)


class FineTuneOneMod(TemplateClassifModule):
    def __init__(
        self,
        train_config,
        ft_params: FTParams | dict,
        input_channels: int,
        num_classes: int,
        decoder_config: DictConfig | ConfigDecoder | ConfigShallowClassifier,
        mod: Literal["s1", "s2"] = "s2",
        decoder_type: Literal["linear", "master_query"] = "linear",
        stats: Stats | None = None,
    ):
        super().__init__(train_config, input_channels, num_classes, stats)
        if isinstance(ft_params, dict):
            ft_params = FTParams(**ft_params)
        pretrained_module: AliseMM = load_malice(
            pl_module=ft_params.pl_module,
            path_ckpt=ft_params.ckpt_path,
        )

        self.freeze_repr_encoder = ft_params.freeze_representation_encoder
        if self.freeze_repr_encoder:
            pretrained_module.freeze()
            self.repr_encoder: MonoSITSEncoder = build_encoder(
                pretrained_module=pretrained_module, mod=mod
            )
            self.repr_encoder.eval()
        else:
            self.repr_encoder = build_encoder(
                pretrained_module=pretrained_module, mod=mod
            )
            self.repr_encoder.train()
        self.d_model=self.repr_encoder.d_model
        if mod == "s1":
            self.stats = pretrained_module.stats[1]
        elif mod == "s2":
            self.stats = pretrained_module.stats[0]
        else:
            raise NotImplementedError
        if isinstance(decoder_config, nn.Module):
            self.shallow_classifier = decoder_config
        elif isinstance(
            decoder_config,
            DictConfig | ConfigDecoder | ConfigShallowClassifier,
        ):
            len_ref_time_points = self.repr_encoder.nq
            if decoder_type == "linear":
                my_logger.info(
                    "Linear model {self.d_model * len_ref_time_points}"
                )
                self.shallow_classifier: LastBlock | MasterQueryDecoding = (
                    instantiate(
                        decoder_config,
                        inplanes=self.d_model * len_ref_time_points,
                        planes=num_classes,
                    )
                )
            elif decoder_type == "master_query":
                self.shallow_classifier: LastBlock | MasterQueryDecoding = (
                    instantiate(
                        decoder_config,
                        inplanes=self.d_model,
                        planes=num_classes,
                    )
                )
            else:
                raise NotImplementedError
        my_logger.info(
            f"The shallow classifier is : {type(self.shallow_classifier)}"
        )
        metrics = MetricCollection(init_classif_metrics(num_classes))
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.save_hyperparameters(ignore=["ft_params"])
        self.ft_params = ft_params

    def forward(
        self, batch: ClassifBInput, return_attns: bool = False
    ) -> OutFTForward:
        #
        # PATH_DIR=Path("/work/scratch/data/kalinie/results/alise_preentrained")# modify
        #
        # PATH_DATA="/work/CESBIO/projects/DeepChange/Iris/PASTIS/PT_FORMAT/S2_10000.pt"
        # PATH_CSV=PATH_DIR.joinpath("data") #modify
        # data=torch.load(PATH_DATA)
        # transform=load_transform_one_mod(PATH_CSV,mod="s2").transform
        #
        # sits_s2 = rearrange(data.sits, "c t h w-> t c h w")
        # doy = data.doy
        #
        # sits_s2_, doy_, padd_index_ = apply_padding(allow_padd=True, max_len=61, t=sits_s2.shape[0], sits=sits_s2, doy=doy)
        #
        # norm_s2_=rearrange(transform(rearrange(sits_s2_,'t c h w -> c t h w')),'c t h w -> 1 t c h w ')[:, :, :, 32:-32, 32:-32]
        #
        # from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
        # from mt_ssl.data.classif_class import ClassifBInput
        #
        # DEVICE = 'cuda'
        #
        # input_12 = ClassifBInput(sits=norm_s2_.to(DEVICE), input_doy=doy_[None, :].to(DEVICE),
        #                   padd_index=padd_index_[None, :].to(DEVICE),
        #                   mask=None, labels=None)
        #
        # output_torch_file = "/work/scratch/data/kalinie/results/alise_preentrained/malice_new_s2.pth"
        #
        # repr = torch.load(output_torch_file)
        # repr.eval()
        # out2 = repr.forward_keep_input_dim(input_12).repr
        # if self.freeze_repr_encoder:
        #     self.repr_encoder.eval()
        #     with torch.no_grad():
        #         out_repr = self.forward_representation_encoder(
        #             input_12, return_attns=return_attns
        #         )
        # torch.set_printoptions(precision=20)
        #
        # print(out_repr.repr)
        # print(out_repr.repr[-1, -1, -1, -1, -10])
        # exit()
        if self.freeze_repr_encoder:
            self.repr_encoder.eval()
            with torch.no_grad():
                out_repr = self.forward_representation_encoder(
                    batch, return_attns=return_attns
                )
        else:
            out_repr = self.forward_representation_encoder(
                batch, return_attns=return_attns
            )

        if isinstance(self.shallow_classifier, MasterQueryDecoding):
            if isinstance(self.repr_encoder, BERTReprEncoder):
                padd_index = batch.padd_index
            else:
                padd_index = None
            out_class = self.shallow_classifier(
                out_repr.repr, key_padding_mask=padd_index
            )  # out dim is b h w nc
        else:
            repr = rearrange(
                out_repr.repr, " b k c h w -> b h w (k c)"
            )  # features are fetures and dates
            out_class = self.shallow_classifier(repr)  # out dim is b h w nc

        return OutFTForward(out_class, out_repr.repr)

    def forward_representation_encoder(
        self, batch: ClassifBInput, return_attns: bool = False
    ) -> BOutputReprEnco:
        input_sits = batch.sits
        my_logger.debug(batch.padd_index.shape)
        my_logger.debug(input_sits.shape)
        # print("begin {} {}".format(input_sits.shape, input_doy))

        out_repr = self.repr_encoder.forward_keep_input_dim(batch)

        return out_repr

    def validation_step(
        self, batch, batch_idx
    ) -> tuple[dict, OutFTSharedStep] | None:
        out_shared_step = self.shared_step(
            batch, batch_idx, loss_fun=self.train_loss
        )
        loss = out_shared_step.loss
        if not torch.isnan(out_shared_step.loss):
            metrics={}
            metrics["val_loss"] = loss.item()
                        
            self.log_dict(
                metrics,
                on_epoch=True,
                on_step=True,
                batch_size=self.bs,
                prog_bar=True,
            )
            r_pred = out_shared_step.pred_flatten.detach()
            r_trg = out_shared_step.trg_flatten.detach()
            my_loss = loss.item()
            self.val_metrics.update(r_pred, r_trg)
            # my_logger.info("Metrics in test {}".format(metrics))
        else:
            my_loss = None
        out_shared_step.loss = my_loss
        return my_loss, out_shared_step

    def on_validation_epoch_end(self) -> None:
        outputs = self.val_metrics.compute()
        outputs = {
            k: v.to(device="cpu", non_blocking=True)
            for k, v in outputs.items()
        }
        self.save_test_metrics = outputs

    def training_step(self, batch: ClassifBInput, batch_idx):
        # my_logger.info("train step start {}".format(torch.cuda.memory_allocated()))
        out_shared_step = self.shared_step(
            batch, batch_idx, loss_fun=self.train_loss
        )
        loss = out_shared_step.loss
        metrics={}
        if not torch.isnan(out_shared_step.loss):

            metrics["train_loss"] = loss.item()

            self.log_dict(
                metrics,
                on_epoch=True,
                on_step=True,
                batch_size=self.bs,
                prog_bar=True,
            )
            # my_logger.info("train step end {}".format(torch.cuda.memory_allocated()))
            return loss

        return None
