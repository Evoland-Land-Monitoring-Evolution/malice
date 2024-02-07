from hydra.utils import instantiate
from omegaconf import DictConfig
from openeo_mmdc.dataset.dataclass import Stats

from mmmv_ssl.model.config import ProjectorConfig, UTAEConfig
from mmmv_ssl.module.vicregl_ssl import LVicRegModule
from mmmv_ssl.train.config import VicRegTrainConfig


def instantiate_vicregssl_module(
    in_channels: int,
    projector_local: DictConfig | ProjectorConfig,
    projector_bottleneck: DictConfig | ProjectorConfig,
    train_config: DictConfig | VicRegTrainConfig,
    d_emb: int = 64,
    d_model: int = 64,
    utae: DictConfig | UTAEConfig = UTAEConfig(),
    stats: None | Stats = None,
) -> LVicRegModule:
    utae = instantiate(utae, in_channels=in_channels, out_channels=d_model)
    projector_local = instantiate(projector_local, d_in=d_model, d_out=d_emb)
    projector_bottleneck = instantiate(
        projector_bottleneck, d_in=utae.encoder_widths[-1], d_out=d_emb
    )
    return LVicRegModule(
        train_config=train_config,
        d_emb=d_emb,
        stats=stats,
        model=utae,
        d_model=d_model,
        projector_bottleneck=projector_bottleneck,
        projector_local=projector_local,
    )
