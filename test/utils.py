import torch

from mmmv_ssl.data.dataclass import BatchOneMod, BatchVicReg
from mmmv_ssl.model.config import ProjectorConfig, UTAEConfig
from mmmv_ssl.module.instantiate import instantiate_vicregssl_module
from mmmv_ssl.train.config import AdamConfig, CAWConfig, VicRegTrainConfig


def default_lvicregmodule(d_in):
    d_emb = 32
    d_model = 64
    default_proj_conf = ProjectorConfig(l_dim=[64, 64])
    default_train_config = VicRegTrainConfig(
        scheduler=CAWConfig(), optimizer=AdamConfig(), lr=0.001
    )
    return instantiate_vicregssl_module(
        in_channels=d_in,
        projector_local=default_proj_conf,
        projector_bottleneck=default_proj_conf,
        train_config=default_train_config,
        d_emb=d_emb,
        d_model=d_model,
        utae=UTAEConfig(),
        stats=None,
    )


def create_fake_batch(b=2, t=10, c=3, h=8, w=8) -> BatchVicReg:
    t2 = 20
    batch_1 = BatchOneMod(
        sits=torch.randn(b, t, c, h, w),
        doy=torch.arange(t)[None, :].expand(b, -1),
        padd_mask=torch.zeros(b, t).bool(),
    )
    batch_2 = BatchOneMod(
        sits=torch.randn(b, t2, c, h, w),
        doy=torch.arange(t2)[None, :].expand(b, -1),
        padd_mask=torch.zeros(b, t2).bool(),
    )
    return BatchVicReg(sits1=batch_1, sits2=batch_2)
