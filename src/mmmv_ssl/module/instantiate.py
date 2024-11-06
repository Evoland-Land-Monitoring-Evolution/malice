import logging

import torch
from hydra.utils import instantiate
from pathlib import Path
from mt_ssl.data.datamodule.pastis_datamodule import PASTISDataModule
from mt_ssl.module.template_ft_module import FTParams
from mt_ssl.utils.open import find_file, find_good_ckpt, open_yaml
from omegaconf import DictConfig
from openeo_mmdc.dataset.dataclass import Stats

from mmmv_ssl.data.datamodule.mm_datamodule import MMMaskDataModule
from mmmv_ssl.model.config import ProjectorConfig, UTAEConfig
from mmmv_ssl.model.projector import MMProjector
from mmmv_ssl.module.alise_mm import AliseMM
from mmmv_ssl.module.fine_tune import FineTuneOneMod
from mmmv_ssl.module.vicregl_ssl import LVicRegModule
from mmmv_ssl.train.config import VicRegTrainConfig

my_logger = logging.getLogger(__name__)


def instantiate_vicregssl_module(
    in_channels1: int,
    in_channels2: int,
    projector_local: DictConfig | ProjectorConfig,
    projector_bottleneck: DictConfig | ProjectorConfig,
    train_config: DictConfig | VicRegTrainConfig,
    d_emb: int = 64,
    d_model: int = 64,
    utae: DictConfig | UTAEConfig = UTAEConfig(),
    stats: None | Stats = None,
) -> LVicRegModule:
    utae1 = instantiate(utae, in_channels=in_channels1, out_channels=d_model)
    utae2 = instantiate(utae, in_channels=in_channels2, out_channels=d_model)
    projector_local = MMProjector(
        proj1=instantiate(projector_local, d_in=d_model, d_out=d_emb),
        proj2=instantiate(projector_local, d_in=d_model, d_out=d_emb),
    )
    projector_bottleneck = MMProjector(
        proj1=instantiate(
            projector_bottleneck, d_in=utae1.encoder_widths[-1], d_out=d_emb
        ),
        proj2=instantiate(
            projector_bottleneck, d_in=utae2.encoder_widths[-1], d_out=d_emb
        ),
    )
    return LVicRegModule(
        train_config=train_config,
        d_emb=d_emb,
        stats=stats,
        model1=utae1,
        model2=utae2,
        d_model=d_model,
        projector_bottleneck=projector_bottleneck,
        projector_local=projector_local,
    )


def instantiate_fs_seg_module(
    myconfig: DictConfig,
    path_ckpt: None | str = None,
    load_ours: bool = False,
):
    datamodule = instantiate(
        myconfig.datamodule.datamodule,
        config_dataset=myconfig.dataset,
        batch_size=myconfig.train.batch_size,
        _recursive_=False,
    )
    # if not torch.cuda.is_available():
    #     load_params = {"map_location": torch.device("cpu")}
    # else:
    #     load_params = {}
    if path_ckpt is not None:
        raise NotImplementedError
    else:
        pl_module = instantiate(
            myconfig.module,
            train_config=myconfig.train,
            input_channels=datamodule.num_channels,
            num_classes=datamodule.num_classes,
            stats=datamodule.s2_transform.stats,
            _recursive_=False,
        )
    return pl_module, datamodule


def instantiate_pretrained_module(
    myconfig: DictConfig, path_ckpt: str | None = None
):
    datamodule: PASTISDataModule = instantiate(
        myconfig.datamodule.datamodule,
        config_dataset=myconfig.dataset,
        batch_size=myconfig.train.batch_size,
        _recursive_=False,
    )  # TODO change for various downstream tasks
    if not torch.cuda.is_available():
        load_params = {"map_location": torch.device("cpu")}
    else:
        load_params = {}

    # pretrain_config_file_path = find_file(
    #     myconfig.dwnd_params.path_dir_model,
    #     myconfig.dwnd_params.dir_training,
    # )
    pretrain_config_file_path = "/work/scratch/data/kalinie/results/alise_preentrained/ckpt_alise_mm/.hydra/config.yaml"
    # pretrain_config_file_path = "/work/scratch/data/kalinie/results/alise_preentrained/ckpt_alise_mm/metric-epoch=136-val_total_loss=0.2422.ckpt"
    pretrain_module_config = DictConfig(open_yaml(pretrain_config_file_path))
    d_model = None
    my_logger.info(f"Found {pretrain_config_file_path}")

    old_datamodule: MMMaskDataModule = instantiate(
        pretrain_module_config.datamodule.datamodule,
        config_dataset=pretrain_module_config.dataset,
        batch_size=pretrain_module_config.train.batch_size,
        _recursive_=False,
    )
    pretrained_pl_module: AliseMM = instantiate(
        pretrain_module_config.module,
        train_config=pretrain_module_config.train,
        input_channels=old_datamodule.num_channels,
        stats=(
            old_datamodule.all_transform.s2.stats,
            old_datamodule.all_transform.s1_asc.stats,
        ),  # TODO do better than that load stats of each mod
        _recursive_=False,
    )
    if myconfig.dwnd_params.load_model:
        pretrain_module_ckpt_path = "/work/scratch/data/kalinie/results/alise_preentrained/ckpt_alise_mm/metric-epoch=136-val_total_loss=0.2422.ckpt"

        # if myconfig.precise_ckpt_path is not None:
        #     print(f"looking at {myconfig.precise_ckpt_path}")
        #     pretrain_module_ckpt_path=Path(myconfig.dwnd_params.path_dir_model).joinpath(myconfig.dwnd_params.dir_training).joinpath(myconfig.precise_ckpt_path)
        # else:
        #     pretrain_module_ckpt_path = find_good_ckpt(
        #     myconfig.dwnd_params.path_dir_model,
        #     myconfig.dwnd_params.dir_training,
        #     metric_name=myconfig.dwnd_params.ckpt_type,
        # )
    else:
        my_logger.info("Model randomly intialized")
        pretrain_module_ckpt_path = None
    ft_params = FTParams(
        pl_module=pretrained_pl_module,
        no_model=myconfig.dwnd_params.no_pretrain_model,
        ckpt_path=pretrain_module_ckpt_path,
        freeze_representation_encoder=myconfig.dwnd_params.freeze_representation_encoder,
        pretrain_module_config=pretrain_module_config,
        d_model=d_model,
    )
    if path_ckpt is not None:
        pl_module: FineTuneOneMod = FineTuneOneMod.load_from_checkpoint(
            path_ckpt,
            ft_params=ft_params,
            **load_params,
        )
    else:
        my_logger.debug(myconfig.module)
        my_logger.debug(myconfig.train)
        my_logger.debug(ft_params)
        ft_params.pretrain_module_config["path_run_dir"] = None
        pl_module: FineTuneOneMod = instantiate(
            myconfig.module,
            train_config=myconfig.train,
            input_channels=datamodule.num_channels,
            ft_params=ft_params,
            num_classes=datamodule.num_classes,
            _recursive_=False,
        )
    #
    # output_torch_file = "/work/scratch/data/kalinie/results/alise_preentrained/malice_s2.pth"
    # pl_module.repr_encoder.eval()
    # torch.save(pl_module.repr_encoder, output_torch_file)
    # exit()
    del pretrained_pl_module
    return pl_module, datamodule
