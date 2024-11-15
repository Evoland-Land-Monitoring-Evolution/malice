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

# from openeo_mmdc.dataset.to_tensor import load_transform_one_mod
# from einops import rearrange,reduce
# import torch.nn as nn 
# import torch
# from einops import repeat
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from torch.nn import functional as F
my_logger = logging.getLogger(__name__)


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


# def apply_padding(allow_padd, max_len, t, sits, doy):
#     if allow_padd:
#         padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
#         padd_doy = (0, max_len - t)
#         # padd_label = (0, 0, 0, 0, 0, self.max_len - t)
#         sits = F.pad(sits, padd_tensor)
#         doy = F.pad(doy, padd_doy)
#         padd_index = torch.zeros(max_len)
#         padd_index[t:] = 1
#         padd_index = padd_index.bool()
#     else:
#         padd_index = None

#     return sits, doy, padd_index

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
    pretrain_config_file_path = "/work/scratch/data/kalinie/MMDC/results/malice/logs/malice_pretrain/2024-11-15_08-41-31/.hydra/config.yaml"
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
        myconfig.module,
        _recursive_=False
        # TODO do better than that load stats of each mod
    )
    if myconfig.dwnd_params.load_model:
        pretrain_module_ckpt_path = "/work/scratch/data/kalinie/MMDC/results/malice/checkpoints/malice_pretrain/2024-11-15_08-41-31/epoch=53.ckpt"

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
            stats=(
                      old_datamodule.all_transform.s2.stats,
                      old_datamodule.all_transform.s1_asc.stats,
                  ),
                  ** load_params,
        )
    else:
        print(myconfig.module)
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
            stats=(
                old_datamodule.all_transform.s2.stats,
                old_datamodule.all_transform.s1_asc.stats,
            ),
            _recursive_=False,
        )

    #     PATH_DIR=Path("/work/scratch/data/kalinie/results/alise_preentrained")# modify

    #     PATH_DATA="/work/CESBIO/projects/DeepChange/Iris/PASTIS/PT_FORMAT/S2_10000.pt"
    #     PATH_CSV=PATH_DIR.joinpath("data") #modify
    #     data=torch.load(PATH_DATA)
    #     transform=load_transform_one_mod(PATH_CSV,mod="s2").transform

    #     sits_s2 = rearrange(data.sits, "c t h w-> t c h w")
    #     doy = data.doy

    #     sits_s2_, doy_, padd_index_ = apply_padding(allow_padd=True, max_len=61, t=sits_s2.shape[0], sits=sits_s2, doy=doy)

    #     norm_s2_=rearrange(transform(rearrange(sits_s2_,'t c h w -> c t h w')),'c t h w -> 1 t c h w ')[:, :, :, 32:-32, 32:-32]

    #     from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
    #     from mt_ssl.data.classif_class import ClassifBInput

    #     DEVICE = 'cuda'

    #     input_12 = ClassifBInput(sits=norm_s2_.to(DEVICE), input_doy=doy_[None, :].to(DEVICE),
    #                       padd_index=padd_index_[None, :].to(DEVICE),
    #                       mask=None, labels=None)

    #     output_torch_file = "/work/scratch/data/kalinie/results/alise_preentrained/malice_new_s2.pth"
    #     pl_module.repr_encoder.eval()

    #     out1 = pl_module.repr_encoder.forward_keep_input_dim(input_12).repr

    #     torch.save(pl_module.repr_encoder, output_torch_file)

    #     repr = torch.load(output_torch_file)
    #     repr.eval()
    #     out2 = repr.forward_keep_input_dim(input_12).repr

    #     print(out1[0])

    #     exit()
    del pretrained_pl_module
    return pl_module, datamodule
