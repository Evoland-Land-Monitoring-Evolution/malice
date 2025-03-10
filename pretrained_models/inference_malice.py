import os
from pathlib import Path

import torch
from einops import rearrange
from hydra.utils import instantiate
from mt_ssl.utils.open import open_yaml
from omegaconf import DictConfig
from openeo_mmdc.dataset.to_tensor import load_all_transforms
from ruamel.yaml import YAML

from mmmv_ssl.data.dataclass import SITSOneMod, BatchOneMod
from mmmv_ssl.data.dataset.collate_fn import collate_one_mode
from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
from mmmv_ssl.model.utils import build_encoder
from mmmv_ssl.module.alise_mm import AliseMM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def open_yaml(path_yaml: str):
    """Open yaml config"""
    with open(path_yaml) as f:
        yaml = YAML(typ="safe", pure=True)
        return yaml.load(f)


def load_checkpoint(
    hydra_conf: str | Path, pretrain_module_ckpt_path: str | Path, mod: str
) -> MonoSITSEncoder:
    pretrain_module_config = DictConfig(open_yaml(hydra_conf))

    pretrained_pl_module: AliseMM = instantiate(
        pretrain_module_config.module,
    )

    pretrained_module = pretrained_pl_module.load_from_checkpoint(
        pretrain_module_ckpt_path, map_location=torch.device(DEVICE)
    )

    repr_encoder: MonoSITSEncoder = build_encoder(
        pretrained_module=pretrained_module, mod="s2" if "2" in mod else "s1"
    )

    repr_encoder.eval()

    return repr_encoder


def apply_transform_basic(
        batch_sits: torch.Tensor, transform: torch.nn.Module
) -> torch.Tensor:
    """Normalize"""
    b, *_ = batch_sits.shape
    batch_sits = rearrange(batch_sits, "b t c h w -> c (b t ) h w")
    batch_sits = transform(batch_sits)
    batch_sits = rearrange(batch_sits, "c (b t ) h w -> b t c h w", b=b)
    return batch_sits


# For inference, we open a model checkpoint and config file
path_alise_model = "malice/malice-wr1-winv1-wcr0_f64_seed0_same_mod_epoch=142.ckpt"
hydra_conf = path_alise_model.split(".")[0] + "_config.yaml"
path_csv = "malice/"

for sat in ["S1_ASC", "S2"]:
    print(sat)
    if sat == "S1_ASC":
        b, t, c, h, w = 1, 20, 3, 64, 64    # 3 bands (VH, VV, VH/VV)
    else:
        b, t, c, h, w = 1, 20, 10, 64, 64   # 10 bands

    transforms = load_all_transforms(
        path_csv, modalities=["s1_asc", "s2"]
    )

    repr_encoder = load_checkpoint(hydra_conf, path_alise_model, sat.lower()).to(DEVICE)

    # Dummy inputs
    img = torch.randint(low=0, high=10, size=(b, t, c, h, w))

    doy = torch.sort(torch.randint(low=0, high=10, size=[b, t]), dim=1).values
    print(doy)


    # Normalize data and prepare patch
    ref_norm = apply_transform_basic(
        img, getattr(transforms, sat.lower()).transform
    )

    if b > 1:
        input_before_collate = [SITSOneMod(
            sits=ref_norm[i].to(DEVICE),
            input_doy=doy[i].to(DEVICE),
        ).apply_padding(ref_norm.shape[1] + 1) for i in range(b)]
        input = collate_one_mode(input_before_collate).to(DEVICE)

    else:
        input = SITSOneMod(
            sits=ref_norm.to(DEVICE)[0],
            input_doy=doy.to(DEVICE)[0],
        ).apply_padding(ref_norm.shape[1] + 1)

        input = BatchOneMod(
            sits=input.sits[None, ...].to(DEVICE),
            input_doy=input.input_doy[None, ...].to(DEVICE),
            padd_index=input.padd_mask[None, ...].to(DEVICE),
        )

    # Inference
    repr_encoder.eval()
    with torch.no_grad():
        ort_out = repr_encoder.forward_keep_input_dim(
            input
        ).repr

    print(ort_out.shape)
