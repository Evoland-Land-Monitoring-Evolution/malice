from typing import Literal

from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
from mmmv_ssl.module.alise_mm import AliseMM


def build_encoder(
    pretrained_module: AliseMM, mod: Literal["s1", "s2"] = "s2"
) -> MonoSITSEncoder:
    if mod == "s1":
        ubarn = pretrained_module.model.encoder.encoder_s1.ubarn
        temp_proj = pretrained_module.model.encoder.common_temp_proj.ca_s1
    elif mod == "s2":
        ubarn = pretrained_module.model.encoder.encoder_s2.ubarn
        temp_proj = pretrained_module.model.encoder.common_temp_proj.ca_s2
    else:
        raise NotImplementedError
    query = pretrained_module.model.encoder.common_temp_proj.Q
    return MonoSITSEncoder(
        ubarn=ubarn,
        temp_proj_one_mod=temp_proj,
        query=query,
        d_model=ubarn.d_model,
        input_channels=ubarn.input_channels,
    )
