"""
Util to build encoder from Malice Lightning module
"""

from typing import Literal

from mmmv_ssl.model.sits_encoder import MonoSITSEncoder, MonoSITSAuxEncoder
from mmmv_ssl.module.alise_mm import AliseMM, AliseMMModuleAux


def build_encoder(
        pretrained_module: AliseMM, mod: Literal["s1", "s2"] = "s2"
) -> MonoSITSEncoder | MonoSITSAuxEncoder:
    """
    Builds one modality encoder from Malice lightning module.
    """
    repr_encoder = pretrained_module.model.encoder
    if mod == "s1":
        encoder = repr_encoder.encoder_s1
        temp_proj = repr_encoder.common_temp_proj.ca_s1
    elif mod == "s2":
        encoder = repr_encoder.encoder_s2
        temp_proj = repr_encoder.common_temp_proj.ca_s2
    else:
        raise NotImplementedError
    query = repr_encoder.common_temp_proj.Q
    if isinstance(pretrained_module.model, AliseMMModuleAux):
        return MonoSITSAuxEncoder(
            encoder=encoder,
            temp_proj_one_mod=temp_proj,
            query=query
        )
    return MonoSITSEncoder(
        encoder=encoder,
        temp_proj_one_mod=temp_proj,
        query=query
    )


