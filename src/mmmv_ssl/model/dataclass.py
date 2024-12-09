""" Output dataclasses """

from dataclasses import dataclass

import torch

@dataclass
class OutUTAEForward:
    """ UTAE outputs """
    seg_map: torch.Tensor
    attn: torch.Tensor | None = None
    feature_maps: list[torch.Tensor] | None = None


@dataclass
class OutTempProjForward:
    """ Temporal projector outputs """
    s1: torch.Tensor    # pylint: disable=C0103
    s2: torch.Tensor    # pylint: disable=C0103
