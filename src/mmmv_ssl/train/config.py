from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CAWConfig:
    _target_: Any = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    T_0: int = 100
    T_mult: int = 2


@dataclass
class AdamConfig:
    _target_: Any = torch.optim.Adam


@dataclass
class VicRegTrainConfig:
    lr: float
    scheduler: CAWConfig
    optimizer: AdamConfig
    w_var: float = 25
    w_inv: float = 15
    w_cov: float = 1
    w_bottle: float = 1
    w_local: float = 1
    batch_size: int = 2
