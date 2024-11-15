import torch

from mmmv_ssl.data.dataclass import BatchMMSits, BatchOneMod


def create_fake_batch(b=2, t1=10, c1=3, t2=20, c2=6, h=8, w=8) -> BatchMMSits:
    batch_1 = BatchOneMod(
        sits=torch.randn(b, t1, c1, h, w),
        input_doy=torch.arange(t1)[None, :].expand(b, -1),
        padd_index=torch.zeros(b, t1).bool(),
    )
    batch_2 = BatchOneMod(
        sits=torch.randn(b, t2, c2, h, w),
        input_doy=torch.arange(t2)[None, :].expand(b, -1),
        padd_index=torch.zeros(b, t2).bool(),
    )
    return BatchMMSits(sits1=batch_1, sits2=batch_2)
