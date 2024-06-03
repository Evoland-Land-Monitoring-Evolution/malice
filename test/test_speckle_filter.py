import torch

from mmmv_ssl.utils.speckle_filter import despeckle_batch, window_reduction


def test_despeckle_batch():
    b, c, h, w = 2, 3, 64, 64
    batch = torch.randn(b, c, h, w)
    # print(batch[0, ...])
    out = despeckle_batch(batch)
    print(out[0, :, 3:61, 3:61])
    print(out)


def test_window_reduction():
    b, c, h, w = 2, 3, 64, 64
    batch = torch.randn(b, c, h, w)
    out = window_reduction(batch, torch.mean, kernel_size=7)
    print(out.shape)
    print(out[:, :, 3:61, 3:61])
