"""Image processing filters (speckle, etc.)"""

from collections.abc import Callable

import findpeaks
import torch
import torch.nn.functional as F


def window_reduction(
    data: torch.Tensor,
    reduction: Callable,
    kernel_size: int = 3,
    stride: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Apply a reduction function to a batch of image patches.
    The reduction function will be used as a sliding window with
    the given kernel size for the reduction computation and the result will
    be assigned to the center pixel of the window.
    The patches are paded to take into account the kernel size and keep the
    initial patch size."""
    nb_samples, nb_channels, _, _ = data.shape
    if stride is None:
        stride = 1
    margin = (kernel_size - 1) // 2
    p_x = data.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    unfold_shape = p_x.size()
    patches = p_x.contiguous().view(-1, kernel_size, kernel_size)
    running_red = reduction(patches, dim=(-1, -2))
    refold_shape = list(unfold_shape)
    refold_shape[-2:] = [1, 1]

    patches_orig = running_red.view(tuple(refold_shape))
    output_h = refold_shape[2] * refold_shape[4]
    output_w = refold_shape[3] * refold_shape[5]
    patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
    patches_orig = patches_orig.view(
        nb_samples, nb_channels, output_h, output_w
    )
    # print(f"margin {margin}")
    # print(f"patcg orig {patches_orig}")
    return F.pad(patches_orig, (margin, margin, margin, margin)), margin


def lee_filter(
    data: torch.Tensor, win_size: int, c_u: float = 0.25
) -> tuple[torch.Tensor, int]:
    """Basic Lee despeckle filter. Custom implementation."""
    x_mean, margin = window_reduction(data, torch.mean, kernel_size=win_size)
    x_var, _ = window_reduction(data, torch.var, kernel_size=win_size)
    # print(f"var {x_var[0,:,:,:]}")
    c_i = torch.sqrt(x_var) / (x_mean + 1e-5)
    # print(f"ci:{c_i}")
    w_t = F.relu(1.0 - (c_u * c_u / (c_i * c_i))) + 1e-5
    # print(f"wt : {w_t[0,...]}")
    # print(f"data {data[0,:,:3,:3]}")
    res: torch.Tensor = data * w_t + x_mean * (1.0 - w_t)
    return res, margin


def despeckle(data: torch.Tensor) -> torch.Tensor:
    """Lee filter using the findpeaks library"""
    d_min0, d_max0 = data.min(), data.max()
    data = data - d_min0
    img = findpeaks.stats.scale(data)
    d_min1, d_max1 = img.min(), img.max()
    winsize = 3
    cu_value = 0.25
    image_lee = findpeaks.lee_filter(img, win_size=winsize, cu=cu_value)
    image_lee = (
        image_lee * (d_max0 - d_min0) / (d_max1 - d_min1 + 1e-3) + d_min0
    )
    return torch.tensor(image_lee)


def despeckle_sample(data: torch.Tensor) -> torch.Tensor:
    """Use findpeaks to despeckle an image patch"""
    x_linear = (
        torch.exp(data).cpu().detach().numpy()
    )  # remove log before despeckle
    vv_asc = despeckle(x_linear[0, :, :])
    vh_asc = despeckle(x_linear[1, :, :])
    res = torch.stack(
        [vv_asc, vh_asc, vv_asc / vh_asc],
        dim=0,
    )
    return torch.log(res)


def despeckle_batch_fp(data: torch.Tensor) -> torch.Tensor:
    """Despeckle all the patches in a batch using findpeaks and Lee filter"""
    out_batch_samples = [
        despeckle_sample(data[j, :, :, :]).unsqueeze(0)
        for j in range(data.shape[0])
    ]
    return torch.cat(out_batch_samples, dim=0).to(data.device)


def despeckle_batch(data: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Despeckle all the patches in a batch using window reduction and Lee filter"""
    res, margin = lee_filter(torch.exp(data), win_size=7)
    # print(res[0,...])
    return torch.log(res), margin
