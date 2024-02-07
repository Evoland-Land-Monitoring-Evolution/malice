import torch

from mmmv_ssl.loss.vicreg_loss import CovarianceLoss, VarianceLoss


def test_forward():
    b, t, c, h, w = 2, 1, 64, 16, 16
    repr_1 = torch.randn(b, t, c, h, w)
    repr_2 = torch.randn(b, t, c, h, w)
    var_loss = VarianceLoss()
    out_loss = var_loss(repr_1, repr_2)
    print(out_loss.shape)
    # print(out_loss)


def test_forward_cov():
    b, t, c, h, w = 2, 1, 64, 16, 16
    repr_1 = torch.ones(b, t, c, h, w)
    repr_2 = torch.ones(b, t, c, h, w)
    cov_loss = CovarianceLoss(c)
    out_loss = cov_loss(repr_1, repr_2, batch_size=b)
    print(out_loss)
