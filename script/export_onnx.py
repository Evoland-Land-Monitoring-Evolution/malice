import torch
from mt_ssl.data.mt_batch import BInput4d
from torch import Tensor

from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
from mmmv_ssl.model.utils import build_encoder
from mmmv_ssl.module.alise_mm import AliseMM

PATH_MODEL = "/lustre/fswork/projects/rech/wfi/ubc65oh/trainings/mm_mae_trainings/mm_mae_sd_manuscript/wrec1_winv1_wcr0.5/bs1/proj_freezeTrue/seed4/true_requeue/version_2/checkpoints/metric-epoch=37-val_total_loss=0.2342.ckpt"
MOD = "s2"


class MaliceOnnx(torch.nn.Module):
    def __init__(self, repr_encoder):
        super().__init__()
        self.repr_encoder = repr_encoder

    def forward(
        self, sits: Tensor, tpe: Tensor, padd_mask: Tensor, cld_mask: Tensor
    ):
        input_alise = BInput4d(
            sits=sits, input_doy=tpe, padd_index=padd_mask, cld_mask=cld_mask
        )
        out = self.repr_encoder.forward_keep_input_dim(input_alise)
        return out.repr


def main():
    b = 1
    n = 11
    h, w = 64, 64
    if torch.cuda.is_available():
        module = AliseMM.load_from_checkpoint(PATH_MODEL)
        dummy_input = (
            torch.randn(b, n, 10, h, w).cuda(),
            torch.ones(b, n).cuda(),
            torch.zeros(b, n).bool().cuda(),
            torch.zeros(b, n, h, w).bool().cuda(),
        )
    else:
        module = AliseMM.load_from_checkpoint(
            PATH_MODEL, map_location=torch.device("cpu")
        )
        dummy_input = (
            torch.randn(b, n, 10, h, w),
            torch.ones(b, n),
            torch.zeros(b, n).bool(),
            torch.zeros(b, n, h, w).bool(),
        )
    module.load_weights(PATH_MODEL)

    input_names = ["sits", "tpe", "padd_mask", "cld_mask"]
    dynamic_axes = {
        "sits": {0: "b", 1: "n", 3: "h", 4: "w"},
        "tpe": {0: "b", 1: "n"},
        "padd_mask": {0: "b", 1: "n"},
        "cld_mask": {0: "b", 1: "n", 3: "h", "w": 4},
    }
    repr_encoder: MonoSITSEncoder = build_encoder(
        pretrained_module=module, mod=MOD
    )
    repr_encoder.eval()
    alise_model = MaliceOnnx(repr_encoder)
    alise_model.eval()
    torch.onnx.export(
        alise_model,
        dummy_input,
        f"/lustre/fswork/projects/rech/wfi/ubc65oh/results/malice_flexible_{MOD}.onnx",
        opset_version=18,
        export_params=True,
        verbose=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    main()
