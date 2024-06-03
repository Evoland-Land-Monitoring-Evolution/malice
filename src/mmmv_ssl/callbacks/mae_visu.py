from collections import namedtuple
from collections.abc import Iterable
from typing import Literal

import torch
import torchvision
from lightning import Callback, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from mt_ssl.data.normalize import unscale_data
from openeo_mmdc.dataset.dataclass import Stats
from torch import Tensor

from mmmv_ssl.data.dataclass import BatchMMSits
from mmmv_ssl.module.alise_mm import AliseMM
from mmmv_ssl.module.dataclass import OutMMAliseF


class ImageCallbacks(Callback):

    def __init__(
            self,
            n_images,
            plot_bands=None,
            normalize=False,
            value_range=(100, 3000),
            q=0.05,
            batch_max: int = 10,
    ):
        super().__init__()
        if plot_bands is None:
            plot_bands = [2, 1, 0]
        self.n_images = n_images
        self.plot_bands = plot_bands
        self.normalize = normalize
        self.value_range = (int(value_range[0]), int(value_range[1]))
        self.q = float(q)
        self.batch_max = batch_max

    def load_grid_logger(
        self,
        grid: Tensor,
        trainer: Trainer | Iterable[Trainer],
        description: str = "",
        opt: Literal["image", "figure"] = "image",
    ):
        if isinstance(trainer.logger, TensorBoardLogger):
            if opt == "image":
                trainer.logger.experiment.add_image(
                    description,
                    grid,
                    global_step=trainer.global_step,
                )
            elif opt == "figure":
                trainer.logger.experiment.add_figure(
                    description,
                    grid,
                    global_step=trainer.global_step,
                )
            else:
                raise NotImplementedError
        elif isinstance(trainer.loggers, list):
            for logg in trainer.loggers:
                if isinstance(logg, TensorBoardLogger):
                    if opt == "image":
                        trainer.logger.experiment.add_image(
                            description,
                            grid,
                            global_step=trainer.global_step,
                        )
                    elif opt == "figure":
                        trainer.logger.experiment.add_figure(
                            description,
                            grid,
                            global_step=trainer.global_step,
                        )
                    else:
                        raise NotImplementedError
        else:
            print(f"Unable to find tensorboard logger {trainer.loggers}")


class MAECrossRecClb(ImageCallbacks):

    def __init__(
        self,
        n_images,
        plot_bands=None,
        normalize=False,
        value_range=(100, 3000),
        q=0.05,
        batch_max: int = 10,
        opt: Literal["s1a", "s1b", "s2a", "s2b"] = "s1a",
    ):
        super().__init__(n_images, plot_bands, normalize, value_range, q,
                         batch_max)
        self.opt = opt

    def on_validation_batch_end(
        self,
        trainer: Trainer | Iterable[Trainer],
        pl_module: AliseMM,
        outputs: OutMMAliseF,
        batch: BatchMMSits,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        images_bf = self.show_reconstructions(outputs,
                                              tuple_stats=pl_module.stats,
                                              opt=self.opt,
                                              batch=batch)
        final_grid_bf = self.make_grid(images_bf)
        self.load_grid_logger(final_grid_bf, trainer, description=self.opt)

    def extract_per_view(
        self,
        batch: BatchMMSits,
        out_model: OutMMAliseF,
        opt: Literal["s1a", "s1b", "s2a", "s2b"] = "s1a",
    ):
        if opt == "s1a":
            trg = batch.sits1a.sits
            pred = out_model.rec.s1a
        elif opt == "s1b":
            trg = batch.sits1b.sits
            pred = out_model.rec.s1b
        elif opt == "s2a":
            trg = batch.sits2a.sits
            pred = out_model.rec.s2a
        elif opt == "s2b":
            trg = batch.sits2b.sits
            pred = out_model.rec.s2b
        else:
            raise NotImplementedError
        return trg, pred

    def show_reconstructions(
        self,
        out_model: OutMMAliseF,
        tuple_stats: tuple[Stats,Stats],
        opt: Literal["s1a", "s1b", "s2a", "s2b"] = "s1a",
        export_doy: bool = False,
        batch: BatchMMSits | None = None,
    ):
        trg, pred = self.extract_per_view(batch=batch,
                                          out_model=out_model,
                                          opt=opt)
        # print(f"in clb tragte {trg[0,0,0,...]}")
        if "s1" in opt:
            stats = None
        elif "s2" in opt:
            stats = tuple_stats[0]
        else:
            stats = tuple_stats[1]
        if stats is not None:
            unscale_trg = unscale_data(
                stats,
                trg[0, :self.n_images, ...].cpu(),
            )[:, self.plot_bands, ...]
            unscale_mnm = unscale_data(
                stats,
                pred.same_mod[0, :self.n_images, ...].cpu(),
            )[:, self.plot_bands, ...]
            unscale_crm = unscale_data(
                stats,
                pred.other_mod[0, :self.n_images, ...].cpu(),
            )[:, self.plot_bands, ...]
        else:
            # print(f"in callbakcs {trg[0, 0, 0, ...]}")
            unscale_trg = trg[0, :self.n_images, self.plot_bands, ...].cpu()
            unscale_mnm = pred.same_mod[0, :self.n_images, self.plot_bands,
                                        ...].cpu()
            unscale_crm = pred.other_mod[0, :self.n_images, self.plot_bands,
                                         ...].cpu()
        # print(f"in callbakcs {unscale_trg[0,0, ...]}")
        OutVisu = namedtuple("OutVisu", ["trg", "mnm", "crm"])
        torch.save(unscale_trg, f"{opt}_out_visu.pt")
        return OutVisu(unscale_trg, unscale_mnm, unscale_crm)

    def make_grid(self, input) -> Tensor:
        grid1 = torchvision.utils.make_grid(
            input.trg,
            nrow=self.n_images,
            normalize=self.normalize,
            value_range=self.value_range,
        )
        grid2 = torchvision.utils.make_grid(
            input.mnm,
            nrow=self.n_images,
            normalize=self.normalize,
            value_range=self.value_range,
        )
        grid3 = torchvision.utils.make_grid(
            input.crm,
            nrow=self.n_images,
            normalize=self.normalize,
            value_range=self.value_range,
        )
        return torch.cat([grid1, grid2, grid3], dim=1)


class EmbeddingsVisu(ImageCallbacks):

    def __init__(
        self,
        n_images,
        plot_bands=None,
        normalize=False,
        value_range=(100, 3000),
        q=0.05,
        batch_max: int = 10,
        opt: Literal["s1a", "s1b", "s2a", "s2b"] = "s1a",
    ):
        super().__init__(n_images, plot_bands, normalize, value_range, q,
                         batch_max)
        self.opt = opt

    def on_validation_batch_end(
        self,
        trainer: Trainer | Iterable[Trainer],
        pl_module: AliseMM,
        outputs: OutMMAliseF,
        batch: BatchMMSits,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.opt == "s2a":
            repr = outputs.repr.s2a
        elif self.opt == "s1b":
            repr = outputs.repr.s1b
        elif self.opt == "s1a":
            repr = outputs.repr.s1a
        elif self.opt == "s2b":
            repr = outputs.repr.s2b
        else:
            raise NotImplementedError
        images = repr[0, :self.n_images, [0, 1, 2], ...].cpu().float()
        grid = torchvision.utils.make_grid(
            images,
            nrow=self.n_images,
            normalize=self.normalize,
            value_range=self.value_range,
        )

        self.load_grid_logger(grid,
                              trainer,
                              description=f"Embedding {self.opt}")
