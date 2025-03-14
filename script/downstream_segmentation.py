import logging
import signal
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from mt_ssl.logs.logs import log_hyperparameters
from mt_ssl.utils.open import find_file, find_good_ckpt, open_yaml
from omegaconf import DictConfig

from mmmv_ssl.module.instantiate import (
    instantiate_fs_seg_module,
    instantiate_pretrained_module,
)

my_logger = logging.getLogger(__name__)


@hydra.main(config_path="../config/", config_name="downstream_task_seg.yaml")
def main(myconfig: DictConfig):
    if myconfig.verbose == 0:
        my_logger.setLevel(logging.WARN)
    elif myconfig.verbose == 1:
        my_logger.setLevel(logging.INFO)
    elif myconfig.verbose == 2:
        my_logger.setLevel(logging.DEBUG)
    callbacks = [
        instantiate(cb_conf) for _, cb_conf in myconfig.callbacks.items()
    ]
    logger = [
        instantiate(logg_conf, save_dir=Path.cwd())
        for _, logg_conf in myconfig.logger.items()
    ]
    if myconfig.train.slurm_restart:
        print("Automatic restart")
        plugin = [
            SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)
        ]
    else:
        plugin = None
    my_trainer = instantiate(
        myconfig.train.trainer,
        callbacks=callbacks,
        logger=logger,
        max_epochs=myconfig.train.trainer.max_epochs,
        plugins=plugin,
        _convert_="partial",
    )
    if myconfig.checkpoint_dir is not None:
        print(myconfig)
        config_path = find_file(
            myconfig.checkpoint_dir, myconfig.checkpoint_tr
        )

        ckpt_path = find_good_ckpt(
             myconfig.checkpoint_dir,
             myconfig.checkpoint_tr,
             myconfig.metrics_pretrained)
        myconfig = DictConfig(open_yaml(config_path))
        print(f"We are loading {ckpt_path}")
    else:
        my_logger.info("We ate not loading pretrained model")
        ckpt_path = None
    # ckpt_path = "/work/scratch/data/kalinie/MMDC/results/malice/fine_tune/fine_tune2023-11-10_14-33-22/no_pretrain_modelFalse_load_model_True_freeze_True/val[4]/training_folds[1, 2, 3]all/requeue/fine_tune/2024-10-29_13-51-30/checkpoints/last.ckpt"

    if myconfig.fully_supervised:
        pl_module, datamodule = instantiate_fs_seg_module(
            myconfig, path_ckpt=None, load_ours=True
        )
    else:
        pl_module, datamodule = instantiate_pretrained_module(
            myconfig, ckpt_path
        )
    if myconfig.get("seed"):
        seed_everything(myconfig.seed, workers=True)
    else:
        my_logger.info("No seed set")
    log_hyperparameters(config=myconfig, model=pl_module, logger=logger)
    if myconfig.compile:
        my_logger.info("Run pytorch compile")
        # torch._dynamo.config.verbose = True
        # torch._dynamo.config.log_level = logging.DEBUG
        # torch._dynamo.config.suppress_errors = True
        compiled_pl_module = torch.compile(pl_module, backend="eager")
        my_trainer.fit(
            compiled_pl_module, datamodule=datamodule, ckpt_path=ckpt_path
        )
    else:

        my_trainer.fit(pl_module, datamodule=datamodule, ckpt_path=ckpt_path)
    datamodule.setup("test")
    my_trainer.test(pl_module,datamodule)
    test_metrics=pl_module.save_test_metrics
    torch.save(test_metrics,"test_metrics.pt")
if __name__ == "__main__":
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    main()
