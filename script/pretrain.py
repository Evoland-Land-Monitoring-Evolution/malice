import logging
import signal
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from mt_ssl.logs.logs import log_hyperparameters
from mt_ssl.optimizer.mixed_precision import CustomMixedPrecisionPlugin
from mt_ssl.utils.open import find_file, find_good_ckpt, open_yaml
from omegaconf import DictConfig

from mmmv_ssl.data.datamodule.mm_datamodule import MMMaskDataModule
from mmmv_ssl.module.alise_mm import AliseMM

my_logger = logging.getLogger(__name__)


@hydra.main(config_path="../config/", config_name="pretrain.yaml")
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
        instantiate(logg_conf)
        for _, logg_conf in myconfig.logger.items()
    ]
    if myconfig.train.trainer.precision in (16, "16"):
        plugins = [
            CustomMixedPrecisionPlugin(precision="16-mixed", device="cuda")
        ]
    else:
        plugins = None
    if myconfig.train.slurm_restart:
        print("Automatic restart")
        plugin = [
            SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)
        ]
        if plugins is not None:
            plugins += plugin
        else:
            plugins = plugin
    my_trainer = instantiate(
        myconfig.train.trainer,
        callbacks=callbacks,
        logger=logger,
        max_epochs=myconfig.train.trainer.max_epochs,
        plugins=plugins,
        _convert_="partial",
    )
    # if myconfig.load_model:  # To continue training
    #     print("We are required to load a model ")
    #     config_path = find_file(myconfig.path_dir_model, myconfig.dir_training)
    #     ckpt_path = find_good_ckpt(myconfig.path_dir_model,
    #                                myconfig.dir_training, "last")
    #     myconfig = DictConfig(open_yaml(config_path))
    #     print(f"We are loading {ckpt_path}")
    #     possible_load_weights = False
    # else:
    #     ckpt_path = None


    # Train the model
    ckpt_path = myconfig.get("resume_from_checkpoint")
    if ckpt_path is not None:
        my_logger.info("Training from checkpoint %s", ckpt_path)
    else:
        my_logger.info("Training from scratch")


    datamodule: MMMaskDataModule = instantiate(
        myconfig.datamodule.datamodule,
        config_dataset=myconfig.dataset,
        batch_size=myconfig.train.batch_size,
        _recursive_=False,
    )
    pl_module: AliseMM = instantiate(
        myconfig.module,
        train_config=myconfig.train,
        stats=(datamodule.all_transform.s2.stats,
               datamodule.all_transform.s1_asc.stats
               ),  # TODO do better than that load stats of each mod
        # _recursive_=False,
    )
    if myconfig.get("seed"):
        seed_everything(myconfig.seed, workers=True)
    else:
        my_logger.info("No seed set")
    if isinstance(logger, list):
        log_hyperparameters(config=myconfig, model=pl_module, logger=logger)

    if ckpt_path is not None:
        my_trainer.fit(pl_module, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        my_trainer.fit(pl_module, datamodule=datamodule)
        # TODO check that it does not reset the weights of the U-BARN


if __name__ == "__main__":
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    main()
