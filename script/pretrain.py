"""Main model pretrain function"""
import logging
import signal

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from mt_ssl.logs.logs import log_hyperparameters
from mt_ssl.optimizer.mixed_precision import CustomMixedPrecisionPlugin
from mt_ssl.utils.open import open_yaml
from omegaconf import DictConfig

from mmmv_ssl.data.datamodule.mm_datamodule import MMMaskDataModule
from mmmv_ssl.module.alise_mm import AliseMM

my_logger = logging.getLogger(__name__)


@hydra.main(config_path="../config/", config_name="pretrain.yaml", version_base=None)
def main(config: DictConfig):
    """Main pre-training function"""
    plugins = []
    if config.verbose == 0:
        my_logger.setLevel(logging.WARN)
    elif config.verbose == 1:
        my_logger.setLevel(logging.INFO)
    elif config.verbose == 2:
        my_logger.setLevel(logging.DEBUG)
    callbacks = [
        instantiate(cb_conf) for _, cb_conf in config.callbacks.items()
    ]
    logger = [
        instantiate(logg_conf)
        for _, logg_conf in config.logger.items()
    ]
    if config.train.trainer.precision in (16, "16"):
        plugins.append(
            CustomMixedPrecisionPlugin(precision="16-mixed", device="cuda")
        )
    if config.train.slurm_restart:
        print("Automatic restart")
        plugins.append(
            SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)
        )
    if len(plugins) == 0:
        plugins = None
    my_trainer = instantiate(
        config.train.trainer,
        callbacks=callbacks,
        logger=logger,
        max_epochs=config.train.trainer.max_epochs,
        plugins=plugins,
        _convert_="partial",
    )
    # if config.load_model:  # To continue training
    #     print("We are required to load a model ")
    #     config_path = find_file(config.path_dir_model, config.dir_training)
    #     ckpt_path = find_good_ckpt(config.path_dir_model,
    #                                config.dir_training, "last")
    #     config = DictConfig(open_yaml(config_path))
    #     print(f"We are loading {ckpt_path}")
    #     possible_load_weights = False
    # else:
    #     ckpt_path = None

    # Check if there is a hydra config with hyperparameters
    config_path = config.get("hydra_config")
    if config_path is not None:
        config = DictConfig(open_yaml(config_path))

    # Resume from checkpoint
    ckpt_path = config.get("resume_from_checkpoint")
    if ckpt_path is not None:
        my_logger.info("Training from checkpoint %s", ckpt_path)
    else:
        my_logger.info("Training from scratch")
    # Train the model
    datamodule: MMMaskDataModule = instantiate(
        config.datamodule.datamodule,
        config_dataset=config.dataset,
        _recursive_=False,
    )
    pl_module: AliseMM = instantiate(
        config.module,
        _recursive_=False
    )
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    else:
        my_logger.info("No seed set")
    if isinstance(logger, list):
        log_hyperparameters(config=config, model=pl_module, logger=logger)

    my_trainer.fit(pl_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    main()  # pylint: disable=no-value-for-parameter
