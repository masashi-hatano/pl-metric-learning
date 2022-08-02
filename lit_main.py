import os
import logging
import warnings

warnings.filterwarnings("ignore")

import hydra
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from data_module.lit_cifar10_data_module import Cifar10DataModule
from model.lit_MetricTrainer import MetricTrainer


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config):
    cfg = config.main
    seed_everything(cfg.seed)
    model_cfg = dict(config.model_config)
    trainer_cfg = dict(cfg.trainer)
    # logger_cfg = dict(cfg.logger)

    data_module = Cifar10DataModule(config=config)
    labels = list(config.data_module.class_names)

    model = MetricTrainer(
        **model_cfg,
        class_names=labels,
    )

    train_logger = loggers.TensorBoardLogger("tensor_board", default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
        mode="min",
    )

    trainer = Trainer(
        **trainer_cfg,
        logger=train_logger,
        callbacks=[checkpoint_callback],
    )

    if cfg.train_mode == "train":
        trainer.fit(model, data_module)
        print(trainer.callback_metrics)
    elif cfg.train_mode == "test":
        logging.basicConfig(level=logging.DEBUG)
        model = MetricTrainer.load_from_checkpoint(cfg.ckpt_pth, **model_cfg, class_names=labels)
        trainer.test(model, datamodule=data_module)
    else:
        raise AssertionError("Make sure train mode is in train, test, infer")



if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()