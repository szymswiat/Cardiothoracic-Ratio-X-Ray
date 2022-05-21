import logging
from pprint import pprint
from pathlib import Path
from typing import Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load

from data.shenzen_cxr_dm import ShenzenCXRDataModule
from inference.models.unet_module import UNetModule
from training.common_training_object import CommonTrainingObject
from training.modules.common_training_module import CommonTrainingModule
from training.modules.segmentation_training_module import SegmentationTrainingModule
from utils.upload_weigths_callback import UploadWeightsCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShenzenSegTrainingObject(CommonTrainingObject):

    def __init__(
            self,
            project_name: str,
            cfg: DictConfig,
            run_offline: bool,
            run_gpu: bool,
            run_ddp: bool,
            run_remote: bool
    ):
        super().__init__(project_name, cfg, run_offline, run_gpu, run_ddp, run_remote)

        self.model_class = UNetModule

    def _setup_training_module(self) -> CommonTrainingModule:
        model = self.model_class(self.cfg.hparams)
        training_module = SegmentationTrainingModule(self.cfg.hparams, model)

        return training_module

    def _load_training_module_before_test(self) -> LightningModule:
        if self.model_checkpoint is None:
            return self.training_module

        best_ckpt_path = Path(self.model_checkpoint.best_model_path)

        checkpoint = pl_load((Path(self.paths.checkpoint_dir) / best_ckpt_path).as_posix())

        training_module = SegmentationTrainingModule._load_model_state(
            checkpoint=checkpoint,
            model=self.model_class(self.cfg.hparams)
        )

        return training_module

    def _setup_data_module(self) -> ShenzenCXRDataModule:

        data_module = ShenzenCXRDataModule(self.cfg)

        return data_module

    def _setup_model_checkpoint(self) -> Optional[ModelCheckpoint]:
        return ModelCheckpoint(
            filename='epoch={epoch}_val_dice_avg={dice_avg/val:.3f}_top',
            monitor='dice_avg/val',
            mode='max',
            dirpath=self.paths.checkpoint_dir,
            verbose=True,
            save_top_k=1,
            auto_insert_metric_name=False
        )

    def _upload_output_model(self):
        best_ckpt_path = Path(self.model_checkpoint.best_model_path)
        weights_path = Path(self.paths.checkpoint_dir) / (best_ckpt_path.stem + '.pt')

        if self.trainer.is_global_zero:
            self.training_module.model.save_state_to_file(weights_path)
            self.task.update_output_model(weights_path.as_posix(), tags=['dice_best'])

    def _setup_callbacks(self) -> List[Callback]:
        callbacks = []
        if self.cfg.hparams.weights_upload_interval:
            w_callback = UploadWeightsCallback(self.cfg.hparams.weights_upload_interval, self.task)
            callbacks.append(w_callback)

        return callbacks


@hydra.main('../config', 'config')
def train(cfg: DictConfig):

    pprint(OmegaConf.to_object(cfg))

    training_obj = ShenzenSegTrainingObject(
        project_name=cfg.project_name,
        cfg=cfg,
        run_offline=cfg.run_config.offline,
        run_gpu=cfg.run_config.run_gpu,
        run_ddp=cfg.run_config.run_ddp,
        run_remote=cfg.run_config.remote
    )
    training_obj.train_and_test()


if __name__ == '__main__':
    train()
