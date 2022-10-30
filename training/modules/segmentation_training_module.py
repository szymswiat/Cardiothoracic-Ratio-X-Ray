import torch_optimizer as optim
from monai.networks import one_hot
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from typing import List, Any

from torchtools.lr_scheduler import DelayedCosineAnnealingLR

from training.modules.common_training_module import CommonTrainingModule
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torchtools.optim import RangerLars


# from segmentation_models_pytorch.utils.metrics import IoU


class SegmentationTrainingModule(CommonTrainingModule):

    def __init__(
            self,
            hparams: DictConfig,
            model: LightningModule
    ):
        super().__init__(hparams)

        self.model = model

        self.criterion = DiceCELoss(include_background=False, sigmoid=True)

        self.val_dice = DiceMetric(include_background=False)
        # self.val_iou = IoU(ignore_channels=[0])

    def forward(self, x):
        raise AttributeError('Not supported in training module.')

    def training_step(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            optimizer = self.optimizers()
            for i, group in enumerate(optimizer.param_groups):
                self.log(f'lr/param_group_{i}', group['lr'], on_step=True, on_epoch=False,
                         logger=True, prog_bar=True, rank_zero_only=True)

        x, y_true = batch
        y_pred = self.model(x, apply_act=False)

        loss = self.criterion(y_pred, y_true)

        self.log('loss/train', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x, apply_act=False)

        loss = self.criterion(y_pred, y_true)

        y_pred = self.model.final_activation(y_pred)
        y_pred = one_hot(y_pred.argmax(dim=1).unsqueeze(dim=1), y_pred.size()[1])

        self.log('loss/val', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        self.val_dice(y_pred, y_true)
        # self.val_iou(y_pred, y_true)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_dice = self.val_dice.aggregate()
        # avg_iou = self.val_iou.compute()

        self.log('dice_avg/val', value=avg_dice, on_epoch=True, on_step=False,
                 logger=False, prog_bar=False)

        self.cml_logger.report_scalar(title='avg_dice',
                                      series='val',
                                      value=avg_dice,
                                      iteration=self.trainer.current_epoch)

    def configure_optimizers(self):
        opt = self.hparams.optimizer
        common_params = dict(
            params=self.parameters(),
            lr=opt.get('lr_initial'),
            weight_decay=opt.get('weight_decay', 0)
        )

        result = {}

        if opt.type == 'swats':
            result['optimizer'] = optim.SWATS(**common_params)
        elif opt.type == 'ranger_lars':
            result['optimizer'] = RangerLars(**common_params)
            result['scheduler'] = DelayedCosineAnnealingLR(result['optimizer'],
                                                           self.hparams.epochs - opt.annealing_epochs,
                                                           opt.annealing_epochs)
        else:
            raise ValueError('Invalid optimizer')

        return result
