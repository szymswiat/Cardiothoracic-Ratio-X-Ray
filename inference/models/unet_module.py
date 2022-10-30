from typing import Any

import segmentation_models_pytorch as smp
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Sigmoid

from utils.loadable_module import LoadableModule


class UNetModule(LoadableModule):

    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        self.backbone = smp.Unet(encoder_name=hparams.backbone,
                                 encoder_weights=hparams.pretrained,
                                 classes=3,
                                 in_channels=3)

        self.final_activation = Sigmoid()

    def forward(self, x: Tensor, apply_act=True) -> Any:
        features = self.backbone(x)

        if apply_act is False:
            return features

        return self.final_activation(features)
