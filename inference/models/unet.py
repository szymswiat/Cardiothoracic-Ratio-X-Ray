from typing import Optional, Any

from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp
from torch import Tensor
from torch.nn import Softmax2d


class UnetModule(LightningModule):

    def __init__(
            self,
            encoder: str = 'efficientnet-b1',
            pretrained_weights: Optional[str] = 'imagenet',

    ):
        super().__init__()

        self.backbone = smp.Unet(encoder_name=encoder,
                                 encoder_weights=pretrained_weights,
                                 classes=3,
                                 in_channels=3)

        self.final_activation = Softmax2d()

    def forward(self, x: Tensor) -> Any:
        features = self.backbone(x)

        return self.final_activation(features)

