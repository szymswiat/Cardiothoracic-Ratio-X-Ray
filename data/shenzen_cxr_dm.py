import math
from pathlib import Path

import monai.transforms as M
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.shenzen_cxr_dataset import ShenzenCXRDataset
from utils.CV2Reader import CV2Reader


class ShenzenCXRDataModule(LightningDataModule):

    def __init__(
            self,
            img_path: Path,
            mask_path: Path,
            batch_size: int = 32,
            augment_rate: float = 0.4,
            workers: int = 4
    ):
        super().__init__()

        self.prob = augment_rate
        self.workers = workers
        self.batch_size = batch_size

        df = ShenzenCXRDataset.get_dataset_df(img_path, mask_path)

        self.train_df = df.sample(frac=0.9, random_state=0)
        self.val_df = df.drop(index=self.train_df.index)

    @property
    def _transforms_train(self):
        middle_tf = M.Compose([
            M.RandFlipd(['image', 'seg'], prob=self.prob, spatial_axis=1),
            M.Rand2DElasticd(['image', 'seg'], prob=self.prob,
                             spacing=70, magnitude_range=(0, 0.5),
                             mode=['bilinear', 'nearest']),
            M.RandRotated(['image', 'seg'], prob=self.prob, range_x=0.1 * math.pi,
                          mode=['bilinear', 'nearest']),
            # M.RandAdjustContrastd(['image'], prob=self.prob, gamma=(1.0, 2.0))
        ])

        return self.get_transforms_common(middle_tf)

    @property
    def _transforms_val(self):
        return self.get_transforms_common()

    def get_transforms_common(self, middle_tf: M.Transform = M.Compose([])):
        return M.Compose([
            M.LoadImaged(['image', 'seg'], reader=CV2Reader()),
            M.AddChanneld(['image', 'seg']),
            M.Resized(['image', 'seg'], spatial_size=(512, 512), size_mode='all',
                      mode=['area', 'nearest']),
            M.Lambdad(['seg'], func=self.split_masks),
            middle_tf,
            M.HistogramNormalized(['image'], num_bins=256, min=0, max=1),
            M.RepeatChanneld(['image'], repeats=3),
            M.ToTensord(['image', 'seg']),
            # M.AsChannelLastd(['image', 'seg']),
        ])

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader('val')

    def _create_dataloader(self, split: str) -> DataLoader:
        assert split in ['train', 'val']
        shuffle = split == 'train'
        transforms = self._transforms_train if split == 'train' else self._transforms_val

        split_set = ShenzenCXRDataset(getattr(self, f'{split}_df'), transforms)

        return DataLoader(split_set, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=self.workers)

    @staticmethod
    def split_masks(mask: np.ndarray):
        mapping = ShenzenCXRDataset.MASK_MAP
        mask = mask.squeeze(axis=0)

        masks = np.zeros([len(mapping), *mask.shape])

        for i in range(len(mapping)):
            masks[i][mask == i] = 1

        return masks

    @staticmethod
    def align_masks(masks: np.ndarray):
        masks[masks > 0] = 1

        masks[0][masks[1:].sum(axis=0) == 0] = 1

        return masks
