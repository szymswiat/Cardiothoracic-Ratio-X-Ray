import math
from pathlib import Path

import monai.transforms as M
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mimage_utils.cxr_shenzen.dataset.shenzen_cxr_dataset import ShenzenCXRDataset
from utils.CV2Reader import CV2Reader


class ShenzenCXRDataModule(LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.prob = cfg.hparams.augment_rate
        self.workers = cfg.data.dl_workers
        self.batch_size = cfg.hparams.batch_size
        self.img_size = cfg.hparams.image_size

        df = ShenzenCXRDataset.get_dataset_df(
            Path(cfg.data.shenzen.img_path),
            Path(cfg.data.shenzen.mask_path)
        )

        self.all_df = df
        self.train_df = df.sample(frac=0.9, random_state=0)
        self.val_df = df.drop(index=self.train_df.index)

    def _transforms_train(self):
        size = self.img_size
        l_size = int(size * 1.2)
        return M.Compose([
            M.LoadImaged(['image', 'seg'], reader=CV2Reader()),
            M.AddChanneld(['image', 'seg']),
            M.Resized(['image', 'seg'], spatial_size=(l_size, l_size), size_mode='all',
                      mode=['area', 'nearest']),
            M.RandFlipd(['image', 'seg'], prob=self.prob, spatial_axis=1),
            M.Rand2DElasticd(['image', 'seg'], prob=self.prob,
                             spacing=70, magnitude_range=(0, 0.5),
                             mode=['bilinear', 'nearest']),
            M.RandRotated(['image', 'seg'], prob=self.prob, range_x=0.1 * math.pi,
                          mode=['bilinear', 'nearest']),
            M.RandZoomd(['image', 'seg'], prob=self.prob, min_zoom=0.6, max_zoom=1.2,
                        mode=['area', 'nearest']),
            M.Resized(['image', 'seg'], spatial_size=(size, size), size_mode='all',
                      mode=['area', 'nearest']),
            M.RandGaussianNoised(['image'], prob=self.prob, mean=50, std=4),
            M.Lambdad(['seg'], func=ShenzenCXRDataModule.split_masks),
            M.HistogramNormalized(['image'], num_bins=256, min=0, max=1),
            M.RepeatChanneld(['image'], repeats=3),
            M.ToTensord(['image', 'seg']),
        ])

    def _transforms_val(self):
        size = self.img_size
        return M.Compose([
            M.LoadImaged(['image', 'seg'], reader=CV2Reader()),
            M.AddChanneld(['image', 'seg']),
            M.Resized(['image', 'seg'], spatial_size=(size, size),
                      size_mode='all', mode=['area', 'nearest']),
            M.Lambdad(['seg'], func=ShenzenCXRDataModule.split_masks),
            M.HistogramNormalized(['image'], num_bins=256, min=0, max=1),
            M.RepeatChanneld(['image'], repeats=3),
            M.ToTensord(['image', 'seg']),
        ])

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader('val')

    def all_dataloader(self) -> DataLoader:
        return self._create_dataloader('all')

    def _create_dataloader(self, split: str) -> DataLoader:
        assert split in ['train', 'val', 'all']
        shuffle = split == 'train'
        transforms = self._transforms_train() if split == 'train' else self._transforms_val()

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
