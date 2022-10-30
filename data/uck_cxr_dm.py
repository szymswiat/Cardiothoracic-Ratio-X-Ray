import math
from pathlib import Path

import monai.transforms as M
import numpy as np
from mimage_utils.cxr_uck.dataset.uck_cxr_dataset import UCKCXRDataset
from mimage_utils.cxr_uck.readers.zarr_reader import ZarrReader
from monai.data import ITKReader
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class UCKCXRDataModule(LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.prob = cfg.hparams.augment_rate
        self.workers = cfg.data.dl_workers
        self.batch_size = cfg.hparams.batch_size
        self.img_size = cfg.hparams.image_size

        df = UCKCXRDataset.get_dataset_df(
            Path(cfg.data.uck_ctr.img_path),
            Path(cfg.data.uck_ctr.mask_path),
        )
        self.all_df = df
        self.train_df = df.sample(frac=0.9, random_state=0)
        self.val_df = df.drop(index=self.train_df.index)

    @property
    def _transforms_train(self):
        size = self.img_size
        l_size = int(size * 1.2)

        seg_reader_transform = M.LoadImaged(['seg'])
        seg_reader_transform._loader.readers = [ZarrReader(['lung', 'heart'])]

        return M.Compose([
            M.LoadImaged(['image'], reader=ITKReader()),
            seg_reader_transform,
            M.Lambdad(['seg'], UCKCXRDataModule.synthesize_bg_mask),
            M.Lambdad(['image'], UCKCXRDataModule.reverse_channels),
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
            M.HistogramNormalized(['image'], num_bins=4100, min=0, max=1),
            M.RepeatChanneld(['image'], repeats=3),
            M.ToTensord(['image', 'seg']),
        ])

    @property
    def _transforms_val(self):
        size = self.img_size

        seg_reader_transform = M.LoadImaged(['seg'])
        seg_reader_transform._loader.readers = [ZarrReader(['lung', 'heart'])]

        return M.Compose([
            M.LoadImaged(['image'], reader=ITKReader()),
            seg_reader_transform,
            M.Lambdad(['seg'], UCKCXRDataModule.synthesize_bg_mask),
            M.Lambdad(['image'], UCKCXRDataModule.reverse_channels),
            M.Resized(['image', 'seg'], spatial_size=(size, size),
                      size_mode='all', mode=['area', 'nearest']),
            M.HistogramNormalized(['image'], num_bins=4100, min=0, max=1),
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
        transforms = self._transforms_train if split == 'train' else self._transforms_val

        split_set = UCKCXRDataset(getattr(self, f'{split}_df'), transforms)

        return DataLoader(split_set, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=self.workers)

    @staticmethod
    def synthesize_bg_mask(masks: np.ndarray):
        bg_mask = np.zeros(masks.shape[-2:])
        bg_mask[np.sum(masks, axis=0) == 0] = 1
        return np.concatenate([np.expand_dims(bg_mask, axis=0), masks], axis=0)

    @staticmethod
    def reverse_channels(img: np.ndarray):
        img = np.moveaxis(img, -2, 0)
        img = np.moveaxis(img, -1, 0)

        return img
