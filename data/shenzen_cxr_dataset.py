from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from pandas import DataFrame
import cv2


class ShenzenCXRDataset(Dataset):

    def __init__(
            self,
            input_df: DataFrame,
            transforms: A.Compose = A.Compose([])
    ):
        self._df = input_df
        self._transforms = transforms

    def __getitem__(self, index: int) -> Any:
        img = self.read_img(self.get_image_path(index))
        mask = self.read_img(self.get_mask_path(index))

        aug_img = self._transforms(image=img)
        aug_mask = self._transforms(image=mask)

        return aug_img['image'], aug_mask['image']

    def get_image_path(self, index: int) -> Path:
        return Path(self._df['image_path'].iloc[index])

    def get_mask_path(self, index: int) -> Path:
        return Path(self._df['mask_path'].iloc[index])

    @staticmethod
    def read_img(path: Path) -> np.ndarray:
        img = np.array(cv2.imread(path.as_posix()))

        # if len(img.shape) == 2:
        #     img = np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1)

        return img

    @staticmethod
    def get_dataset_df(
            image_root_dir: Path,
            mask_root_dir: Path
    ) -> DataFrame:
        df = DataFrame()

        df['mask_path'] = glob((mask_root_dir / '*.png').as_posix())
        df['image_path'] = df['mask_path'].map(lambda p: (image_root_dir / Path(p).name).as_posix())

        return df
