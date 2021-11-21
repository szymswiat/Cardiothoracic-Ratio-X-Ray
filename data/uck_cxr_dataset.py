from pathlib import Path
from typing import Any

import numpy as np
from glob import glob
from torch.utils.data import Dataset
from pandas import DataFrame
import pydicom as pyd
import monai.transforms as M


class UCKCXRDataset(Dataset):

    def __init__(
            self,
            input_df: DataFrame,
            transforms: M.Compose = M.Compose([])
    ):
        self._df = input_df
        self._transforms = transforms

    def __getitem__(self, index: int) -> Any:
        img_path = self.get_image_path(index).as_posix()

        dcm = pyd.dcmread(img_path)

        pix = np.repeat(np.expand_dims(dcm.pixel_array, axis=-1), repeats=3, axis=-1)

        aug_data = self._transforms(pix)

        return aug_data

    def get_image_path(self, index: int) -> Path:
        return Path(self._df['img_path'].iloc[index])

    @staticmethod
    def get_dataset_df(images_root_dir: Path) -> DataFrame:
        glob_pattern = images_root_dir / '*.dcm'
        img_paths = glob(glob_pattern.as_posix())

        return DataFrame(data=img_paths, columns=['img_path'])
