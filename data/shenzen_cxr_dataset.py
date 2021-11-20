from glob import glob
from pathlib import Path
from typing import Any

import monai.transforms as M
from pandas import DataFrame
from torch.utils.data import Dataset


class ShenzenCXRDataset(Dataset):
    MEAN = 0.6141
    STD = 0.2590

    MASK_MAP = ['bg', 'lung', 'heart']

    def __init__(
            self,
            input_df: DataFrame,
            transforms: M.Transform = M.Compose([])
    ):
        self._df = input_df
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        img = self.get_image_path(index)
        mask = self.get_mask_path(index)

        transformed = self.transforms(dict(
            image=img,
            seg=mask
        ))

        return transformed['image'], transformed['seg']

    def __len__(self):
        return len(self._df)

    def get_image_path(self, index: int) -> Path:
        return Path(self._df['image_path'].iloc[index])

    def get_mask_path(self, index: int) -> Path:
        return Path(self._df['mask_path'].iloc[index])

    @staticmethod
    def get_dataset_df(
            image_root_dir: Path,
            mask_root_dir: Path
    ) -> DataFrame:
        df = DataFrame()

        df['mask_path'] = glob((mask_root_dir / '*.png').as_posix())
        df['image_path'] = df['mask_path'].map(lambda p: (image_root_dir / Path(p).name).as_posix())

        return df
