from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from monai.data import ImageReader
from monai.data.image_reader import _copy_compatible_dict, _stack_images
from monai.utils import ensure_tuple
import cv2


class CV2Reader(ImageReader):

    def __init__(self, converter: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.converter = converter
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        suffixes: Sequence[str] = ["png", "jpg", "jpeg", "bmp"]
        filename = Path(filename)

        return filename.suffix in suffixes

    def read(self, data: Union[Sequence[str], str, np.ndarray], **kwargs):
        img_: List[np.ndarray] = []

        filenames: Sequence[str] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            if callable(self.converter):
                img = self.converter(img)
            img_.append(img)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        header = self._get_meta_dict(img)
        header["spatial_shape"] = self._get_spatial_shape(img)
        # data = np.moveaxis(np.asarray(img), 0, 1)
        img_array.append(img)
        header["original_channel_dim"] = "no_channel" if len(img.shape) == len(header["spatial_shape"]) else -1
        _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.
        Args:
            img: a CV2 Image object loaded from an image file.

        """
        return {
            "width": img.shape[1],
            "height": img.shape[0],
        }

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        return np.asarray(img.shape[0:2])
