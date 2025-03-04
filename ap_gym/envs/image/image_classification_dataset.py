from __future__ import annotations

from abc import ABC, abstractmethod
from typing import SupportsInt, Sequence, overload

import PIL.Image
import numpy as np


class ImageClassificationDataset(ABC):
    def load(self):
        pass

    @abstractmethod
    def _get_num_classes(self) -> int:
        pass

    @abstractmethod
    def _get_length(self) -> int:
        pass

    def _get_data_point(self, idx: int) -> tuple[np.ndarray | PIL.Image, SupportsInt]:
        # TODO: check impl
        imgs, labels = self._get_data_point_batch(np.array([idx]))
        return next(iter(imgs)), next(iter(labels))

    def _get_data_point_batch(
        self, idx: np.ndarray
    ) -> tuple[Sequence[np.ndarray] | Sequence[PIL.Image] | np.ndarray, Sequence[SupportsInt]]:
        # TODO: check impl
        return tuple(map(list, zip(*(self._get_data_point(int(i)) for i in idx))))

    def get_data_point(self, idx: SupportsInt) -> tuple[np.ndarray, int]:
        img, label = self._get_data_point(int(idx))
        return self._process_img(img), int(label)

    def get_data_point_batch(
        self, idx: Sequence[SupportsInt] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(idx)
        if idx.shape[0] == 0:
            raise ValueError("Empty index array")
        imgs, labels = self._get_data_point_batch(idx)
        return self._process_img_batch(imgs), np.asarray(labels).astype(np.int32)

    def _process_img(self, img: PIL.Image | np.ndarray) -> np.ndarray:
        return self._process_img_batch([img])[0]

    def _process_img_batch(
        self, imgs: Sequence[np.ndarray] | Sequence[PIL.Image] | np.ndarray
    ) -> np.ndarray:
        imgs = np.asarray(imgs)
        if imgs.dtype == np.uint8:
            imgs = imgs.astype(np.float32) / 255
        elif imgs.dtype != np.float32:
            imgs = imgs.astype(np.float32)
        if len(imgs.shape) == 3:
            imgs = imgs[..., None]
        if imgs.shape[-1] == 1:
            imgs = np.repeat(imgs, 3, axis=-1)
        return imgs

    @property
    def num_classes(self) -> int:
        return self._get_num_classes()

    @overload
    def __getitem__(self, item: SupportsInt) -> tuple[np.ndarray, int]:
        ...

    @overload
    def __getitem__(self, item: Sequence[SupportsInt]) -> tuple[np.ndarray, np.ndarray]:
        ...

    def __getitem__(self, item: int | Sequence[int] | np.ndarray):
        if isinstance(item, Sequence) or isinstance(item, np.ndarray):
            return self.get_data_point_batch(item)
        else:
            return self.get_data_point(item)

    def __len__(self):
        return self._get_length()
