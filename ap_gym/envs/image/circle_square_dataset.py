from __future__ import annotations

import numpy as np

from .image_classification_dataset import ImageClassificationDataset


class CircleSquareDataset(ImageClassificationDataset):
    def __init__(
        self,
        show_gradient: bool = True,
        image_shape: tuple[int, int] = (28, 28),
        object_extents: int = 8,
    ):
        self.__image_shape = image_shape
        self.__object_extents = object_extents
        self.__show_gradient = show_gradient

    def _get_length(self) -> int:
        return 2 * np.prod(self.__image_shape)

    def _get_num_classes(self) -> int:
        return 2

    def _get_data_point(self, idx: int) -> tuple[np.ndarray, int]:
        label = int(idx >= np.prod(self.__image_shape))
        idx -= np.prod(self.__image_shape) * label
        pos_x = idx % self.__image_shape[1]
        pos_y = idx // self.__image_shape[1]
        position = np.array([pos_y, pos_x])
        max_dist = np.sqrt(np.sum(np.array(self.__image_shape) ** 2))

        coords = np.stack(
            np.meshgrid(
                np.arange(self.__image_shape[0]),
                np.arange(self.__image_shape[1]),
                indexing="ij",
            ),
            axis=-1,
        )
        if self.__show_gradient:
            img = 1 - np.linalg.norm(position - coords, axis=-1) / max_dist
        else:
            img = np.zeros(self.__image_shape)
        if label == 0:
            # Rectangle
            img[
                (position[0] - self.__object_extents / 2 <= coords[:, :, 0])
                & (coords[:, :, 0] <= position[0] + self.__object_extents / 2)
                & (position[1] - self.__object_extents / 2 <= coords[:, :, 1])
                & (coords[:, :, 1] <= position[1] + self.__object_extents / 2)
            ] = 1.0
        else:
            # Circle
            img[
                np.linalg.norm(position - coords, axis=-1) <= self.__object_extents / 2
            ] = 1.0
        return img[:, :, None], label
