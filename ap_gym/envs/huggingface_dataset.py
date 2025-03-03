from __future__ import annotations

from typing import Tuple

import PIL.Image
import aiohttp
import numpy as np
from datasets import load_dataset


class HuggingfaceDataset:
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_feature_name: str = "image",
        label_feature_name: str = "label",
    ):
        dataset = load_dataset(
            dataset_name,
            trust_remote_code=True,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=60 * 60 * 6)}
            },
        )
        self.__data = dataset[split]
        self.__train_split = dataset["train"]
        self.__image_feature_name = image_feature_name
        self.__label_feature_name = label_feature_name

    @property
    def num_classes(self) -> int:
        return self.__train_split.features[self.__label_feature_name].num_classes

    def __getitem__(self, item) -> Tuple[np.ndarray, int]:
        data_point = self.__data[item]
        img = data_point[self.__image_feature_name]
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255
        if len(img.shape) == 2:
            img = img[..., None]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img, data_point[self.__label_feature_name]

    def __len__(self):
        return len(self.__data)
