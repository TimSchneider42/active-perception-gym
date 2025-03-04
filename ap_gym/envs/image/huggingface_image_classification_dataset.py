from __future__ import annotations

from typing import Tuple, Union, SupportsInt

import PIL.Image
import aiohttp
import numpy as np
from datasets import load_dataset

from .image_classification_dataset import ImageClassificationDataset


class HuggingfaceImageClassificationDataset(ImageClassificationDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_feature_name: str = "image",
        label_feature_name: str = "label",
    ):
        self.__dataset_name = dataset_name
        self.__split = split
        self.__train_split = self.__data = None
        self.__image_feature_name = image_feature_name
        self.__label_feature_name = label_feature_name

    def load(self):
        dataset = load_dataset(
            self.__dataset_name,
            trust_remote_code=True,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=60 * 60 * 6)}
            },
        )
        self.__data = dataset[self.__split]
        self.__train_split = dataset["train"]

    def _get_num_classes(self) -> int:
        return self.__train_split.features[self.__label_feature_name].num_classes

    def _get_length(self) -> int:
        return len(self.__data)

    def _get_data_point(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, PIL.Image], SupportsInt]:
        data_point = self.__data[idx]
        return (
            data_point[self.__image_feature_name],
            data_point[self.__label_feature_name],
        )
