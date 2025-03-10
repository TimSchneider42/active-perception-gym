from typing import Iterator, Generic, Any

import numpy as np

from .dataset import Dataset, DataPointBatchType


class DatasetLoader(
    Iterator[tuple[DataPointBatchType, np.ndarray]],
    Generic[DataPointBatchType],
):
    def __init__(
        self,
        dataset: Dataset[Any, DataPointBatchType],
        batch_size: int = 1,
        seed: int = 0,
    ):
        self.__dataset = dataset
        self.__rng = np.random.default_rng(seed)
        self.__batch_size = batch_size

    def __next__(self):
        idx = self.__rng.integers(0, len(self.__dataset), self.__batch_size)
        data = self.__dataset.get_data_point_batch(idx)
        return data, idx
