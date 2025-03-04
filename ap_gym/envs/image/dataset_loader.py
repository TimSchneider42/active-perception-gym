import weakref
from queue import Queue, Full
from threading import Thread, Event
from typing import Iterator

import numpy as np

from .image_classification_dataset import ImageClassificationDataset


class DatasetLoader(Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    def __init__(
        self,
        image_dataset: ImageClassificationDataset,
        batch_size: int = 1,
        seed: int = 0,
    ):
        self.__image_dataset = image_dataset
        self.__rng = np.random.default_rng(seed)
        self.__batch_size = batch_size

    def __next__(self):
        idx = self.__rng.integers(0, len(self.__image_dataset), self.__batch_size)
        data = self.__image_dataset.get_data_point_batch(idx)
        return data + (idx,)


class BufferedIterator(Iterator):
    def __init__(self, iterator: Iterator, buffer_size: int = 1):
        self.__iterator = iterator
        self.__buffer = Queue(maxsize=buffer_size)
        self.__termination_signal = Event()
        self.__thread = Thread(target=self.__thread_func, daemon=True)
        weakref.finalize(self, self.close)
        self.__thread.start()

    def __next__(self):
        res = self.__buffer.get()
        if isinstance(res, Exception):
            raise res
        return res

    def close(self):
        self.__termination_signal.set()
        self.__thread.join()
        self.__thread = None

    def __thread_func(self):
        try:
            for item in self.__iterator:
                while not self.__termination_signal.is_set():
                    try:
                        self.__buffer.put(item, timeout=0.05)
                        break
                    except Full:
                        continue
        except Exception as e:
            self.__buffer.put(e)
