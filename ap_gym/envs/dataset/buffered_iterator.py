import weakref
from queue import Queue, Full
from threading import Thread, Event
from typing import Iterator, Generic, TypeVar

InnerIteratorType = TypeVar("InnerIteratorType")


class BufferedIterator(Iterator[InnerIteratorType], Generic[InnerIteratorType]):
    def __init__(self, iterator: Iterator[InnerIteratorType], buffer_size: int = 1):
        self.__iterator = iterator
        self.__buffer = Queue(maxsize=buffer_size)
        self.__termination_signal = Event()
        self.__thread = Thread(target=self.__thread_func, daemon=True)
        weakref.finalize(self, self.close)
        self.__thread.start()

    def __next__(self) -> InnerIteratorType:
        res = self.__buffer.get()
        if isinstance(res, Exception):
            raise res
        return res

    def close(self):
        if self.__thread is not None:
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
                else:
                    break
        except Exception as e:
            self.__buffer.put(e)
