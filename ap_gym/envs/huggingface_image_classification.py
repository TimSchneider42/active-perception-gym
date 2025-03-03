from __future__ import annotations

import logging
from typing import Literal

import PIL.Image
import aiohttp
import numpy as np
from datasets import load_dataset

from ap_gym import ImageClassificationVectorEnv, ActivePerceptionVectorToSingleWrapper

logger = logging.getLogger(__name__)


class HuggingfaceImageClassificationVectorEnv(ImageClassificationVectorEnv):
    def __init__(
        self,
        num_envs: int,
        dataset_name: str,
        split: str = "train",
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        sensor_size: tuple[int, int] = (5, 5),
        max_episode_steps: int | None = None,
        max_step_length: float = 0.2,
        prefetch: bool = True,
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
        self.__image_feature_name = image_feature_name
        self.__label_feature_name = label_feature_name

        super().__init__(
            num_envs,
            len(self.__data),
            dataset["train"].features[self.__label_feature_name].num_classes,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            prefetch=prefetch,
            sensor_size=sensor_size,
        )

    def _load_image(self, idx: int) -> tuple[np.ndarray, int]:
        data_point = self.__data[idx]
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


def HuggingfaceImageClassificationEnv(
    dataset_name: str,
    split: str = "train",
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    sensor_size: tuple[int, int] = (5, 5),
    max_episode_steps: int | None = None,
    max_step_length: float = 0.2,
    image_feature_name: str = "image",
    label_feature_name: str = "label",
):
    return ActivePerceptionVectorToSingleWrapper(
        HuggingfaceImageClassificationVectorEnv(
            1,
            dataset_name,
            split,
            render_mode=render_mode,
            sensor_size=sensor_size,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            prefetch=False,  # Must be false as we cannot predict the seeds of the reset calls
            image_feature_name=image_feature_name,
            label_feature_name=label_feature_name,
        )
    )
