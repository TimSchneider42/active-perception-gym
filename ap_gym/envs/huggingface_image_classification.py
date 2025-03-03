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
        max_episode_steps: int | None = None,
        max_step_length: float = 0.2,
        prefetch: bool = True,
    ):
        dataset = load_dataset(
            dataset_name,
            trust_remote_code=True,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=60 * 60 * 6)}
            },
        )
        self.__data = dataset[split]

        super().__init__(
            num_envs,
            len(self.__data),
            dataset["train"].features["label"].num_classes,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            prefetch=prefetch,
        )

    def _load_image(self, idx: int) -> tuple[np.ndarray, int]:
        data_point = self.__data[idx]
        img = data_point["image"]
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255
        if len(img.shape) == 2:
            img = img[..., None]
        return img, data_point["label"]


def HuggingfaceImageClassificationEnv(
    dataset_name: str,
    split: str = "train",
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    max_episode_steps: int | None = None,
    max_step_length: float = 0.2,
):
    return ActivePerceptionVectorToSingleWrapper(
        HuggingfaceImageClassificationVectorEnv(
            1,
            dataset_name,
            split,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            prefetch=False,  # Must be false as we cannot predict the seeds of the reset calls
        )
    )
