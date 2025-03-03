from __future__ import annotations

from typing import Literal

import numpy as np

from ap_gym import ImageClassificationVectorEnv, ActivePerceptionVectorToSingleWrapper
from .huggingface_dataset import HuggingfaceDataset


class HuggingfaceImageClassificationVectorEnv(ImageClassificationVectorEnv):
    def __init__(
        self,
        num_envs: int,
        dataset: HuggingfaceDataset,
        render_mode: Literal["rgb_array"] = "rgb_array",
        sensor_size: tuple[int, int] = (5, 5),
        max_episode_steps: int | None = None,
        max_step_length: float = 0.2,
        prefetch: bool = True,
    ):
        self.__dataset = dataset
        super().__init__(
            num_envs,
            len(dataset),
            dataset.num_classes,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            prefetch=prefetch,
            sensor_size=sensor_size,
        )

    def _load_image(self, idx: int) -> tuple[np.ndarray, int]:
        return self.__dataset[idx]


def HuggingfaceImageClassificationEnv(
    dataset: HuggingfaceDataset,
    render_mode: Literal["rgb_array"] = "rgb_array",
    sensor_size: tuple[int, int] = (5, 5),
    max_episode_steps: int | None = None,
    max_step_length: float = 0.2,
):
    return ActivePerceptionVectorToSingleWrapper(
        HuggingfaceImageClassificationVectorEnv(
            1,
            dataset,
            render_mode=render_mode,
            sensor_size=sensor_size,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            prefetch=False,  # Must be false as we cannot predict the seeds of the reset calls
        )
    )
