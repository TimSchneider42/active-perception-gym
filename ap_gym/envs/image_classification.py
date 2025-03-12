from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from scipy.special import softmax

from ap_gym import (
    ActiveClassificationVectorEnv,
    ActivePerceptionVectorToSingleWrapper,
)
from .image import (
    ImagePerceptionModule,
    ImagePerceptionConfig,
)


class ImageClassificationVectorEnv(
    ActiveClassificationVectorEnv[
        dict[Literal["glimpse", "glimpse_pos", "time_step"], np.ndarray], np.ndarray
    ],
):
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        num_envs: int,
        image_perception_config: ImagePerceptionConfig,
        render_mode: Literal["rgb_array"] = "rgb_array",
    ):
        self.__image_perception_module = ImagePerceptionModule(
            num_envs,
            image_perception_config,
        )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.__render_mode = render_mode
        super().__init__(
            num_envs,
            image_perception_config.dataset.num_classes,
            self.__image_perception_module.single_inner_action_space,
        )
        self.single_observation_space = gym.spaces.Dict(
            self.__image_perception_module.observation_space_dict
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.__np_random = None

    def _reset(self, *, options: dict[str, Any | None] = None):
        obs, info = self.__image_perception_module.reset()
        return (
            obs,
            info,
            self.__image_perception_module.current_labels,
        )

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        prediction_quality = softmax(prediction, axis=-1)[
            np.arange(self.num_envs), self.__image_perception_module.current_labels
        ]
        (
            obs,
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
        ) = self.__image_perception_module.step(action, prediction_quality)
        return (
            obs,
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
            self.__image_perception_module.current_labels,
        )

    def render(self) -> np.ndarray | None:
        return self.__image_perception_module.render()

    def close(self):
        self.__image_perception_module.close()
        super().close()

    @property
    def render_mode(self) -> Literal["rgb_array"]:
        return self.__render_mode

    @property
    def _np_random(self):
        return self.__np_random

    @_np_random.setter
    def _np_random(self, np_random):
        self.__image_perception_module.seed(
            np_random.integers(0, 2**32 - 1, endpoint=True)
        )
        self.__np_random = np_random


def ImageClassificationEnv(
    image_perception_config: ImagePerceptionConfig,
    render_mode: Literal["rgb_array"] = "rgb_array",
):
    return ActivePerceptionVectorToSingleWrapper(
        ImageClassificationVectorEnv(
            1,
            image_perception_config,
            render_mode=render_mode,
        )
    )
