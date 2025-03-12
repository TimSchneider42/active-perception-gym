from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from PIL import ImageDraw

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    ActiveRegressionVectorEnv,
    ImageSpace,
)
from .image import (
    ImagePerceptionModule,
    ImagePerceptionConfig,
)
from .style import COLOR_PRED


class ImageLocalizationVectorEnv(
    ActiveRegressionVectorEnv[
        dict[
            Literal["glimpse", "glimpse_pos", "time_step", "target_glimpse"], np.ndarray
        ],
        np.ndarray,
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
            2,
            self.__image_perception_module.single_inner_action_space,
        )
        self.single_observation_space = gym.spaces.Dict(
            {
                **self.__image_perception_module.observation_space_dict,
                "target_glimpse": ImageSpace(
                    image_perception_config.sensor_size[1],
                    image_perception_config.sensor_size[0],
                    image_perception_config.dataset[0][0].shape[-1],
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.__current_prediction_target = None
        self.__prev_done = None
        self.__last_prediction = None
        self.__np_random = None

    def _reset(self, *, options: dict[str, Any | None] = None):
        self.__last_prediction = None
        obs, info = self.__image_perception_module.reset()
        self.__current_prediction_target = self.np_random.uniform(
            -1, 1, (self.num_envs, 2)
        ).astype(np.float32)
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        return (
            {
                **obs,
                "target_glimpse": self.__image_perception_module.get_glimpse(
                    self.__current_prediction_target
                ),
            },
            info,
            self.__current_prediction_target,
        )

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        if np.any(self.__prev_done):
            self.__current_prediction_target[self.__prev_done] = self.np_random.uniform(
                -1, 1, (np.sum(self.__prev_done), 2)
            ).astype(np.float32)
        prediction_quality = 1 - np.linalg.norm(
            prediction - self.__current_prediction_target, axis=-1
        ) / np.sqrt(4)
        self.__last_prediction = prediction
        (
            obs,
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
        ) = self.__image_perception_module.step(action, prediction_quality)
        self.__prev_done = terminated_arr | truncated_arr
        return (
            {
                **obs,
                "target_glimpse": self.__image_perception_module.get_glimpse(
                    self.__current_prediction_target
                ),
            },
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
            self.__current_prediction_target,
        )

    def render(self) -> np.ndarray | None:
        imgs = self.__image_perception_module.render(return_pil_imgs=True)
        last_prediction = self.__last_prediction
        if last_prediction is None:
            last_prediction = [None] * self.num_envs

        glimpse_size = (
            self.__image_perception_module.effective_sensor_size
            * self.__image_perception_module.render_scaling
        )
        target_color = COLOR_PRED + (100,)

        for img, last_pred, target in zip(
            imgs, last_prediction, self.__current_prediction_target
        ):
            draw = ImageDraw.Draw(img, "RGBA")
            t_trans = self.__image_perception_module.to_render_coords(target)
            draw.rectangle(
                (tuple(t_trans - glimpse_size / 2), tuple(t_trans + glimpse_size / 2)),
                outline=target_color,
                width=self.__image_perception_module.glimpse_border_width,
            )
            if last_pred is not None:
                lp_trans = self.__image_perception_module.to_render_coords(last_pred)
                lp_coords = np.concatenate(
                    [lp_trans - glimpse_size / 2, lp_trans + glimpse_size / 2]
                )
                draw.rectangle(
                    tuple(lp_coords),
                    outline=COLOR_PRED,
                    width=self.__image_perception_module.glimpse_border_width,
                )
                draw.rectangle(
                    tuple(
                        lp_coords + self.__image_perception_module.glimpse_border_width
                    ),
                    outline=(0, 0, 0, 80),
                    width=self.__image_perception_module.glimpse_border_width,
                )

        return np.asarray(imgs)

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


def ImageLocalizationEnv(
    image_perception_config: ImagePerceptionConfig,
    render_mode: Literal["rgb_array"] = "rgb_array",
):
    return ActivePerceptionVectorToSingleWrapper(
        ImageLocalizationVectorEnv(
            1,
            image_perception_config,
            render_mode=render_mode,
        )
    )
