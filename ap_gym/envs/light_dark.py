from typing import Any, Literal

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from ap_gym import MSELossFn, ActiveRegressionEnv


class LightDarkEnv(ActiveRegressionEnv[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Literal["rgb_array"] = "rgb_array"):
        super().__init__(2, gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32))
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.__pos = self.__last_obs = self.__rng = self.__last_pred = None
        self.__light_pos = np.array([0, -0.8], dtype=np.float32)
        self.__light_height = 0.2

        res = 500
        coords_x, coords_y = np.meshgrid(
            np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing="ij"
        )
        brightness = self.__compute_brightness(np.stack([coords_y, coords_x], axis=-1))
        ambient_light = 0.1
        self.__base_image = np.broadcast_to(
            ((brightness * 0.9 + ambient_light) * 255).astype(np.uint8)[..., None],
            (res, res, 3),
        )

        self.observation_space = gym.spaces.Box(
            low=-2, high=2, shape=(2,), dtype=np.float32
        )

    def __compute_brightness(self, pos: np.ndarray) -> np.ndarray:
        dist_squared = np.sum(
            (pos - self.__light_pos) ** 2 + self.__light_height**2, axis=-1
        )
        return self.__light_height**2 / dist_squared

    def __get_obs(self):
        self.__last_obs = self.__pos + self.__rng.normal(size=2).astype(
            np.float32
        ) * self.__get_std_dev(self.__pos)
        self.__last_obs = np.clip(self.__last_obs, -2, 2)
        return self.__last_obs

    def __get_std_dev(self, pos: np.ndarray):
        return (1 - self.__compute_brightness(pos)) * 0.3

    def _reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        self.__rng = np.random.default_rng(seed)
        self.__pos = self.__rng.uniform(
            np.array([0.0, -1.0]),
            np.ones(2),
            size=2,
        ).astype(np.float32)
        return self.__get_obs(), {}, self.__pos

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        self.__last_pred = prediction
        base_reward = np.sum(action**2, axis=-1)
        action_clipped = np.clip(action, -1, 1)
        self.__pos += action_clipped * 0.15
        terminated = False
        if np.any(np.abs(self.__pos) >= 1):
            base_reward -= 20
            terminated = True
        self.__pos = np.clip(self.__pos, -1, 1)
        return self.__get_obs(), base_reward, terminated, False, {}, self.__pos

    def render(self):
        img = Image.fromarray(self.__base_image)  # Convert base image to PIL format
        draw = ImageDraw.Draw(img, mode="RGBA")

        base_color = (55, 255, 0)
        pred_color = (255, 55, 0)
        dot_radius = 0.01 * img.size[0]

        pos = (self.__pos + 1) / 2 * np.array(img.size[::-1])
        draw.ellipse(
            [
                tuple(pos - dot_radius),
                tuple(pos + dot_radius),
            ],
            fill=base_color,
            outline=None,
        )

        std_radius = self.__get_std_dev(self.__pos) / 2 * np.array(img.size[::-1])
        draw.ellipse(
            [
                tuple(pos - std_radius),
                tuple(pos + std_radius),
            ],
            fill=base_color + (30,),
            outline=None,
        )

        last_obs = (self.__last_obs + 1) / 2 * np.array(img.size[::-1])
        draw.ellipse(
            [
                tuple(last_obs - dot_radius),
                tuple(last_obs + dot_radius),
            ],
            fill=base_color + (100,),
            outline=None,
        )

        if self.__last_pred is not None:
            last_pred = (self.__last_pred + 1) / 2 * np.array(img.size[::-1])
            draw.ellipse(
                [
                    tuple(last_pred - dot_radius),
                    tuple(last_pred + dot_radius),
                ],
                fill=pred_color + (100,),
                outline=None,
            )

        return np.array(img)
