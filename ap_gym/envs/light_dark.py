from collections import deque
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from ap_gym import ActiveRegressionEnv
from .style import COLOR_PRED, COLOR_AGENT, COLOR_OBS_PRIMARY, quality_color


class LightDarkEnv(ActiveRegressionEnv[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Literal["rgb_array"] = "rgb_array"):
        super().__init__(
            2, gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.__pos = self.__last_obs = self.__last_pred = None
        self.__light_pos = np.array([0, -0.7], dtype=np.float32)
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

        self.__trajectory = deque()

    def __compute_brightness(self, pos: np.ndarray) -> np.ndarray:
        dist_squared = (
            np.sum((pos - self.__light_pos) ** 2, axis=-1) + self.__light_height**2
        )
        return self.__light_height**2 / dist_squared

    def __get_obs(self):
        self.__last_obs = self.__pos + self.np_random.normal(size=2).astype(
            np.float32
        ) * self.__get_std_dev(self.__pos)
        self.__last_obs = np.clip(self.__last_obs, -2, 2)
        return self.__last_obs

    def __get_std_dev(self, pos: np.ndarray):
        return (1 - self.__compute_brightness(pos)) * 0.3

    def _reset(self, *, options: dict[str, Any | None] = None):
        self.__pos = self.np_random.uniform(
            -np.ones(2),
            np.ones(2),
            size=2,
        ).astype(np.float32)
        self.__trajectory.clear()
        self.__trajectory.append((self.__pos, None))
        return self.__get_obs(), {}, self.__pos

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        self.__last_pred = prediction
        base_reward = 1e-3 * np.sum(action**2, axis=-1)
        action_clipped = np.clip(action, -1, 1)
        self.__pos += action_clipped * 0.15
        terminated = False
        if np.any(np.abs(self.__pos) >= 1):
            base_reward -= 20
            terminated = True
        self.__pos = np.clip(self.__pos, -1, 1)
        prediction_quality = np.linalg.norm(prediction - self.__pos, axis=-1) / np.sqrt(
            4
        )
        self.__trajectory.append((self.__pos, prediction_quality))
        return self.__get_obs(), base_reward, terminated, False, {}, self.__pos

    def render(self):
        img = Image.fromarray(self.__base_image)  # Convert base image to PIL format
        draw = ImageDraw.Draw(img, mode="RGBA")

        dot_radius = 0.01 * img.size[0]

        pos = (self.__pos + 1) / 2 * np.array(img.size[::-1])

        std_radius = self.__get_std_dev(self.__pos) / 2 * np.array(img.size[::-1])
        draw.ellipse(
            [
                tuple(pos - std_radius),
                tuple(pos + std_radius),
            ],
            fill=COLOR_OBS_PRIMARY + (30,),
            outline=None,
        )

        traj_hist = list(self.__trajectory)
        for (pos_a, qual_a), (pos_b, qual_b) in zip(
            traj_hist[:-1], list(traj_hist)[1:]
        ):
            pos_a = (pos_a + 1) / 2 * np.array(img.size[::-1])
            pos_b = (pos_b + 1) / 2 * np.array(img.size[::-1])
            draw.line(
                (
                    pos_a[0],
                    pos_a[1],
                    pos_b[0],
                    pos_b[1],
                ),
                width=2,
                fill=quality_color(qual_b),
            )

        last_obs = (self.__last_obs + 1) / 2 * np.array(img.size[::-1])
        draw.line(
            (
                tuple(pos),
                tuple(last_obs),
            ),
            fill=COLOR_OBS_PRIMARY + (80,),
        )
        draw.ellipse(
            [
                tuple(last_obs - dot_radius),
                tuple(last_obs + dot_radius),
            ],
            fill=COLOR_OBS_PRIMARY + (100,),
            outline=None,
        )

        if self.__last_pred is not None:
            last_pred = (self.__last_pred + 1) / 2 * np.array(img.size[::-1])

            draw.line(
                (
                    tuple(pos),
                    tuple(last_pred),
                ),
                fill=COLOR_PRED + (80,),
            )

            draw.ellipse(
                [
                    tuple(last_pred - dot_radius),
                    tuple(last_pred + dot_radius),
                ],
                fill=COLOR_PRED + (100,),
                outline=None,
            )

        draw.ellipse(
            [
                tuple(pos - dot_radius),
                tuple(pos + dot_radius),
            ],
            fill=COLOR_AGENT,
            outline=None,
        )

        return np.array(img)
