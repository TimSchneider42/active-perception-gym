from __future__ import annotations

from typing import Any, Literal, Sequence

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Resampling
from scipy.interpolate import RegularGridInterpolator
from scipy.special import softmax

from ap_gym import (
    ActiveClassificationVectorEnv,
    ImageSpace,
    ActivePerceptionVectorToSingleWrapper,
)
from .image import ImageClassificationDataset
from .image.dataset_loader import DatasetLoader, BufferedIterator


class ImageClassificationVectorEnv(
    ActiveClassificationVectorEnv[
        dict[Literal["glance", "glance_pos"], np.ndarray], np.ndarray
    ],
):
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        num_envs: int,
        dataset: ImageClassificationDataset,
        render_mode: Literal["rgb_array"] = "rgb_array",
        sensor_size: tuple[int, int] = (5, 5),
        sensor_scale: float = 1.0,
        max_episode_steps: int | None = None,
        max_step_length: float | Sequence[float] = 0.2,
        display_visitation: bool = True,
        prefetch: bool = True,
    ):
        self.__dataset = dataset
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if max_episode_steps is None:
            max_episode_steps = 16
        # Target position of the sensor relative to the previous position of the sensor
        single_inner_action_space = gym.spaces.Box(
            -np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)
        )
        self.__sensor_size = sensor_size
        self.__sensor_scale = sensor_scale
        self.__render_mode = render_mode
        self.__current_data_point_idx: int | None = None
        self.__current_images: np.ndarray | None = None
        self.__interpolated_images: RegularGridInterpolator | None = None
        self.__display_visitation = display_visitation
        self.__prefetch = prefetch
        self.__prefetch_buffer_size = 128
        self.__terminating = None
        self.__dataset.load()
        super().__init__(
            num_envs, self.__dataset.num_classes, single_inner_action_space
        )
        *self.__image_size, self.__channels = self.__dataset[0][0].shape
        self.single_observation_space = gym.spaces.Dict(
            {
                "glance": ImageSpace(
                    self.__sensor_size[1],
                    self.__sensor_size[0],
                    self.__channels,
                    dtype=np.float32,
                ),
                "glance_pos": gym.spaces.Box(-1, 1, (2,), np.float32),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.__current_sensor_pos_norm: np.ndarray | None = None
        self.__current_time_step = None
        self.__max_steps = max_episode_steps
        max_step_length = np.array(max_step_length)
        assert max_step_length.shape in {(2,), (1,), ()}
        self.__max_step_length = np.ones(2) * np.array(max_step_length)
        self.__current_rng = self.__sample_rng = None
        self.__render_size = self.__render_scaling = None
        self.__visitation_counts = self.__last_prediction_map = None
        self.__last_prediction = np.zeros(
            (self.num_envs, self.__dataset.num_classes), dtype=np.int32
        )
        self.__prev_done: np.ndarray | None = None
        self.__current_labels: np.ndarray | None = None
        self.__data_loader: DatasetLoader | BufferedIterator | None = None

    def _reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        self.__current_rng = np.random.default_rng(seed)
        self.__data_loader = DatasetLoader(
            self.__dataset,
            batch_size=self.num_envs,
            seed=self.__current_rng.integers(0, 2**32),
        )
        if self.__prefetch:
            self.__data_loader = BufferedIterator(
                self.__data_loader, buffer_size=self.__prefetch_buffer_size
            )
        return self.__reset()

    def __reset(self):
        (
            self.__current_images,
            self.__current_labels,
            self.__current_data_point_idx,
        ) = next(self.__data_loader)
        image_size = np.array(self.__current_images.shape[1:3])
        if np.any(image_size < self.effective_sensor_size):
            raise ValueError(
                f"Image size {tuple(image_size)} cannot be smaller than effective sensor size "
                f"{tuple(self.effective_sensor_size)}."
            )
        coords_y = (
            np.arange(0, self.__current_images.shape[1])
            - (self.__current_images.shape[1] - 1) / 2
        )
        coords_x = (
            np.arange(0, self.__current_images.shape[2])
            - (self.__current_images.shape[2] - 1) / 2
        )
        self.__interpolated_images = [
            RegularGridInterpolator((coords_y, coords_x), img, method="linear")
            for img in self.__current_images
        ]

        self.__current_sensor_pos_norm = self.__current_rng.uniform(
            -1, 1, size=(self.num_envs, 2)
        )
        info = {"index": self.__current_data_point_idx}
        self.__current_time_step = 0
        self.__last_prediction.fill(0)

        obs = self._get_obs()

        if self.__visitation_counts is None:
            render_width = max(128, obs["glance"].shape[2])
            self.__render_scaling = render_width / self.__image_size[1]
            render_height = int(round(self.__render_scaling * self.__image_size[0]))
            self.__render_size = (render_width, render_height)
            self.__visitation_counts = np.zeros(
                (self.num_envs, self.__render_size[1], self.__render_size[0]),
                dtype=np.int32,
            )
            self.__last_prediction_map = np.zeros(
                (
                    self.num_envs,
                    self.__render_size[1],
                    self.__render_size[0],
                    self.single_prediction_target_space.n,
                ),
                dtype=np.int32,
            )
        else:
            self.__visitation_counts.fill(0)
            self.__last_prediction_map.fill(0)

        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        return obs, info, self.__current_labels

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        self._update_visitation_overlay(prediction=prediction)
        if np.any(self.__prev_done):
            if not np.all(self.__prev_done):
                raise NotImplementedError("Partial reset is not supported.")
            obs, info, self.__current_labels = self.__reset()
            terminated = False
            base_reward = np.zeros(self.num_envs)
        else:
            action_clipped = np.clip(action, -1, 1)
            if np.any(np.isnan(action_clipped)):
                raise ValueError("NaN values detected in action.")
            step = self.__max_step_length * action_clipped
            new_sensor_pos_norm = self.__current_sensor_pos_norm + step
            self.__current_sensor_pos_norm = np.clip(new_sensor_pos_norm, -1, 1)
            base_reward = -np.linalg.norm(action, axis=-1) * 1e-3
            info = {"index": self.__current_data_point_idx}
            self.__current_time_step += 1
            terminated = self.__current_time_step >= self.__max_steps
            obs = self._get_obs()
        terminated_arr = np.full(self.num_envs, terminated)
        truncated_arr = np.zeros(self.num_envs, dtype=np.bool_)
        self.__prev_done = terminated_arr | truncated_arr
        self.__last_prediction = prediction
        return (
            obs,
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
            self.__current_labels,
        )

    def _update_visitation_overlay(self, prediction: np.ndarray | None = None):
        pos, size = self.__sensor_rects
        pos = np.round(pos).astype(np.int32)
        size = np.round(np.flip(size)).astype(np.int32)
        x_range = pos[..., 0, np.newaxis] + np.arange(size[0]) - size[0] // 2
        y_range = pos[..., 1, np.newaxis] + np.arange(size[1]) - size[1] // 2
        coords = (
            np.arange(self.num_envs)[:, np.newaxis, np.newaxis],
            np.clip(y_range, 0, self.__visitation_counts.shape[1] - 1)[
                :, :, np.newaxis
            ],
            np.clip(x_range, 0, self.__visitation_counts.shape[2] - 1)[
                :, np.newaxis, :
            ],
        )
        self.__visitation_counts[coords] += 1
        if prediction is not None:
            self.__last_prediction_map[coords] = prediction[:, np.newaxis, np.newaxis]

    def _get_obs(self) -> dict[Literal["glance", "glance_pos"], np.ndarray]:
        sensing_point_offsets = np.meshgrid(
            (np.arange(self.__sensor_size[0]) - (self.__sensor_size[0] - 1) / 2)
            * self.__sensor_scale,
            (np.arange(self.__sensor_size[1]) - (self.__sensor_size[1] - 1) / 2)
            * self.__sensor_scale,
            indexing="ij",
        )
        sensing_points = (
            np.flip(self.current_sensor_pos, axis=-1)[:, np.newaxis, np.newaxis]
            + np.stack(sensing_point_offsets, axis=-1)[np.newaxis]
        )
        sensor_img = np.stack(
            [img(sp) for img, sp in zip(self.__interpolated_images, sensing_points)],
            axis=0,
        ).clip(0, 1)
        return {
            "glance": sensor_img.astype(np.float32),
            "glance_pos": self.__current_sensor_pos_norm.astype(np.float32),
        }

    def render(self) -> np.ndarray | None:
        current_image = self.__current_images
        if self.__channels == 1:
            current_image = current_image[..., 0]
        elif self.__channels != 3:
            raise NotImplementedError()
        rgb_imgs = []
        pos, size = self.__sensor_rects
        top_left = pos - size / 2
        bottom_right = pos + size / 2

        glance_shadow_color = (0, 0, 0, 80)
        glance_border_width = max(1, int(round(1 / 128 * self.__render_size[0])))
        glance_border_color = (0, 55, 255)
        glance_shadow_offset = glance_border_width

        visited = self.__visitation_counts > 0
        correct_color = np.array([0, 255, 0, 80], dtype=np.uint8)
        incorrect_color = np.array([255, 0, 0, 80], dtype=np.uint8)
        last_prediction_map_probs = softmax(self.__last_prediction_map, axis=-1)
        correct_prob_map = last_prediction_map_probs[
            np.arange(self.num_envs), ..., self.__current_labels
        ]
        overlay = visited[..., None] * (
            correct_prob_map[..., np.newaxis]
            * correct_color[np.newaxis, np.newaxis, np.newaxis, :]
            + (1 - correct_prob_map[..., np.newaxis])
            * incorrect_color[np.newaxis, np.newaxis, np.newaxis, :]
        ).round().astype(np.uint8)
        for img, tl, br, ol in zip(current_image, top_left, bottom_right, overlay):
            rgb_img = (
                Image.fromarray((img * 255).astype(np.uint8))
                .resize(self.__render_size, resample=Resampling.NEAREST)
                .convert("RGB")
            )

            if self.__display_visitation:
                # Unfortunately, we cannot use Pillows alpha_composite here because it does not support RBG base images.
                # We cannot change the base image to RGBA because of a bug in Pillow that prevents the rectangle from
                # being drawn correctly. See: https://github.com/python-pillow/Pillow/issues/2496
                # So we do it manually here.
                alpha = ol[..., -1:] / 255
                rgb_img = Image.fromarray(
                    (np.array(rgb_img) * (1 - alpha) + alpha * ol[..., :-1]).astype(
                        np.uint8
                    )
                )

            draw = ImageDraw.Draw(rgb_img, "RGBA")
            glance_coords = np.concatenate([tl, br])
            draw.rectangle(
                tuple(glance_coords + glance_shadow_offset),
                outline=glance_shadow_color,
                width=glance_border_width,
            )
            draw.rectangle(
                tuple(glance_coords),
                outline=glance_border_color,
                width=glance_border_width,
            )
            rgb_imgs.append(rgb_img)
        rgb_img = np.stack(rgb_imgs, axis=0)

        return np.array(rgb_img)

    def close(self):
        if isinstance(self.__data_loader, BufferedIterator):
            self.__data_loader.close()

    @property
    def render_mode(self) -> Literal["rgb_array", "human"]:
        return self.__render_mode

    @property
    def sensor_size(self) -> tuple[int, int]:
        return self.__sensor_size

    @property
    def image_size(self) -> tuple[int, int]:
        return self.__image_size

    @property
    def effective_sensor_size(self):
        return np.array(self.__sensor_size) * self.__sensor_scale

    @property
    def current_sensor_pos(self):
        sensor_pos_lim = (
            np.flip(np.array(self.__current_images.shape[1:3])) - 1
        ) / 2 - (self.effective_sensor_size - 1) / 2
        return sensor_pos_lim * self.__current_sensor_pos_norm

    @property
    def __sensor_rects(self):
        pos = (
            self.current_sensor_pos * self.__render_scaling
            + np.array(self.__render_size) / 2
        )
        size = (
            np.flip(np.array(self.__sensor_size))
            * self.__sensor_scale
            * self.__render_scaling
        )
        return pos, size


def ImageClassificationEnv(
    dataset: ImageClassificationDataset,
    render_mode: Literal["rgb_array"] = "rgb_array",
    sensor_size: tuple[int, int] = (5, 5),
    sensor_scale: float = 1.0,
    max_episode_steps: int | None = None,
    max_step_length: float | Sequence[float] = 0.2,
    display_visitation: bool = True,
):
    return ActivePerceptionVectorToSingleWrapper(
        ImageClassificationVectorEnv(
            1,
            dataset,
            render_mode=render_mode,
            sensor_size=sensor_size,
            sensor_scale=sensor_scale,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            display_visitation=display_visitation,
            prefetch=False,
        )
    )
