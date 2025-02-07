from typing import Literal, Tuple, Optional, Sequence, Union

import numpy as np

from .image_perception_vector_env import ImagePerceptionVectorEnv
from .vector_to_single_wrapper import ActivePerceptionVectorToSingleWrapper


class CircleSquareVectorEnv(ImagePerceptionVectorEnv):
    def __init__(
        self,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        show_gradient: bool = True,
        image_shape: Tuple[int, int] = (28, 28),
        shape_extents: int = 4,
        max_episode_steps: Optional[int] = None,
        max_step_length: Union[float, Sequence[float]] = 0.2,
        constraint_violation_penalty: float = 0.0,
        interpolation_method: str = "linear",
        display_visitation: bool = True,
    ):
        self.__image_shape = image_shape
        self.__object_extents = shape_extents
        self.__show_gradient = show_gradient
        super().__init__(
            num_envs,
            2 * np.prod(image_shape),
            2,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            constraint_violation_penalty=constraint_violation_penalty,
            interpolation_method=interpolation_method,
            display_visitation=display_visitation,
        )

    def _load_image(self, idx: int) -> Tuple[np.ndarray, int]:
        label = int(idx >= np.prod(self.__image_shape))
        idx -= np.prod(self.__image_shape) * label
        pos_x = idx % self.__image_shape[1]
        pos_y = idx // self.__image_shape[1]
        position = np.array([pos_y, pos_x])
        max_dist = np.sqrt(np.sum(np.array(self.__image_shape) ** 2))

        coords = np.stack(
            np.meshgrid(
                np.arange(self.__image_shape[0]),
                np.arange(self.__image_shape[1]),
                indexing="ij",
            ),
            axis=-1,
        )
        if self.__show_gradient:
            img = 1 - np.linalg.norm(position - coords, axis=-1) / max_dist
        else:
            img = np.zeros(self.__image_shape)
        if label == 0:
            # Rectangle
            img[
                (position[0] - self.__object_extents <= coords[:, :, 0])
                & (coords[:, :, 0] <= position[0] + self.__object_extents)
                & (position[1] - self.__object_extents <= coords[:, :, 1])
                & (coords[:, :, 1] <= position[1] + self.__object_extents)
            ] = 1.0
        else:
            # Circle
            img[
                np.linalg.norm(position - coords, axis=-1) <= self.__object_extents
            ] = 1.0
        return img[:, :, None], label


def CircleSquareEnv(
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    show_gradient: bool = True,
    image_shape: Tuple[int, int] = (28, 28),
    shape_extents: int = 4,
    max_episode_steps: Optional[int] = None,
    max_step_length: Union[float, Sequence[float]] = 0.2,
    constraint_violation_penalty: float = 0.0,
    interpolation_method: str = "linear",
    display_visitation: bool = True,
):
    return ActivePerceptionVectorToSingleWrapper(
        CircleSquareVectorEnv(
            1,
            render_mode=render_mode,
            show_gradient=show_gradient,
            image_shape=image_shape,
            shape_extents=shape_extents,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            constraint_violation_penalty=constraint_violation_penalty,
            interpolation_method=interpolation_method,
            display_visitation=display_visitation,
        )
    )
