from typing import Any, Literal

from gymnasium.core import ActType, ObsType
from gymnasium.vector.vector_env import ArrayType

from ap_gym import ActivePerceptionVectorWrapper
from ap_gym.envs.image import CircleSquareDataset
from ap_gym.envs.image_classification import ImageClassificationVectorEnv
import numpy as np


class CircleSquareCatchOrFleeVectorWrapper(
    ActivePerceptionVectorWrapper[
        dict[Literal["glimpse", "glimpse_pos", "time_step"], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[Literal["glimpse", "glimpse_pos", "time_step"], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    def __init__(self, env: ImageClassificationVectorEnv):
        assert isinstance(env.config.dataset, CircleSquareDataset)
        self.__dataset: CircleSquareDataset = env.config.dataset
        super().__init__(env)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        indices = info["index"]
        positions, labels = self.__dataset.get_object_position_and_label(indices)
        sign = labels * 2 - 1
        positions_norm = (
            self.env.image_perception_module.normalize_coords(
                np.flip(positions, axis=-1)
            )
            - 1
        )
        distances = np.linalg.norm(obs["glimpse_pos"] - positions_norm, axis=-1)
        additional_reward = sign * distances
        reward += additional_reward
        info["base_reward"] += additional_reward
        return obs, reward, terminated, truncated, info
