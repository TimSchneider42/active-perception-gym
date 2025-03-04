from __future__ import annotations

from abc import ABC
from typing import Generic, Any

import gymnasium as gym
import numpy as np

from .active_perception_env import (
    ActivePerceptionEnv,
    ActivePerceptionActionSpace,
)
from .active_perception_vector_env import (
    ActivePerceptionVectorEnv,
    FullActType,
    PredType,
)
from .loss_fn import MSELossFn
from .types import ObsType, ActType


class ActiveRegressionEnv(
    ActivePerceptionEnv[ObsType, ActType, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(self, target_dim: int, inner_action_space: gym.Space[ActType]):
        prediction_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_dim,))
        self.action_space = ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.prediction_target_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_dim,)
        )
        self.loss_fn = MSELossFn()
        self.__cumulative_dist = None
        self.__current_step = None
        self.__mse_sum = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__cumulative_dist = 0.0
        self.__current_step = 0
        self.__mse_sum = 0.0
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        target = info["prediction"]["target"]
        prediction = action["prediction"]

        # Compute Euclidean distance and MSE
        euclidean_dist = np.linalg.norm(target - prediction)
        mse = np.mean((target - prediction) ** 2)

        self.__cumulative_dist += euclidean_dist
        self.__mse_sum += mse
        self.__current_step += 1

        done = terminated or truncated
        if done:
            final_dist = euclidean_dist  # Final distance is only stored locally
            info["stats"] = {
                "avg_euclidean_distance": self.__cumulative_dist
                / max(self.__current_step, 1),
                "final_euclidean_distance": final_dist,
                "mse": self.__mse_sum / max(self.__current_step, 1),
            }

        return obs, reward, terminated, truncated, info


class ActiveRegressionVectorEnv(
    ActivePerceptionVectorEnv[ObsType, ActType, np.ndarray, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        num_envs: int,
        target_dim: int,
        single_inner_action_space: gym.Space[ActType],
    ):
        self.num_envs = num_envs
        single_prediction_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_dim,)
        )
        self.single_action_space = ActivePerceptionActionSpace(
            single_inner_action_space, single_prediction_space
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, num_envs
        )
        self.prediction_target_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_envs, target_dim)
        )
        self.loss_fn = MSELossFn()
        self.__cumulative_dist = None
        self.__current_step = None
        self.__prev_done = None
        self.__mse_sum = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__cumulative_dist = np.zeros(self.num_envs, dtype=np.float32)
        self.__current_step = np.zeros(self.num_envs, dtype=np.int32)
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__mse_sum = np.zeros(self.num_envs, dtype=np.float32)
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        target = info["prediction"]["target"]
        prediction = action["prediction"]

        # Compute Euclidean distance and MSE
        euclidean_dist = np.linalg.norm(target - prediction, axis=-1)
        mse = np.mean((target - prediction) ** 2, axis=-1)

        self.__cumulative_dist += euclidean_dist
        self.__mse_sum += mse

        self.__current_step[~self.__prev_done] += 1
        self.__cumulative_dist[self.__prev_done] = 0
        self.__mse_sum[self.__prev_done] = 0
        self.__current_step[self.__prev_done] = 0

        done = terminated | truncated
        if np.any(done):
            final_dist = euclidean_dist  # Local variable for final Euclidean distance
            avg_euclidean_distance = self.__cumulative_dist / np.maximum(
                self.__current_step, 1
            )
            info["stats"] = {
                "avg_euclidean_distance": avg_euclidean_distance,
                "_avg_euclidean_distance": done,
                "final_euclidean_distance": final_dist,
                "_final_euclidean_distance": done,
                "mse": self.__mse_sum / np.maximum(self.__current_step, 1),
                "_mse": done,
            }

        self.__prev_done = done
        return obs, reward, terminated, truncated, info
