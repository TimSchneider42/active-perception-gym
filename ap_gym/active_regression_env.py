from __future__ import annotations

from abc import ABC
from collections import deque, defaultdict
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
from .util import update_info_metrics, update_info_metrics_vec


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
        self.__current_step = None
        self.__metrics: dict[str, deque[float]] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__metrics = defaultdict(deque)
        self.__current_step = 0
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

        self.__metrics["euclidean_distance"].append(euclidean_dist)
        self.__metrics["mse"].append(mse)
        self.__current_step += 1

        if terminated:
            info = update_info_metrics(info, self.__metrics)

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
        self.__current_step = None
        self.__prev_done = None
        self.__metrics: dict[str, tuple[deque[float], ...]] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__current_step = np.zeros(self.num_envs, dtype=np.int32)
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
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

        for i in range(self.num_envs):
            if self.__prev_done[i]:
                self.__metrics["euclidean_distance"][i].clear()
                self.__metrics["mse"][i].clear()
            else:
                self.__metrics["euclidean_distance"][i].append(euclidean_dist[i])
                self.__metrics["mse"][i].append(mse[i])

        self.__current_step[~self.__prev_done] += 1
        self.__current_step[self.__prev_done] = 0

        if np.any(terminated):
            info = update_info_metrics_vec(info, self.__metrics, terminated)
        self.__prev_done = terminated | truncated
        return obs, reward, terminated, truncated, info
