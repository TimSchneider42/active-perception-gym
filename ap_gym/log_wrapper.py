from __future__ import annotations

from collections import deque, defaultdict
from typing import Any, Generic, SupportsFloat

import numpy as np

from .active_perception_env import ActivePerceptionWrapper
from .active_perception_vector_env import ActivePerceptionVectorWrapper
from .types import ObsType, ActType, PredType, PredTargetType, FullActType
from .util import update_info_metrics, update_info_metrics_vec


class ActivePerceptionLogWrapper(
    ActivePerceptionWrapper[
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ObsType,
        ActType,
        PredType,
        PredTargetType,
    ],
    Generic[ObsType, ActType, PredType, PredTargetType],
):
    """Accumulates per-step metrics over an episode and reports them on termination."""

    def __init__(self, env):
        super().__init__(env)
        self.__metrics: dict[str, deque] | None = None

    def _step_metrics(
        self, action: FullActType[ActType, PredType], info: dict[str, Any]
    ) -> dict[str, Any]:
        """Metrics of this step, accumulated alongside the loss function's own."""
        return {}

    def _episode_metrics(self, metrics: dict[str, deque]) -> dict[str, Any]:
        """Scalars derived from the accumulated episode, reported as they are."""
        return {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__metrics = defaultdict(deque)
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        step_metrics = {
            **info["prediction"]["metrics"],
            **self._step_metrics(action, info),
        }
        for name, value in step_metrics.items():
            self.__metrics[name].append(value)
        if terminated or truncated:
            info = update_info_metrics(info, self.__metrics)
            info["stats"]["scalar"].update(self._episode_metrics(self.__metrics))
        return obs, reward, terminated, truncated, info


class ActivePerceptionVectorLogWrapper(
    ActivePerceptionVectorWrapper,
    Generic[ObsType, ActType, PredType, PredTargetType],
):
    """
    The vector counterpart of ActivePerceptionLogWrapper.

    Sub-environments restart individually without a reset(), so each accumulator is
    cleared on its own previous done. The step that follows a done carries the reset
    observation, so its metrics belong to no episode and are dropped.
    """

    def __init__(self, env):
        super().__init__(env)
        self.__prev_done: np.ndarray | None = None
        self.__metrics: dict[str, tuple[deque, ...]] | None = None

    def _step_metrics(
        self, action: FullActType[ActType, PredType], info: dict[str, Any]
    ) -> dict[str, Any]:
        """Metrics of this step per sub-environment, accumulated alongside the loss function's own."""
        return {}

    def _episode_metrics(self, metrics: dict[str, tuple[deque, ...]]) -> dict[str, Any]:
        """Scalars derived from the accumulated episodes, reported as they are."""
        return {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        step_metrics = {
            **info["prediction"]["metrics"],
            **self._step_metrics(action, info),
        }
        for name, values in step_metrics.items():
            per_env = self.__metrics[name]
            for i in range(self.num_envs):
                if self.__prev_done[i]:
                    per_env[i].clear()
                else:
                    per_env[i].append(values[i])
        self.__prev_done = terminated | truncated
        if np.any(self.__prev_done):
            info = update_info_metrics_vec(info, self.__metrics, self.__prev_done)
            info["stats"]["scalar"].update(self._episode_metrics(self.__metrics))
        return obs, reward, terminated, truncated, info
